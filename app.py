import os
import io
import re
import json
import uuid
import time
import math
import fitz  # PyMuPDF
import altair as alt
import requests
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple

# ---- OCR deps (optional at import time; handled in code too)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# ===================== Config & Helpers =====================

st.set_page_config(page_title="Optimizare Cheltuieli", layout="wide")

APP_DATA_DIR = Path(os.getenv("APP_DATA_DIR", "./data"))
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_NAME = "transactions.csv"         # fallback local
CSV_PATH = APP_DATA_DIR / CSV_NAME

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")

def sb_enabled() -> bool:
    return bool(SUPABASE_URL) and bool(SUPABASE_KEY)

def sb_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def sb_table_url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


# ===================== Regex / Parsing ======================

META_RE = re.compile(
    r"""
    (?:Booking\s*Info|
      Booking\s*Date|Valuation\s*Date|Transaction\s*Date|Booking\s*Reference|
      Merchant|Location|Card\s*Number|Virtual\s*card\s*number|Card|Paid\s*with|
      Exchange\s*rate|Comision|Commission|
      Apple\s*Pay|Tranzac(?:t?ie|tie)\s*comerciant|
      Data(?:[:\.\s\w]*Ora)?|
      Ref(?:erence)?|
      Suma\s*platita|Suma\s*decontata|
      Rata\s*de\s*schimb|
      Locat(?:ie|ția)|Loca(?:tie|ție)|
      Bucuresti|București
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

ONLY_QTY_LINE = re.compile(
    r"^\s*(?:x\s*)?\d+(?:[.,]\d+)?\s*(?:buc|pcs|x)?\s*$",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"(?:(\d{1,2})[.\-\/](\d{1,2})[.\-\/](\d{2,4}))|(\d{4}-\d{2}-\d{2})"
)
AMOUNT_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})(?:\s*RON|\s*lei|\s*LEI)?",
    re.IGNORECASE,
)

CURRENCY_RE = re.compile(r"\b(RON|LEI|EUR|USD)\b", re.IGNORECASE)

# categorii simple
CATEGORIES = {
    "Mega": "Grocery",
    "Lidl": "Grocery",
    "Carrefour": "Grocery",
    "Kaufland": "Grocery",
    "OMV": "Fuel",
    "MOL": "Fuel",
    "BCR": "Fees",
    "Signal-Iduna": "Insurance",
    "EP*": "Online",
}


# ===================== OCR ======================

def available_langs() -> List[str]:
    try:
        return pytesseract.get_languages(config="") if pytesseract else []
    except Exception:
        return []

def ensure_pytesseract_path():
    """Attempt to set Windows default path if tesseract is missing."""
    if not pytesseract:
        return
    try:
        _ = pytesseract.get_tesseract_version()
        return
    except Exception:
        pass
    fallback = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(fallback).exists():
        pytesseract.pytesseract.tesseract_cmd = fallback

def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    if not cv2:
        return pil_image.convert("L")
    try:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 11)
        return Image.fromarray(thr)
    except Exception:
        return pil_image.convert("L")

def ocr_text_from_pil(pil_image: Image.Image) -> str:
    """
    Robust OCR: încearcă 'ron+eng', apoi 'ron', apoi 'eng'.
    Dacă Tesseract nu e disponibil -> returnează "" (nu crăpăm aplicația).
    """
    if not pytesseract:
        return ""
    ensure_pytesseract_path()
    cfg = "--oem 3 --psm 6"
    candidates = ["ron+eng", "ron", "eng"]
    langs = available_langs()
    order = [c for c in candidates if all(x in langs for x in c.split("+"))] or candidates
    img = preprocess_for_ocr(pil_image)
    for lang in order:
        try:
            return pytesseract.image_to_string(img, lang=lang, config=cfg)
        except Exception:
            continue
    return ""


# ===================== PDF / Image Reader ======================

def images_from_pdf(file_bytes: bytes) -> List[Image.Image]:
    pages: List[Image.Image] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for p in doc:
            pix = p.get_pixmap(alpha=False, dpi=220)
            pages.append(Image.open(BytesIO(pix.tobytes("png"))))
    return pages

def read_text_from_file(upload) -> str:
    name = upload.name.lower()
    if name.endswith(".pdf"):
        text_pages = []
        for img in images_from_pdf(upload.getvalue()):
            text_pages.append(ocr_text_from_pil(img))
        return "\n".join(text_pages)
    else:
        img = Image.open(upload).convert("RGB")
        return ocr_text_from_pil(img)


# ===================== Parsing Bon / Extras ======================

def split_clean_lines(text: str) -> List[str]:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if META_RE.search(s):  # eliminăm metadate din extrase
            continue
        if ONLY_QTY_LINE.match(s):  # linii gen "x 2"
            continue
        lines.append(s)
    return lines

def guess_merchant(lines: List[str]) -> str:
    if not lines:
        return ""
    head = " ".join(lines[:4])
    # caută nume cunoscute
    for k in CATEGORIES.keys():
        if k.lower() in head.lower():
            return k
    # fallback: prima linie non-VID
    return lines[0][:40]

def guess_date(text: str) -> Optional[date]:
    m = DATE_RE.search(text)
    if not m:
        return None
    if m.group(4):
        # yyyy-mm-dd
        try:
            return datetime.strptime(m.group(4), "%Y-%m-%d").date()
        except Exception:
            return None
    d, mth, y = m.group(1), m.group(2), m.group(3)
    try:
        y = int(y)
        if y < 100:
            y += 2000
        return date(int(y), int(mth), int(d))
    except Exception:
        return None

def guess_total(text: str) -> Optional[float]:
    # ia ultima sumă din text (de obicei totalul)
    matches = AMOUNT_RE.findall(text)
    if not matches:
        return None
    val = matches[-1][0]
    val = val.replace(".", "").replace(",", ".")
    try:
        return float(val)
    except Exception:
        return None

def guess_currency(text: str) -> str:
    m = CURRENCY_RE.search(text)
    if not m:
        return "RON"
    cur = m.group(1).upper()
    return "LEI" if cur == "LEI" else cur


def classify_category(merchant: str) -> str:
    for k, v in CATEGORIES.items():
        if k.lower() in merchant.lower():
            return v
    return "Other"


# ===================== Persistence (Supabase + CSV fallback) ======================

def ensure_csv():
    if not CSV_PATH.exists():
        pd.DataFrame(columns=[
            "date", "merchant", "category", "amount", "currency", "notes", "raw"
        ]).to_csv(CSV_PATH, index=False)

def load_tx() -> pd.DataFrame:
    """
    Dacă SUPABASE_* există -> citim din Supabase (tabela 'transactions' filtrată pe user_id).
    Altfel -> CSV local.
    """
    user = st.session_state.get("user", {})
    user_id = user.get("id") or user.get("email") or "anonymous"

    if sb_enabled():
        try:
            params = {"user_id": f"eq.{user_id}", "order": "date.desc,created_at.desc"}
            r = requests.get(sb_table_url("transactions"), headers=sb_headers(), params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not data:
                return pd.DataFrame(columns=[
                    "date", "merchant", "category", "amount", "currency", "notes", "raw"
                ])
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            return df
        except Exception as e:
            st.warning(f"Supabase indisponibil, revin la CSV: {e}")

    # CSV fallback
    ensure_csv()
    try:
        df = pd.read_csv(CSV_PATH)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        return df
    except Exception:
        ensure_csv()
        return pd.read_csv(CSV_PATH)

def append_rows(df_new: pd.DataFrame) -> pd.DataFrame:
    user = st.session_state.get("user", {})
    user_id = user.get("id") or user.get("email") or "anonymous"

    # normalize columns
    df = df_new.copy()
    for col in ["date", "merchant", "category", "amount", "currency", "notes", "raw"]:
        if col not in df.columns:
            df[col] = None
    df["currency"] = df["currency"].fillna("RON")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Supabase
    if sb_enabled():
        try:
            payload = df.copy()
            payload["user_id"] = user_id
            if "raw" in payload.columns:
                payload["raw"] = payload["raw"].apply(
                    lambda x: x if isinstance(x, (str, type(None))) else json.dumps(x, ensure_ascii=False)
                )
            payload["date"] = payload["date"].astype(str)
            r = requests.post(
                sb_table_url("transactions"),
                headers=sb_headers(),
                data=payload.to_json(orient="records", force_ascii=False).encode("utf-8"),
                timeout=25,
            )
            r.raise_for_status()
            out = r.json() if r.content else []
            return pd.DataFrame(out) if out else df
        except Exception as e:
            st.warning(f"Supabase indisponibil la scriere, salvez în CSV. Motiv: {e}")

    # CSV fallback
    ensure_csv()
    old = load_tx()
    all_df = pd.concat([old, df], ignore_index=True)
    all_df.to_csv(CSV_PATH, index=False)
    return df

def clear_all_data():
    user = st.session_state.get("user", {})
    user_id = user.get("id") or user.get("email") or "anonymous"

    if sb_enabled():
        try:
            # delete for this user
            url = sb_table_url("transactions")
            params = {"user_id": f"eq.{user_id}"}
            r = requests.delete(url, headers=sb_headers(), params=params, timeout=20)
            r.raise_for_status()
            return
        except Exception as e:
            st.warning(f"Nu am putut șterge din Supabase: {e}")

    # CSV fallback
    ensure_csv()
    pd.DataFrame(columns=["date", "merchant", "category", "amount", "currency", "notes", "raw"]).to_csv(CSV_PATH, index=False)


# ===================== Export ======================

def export_buffers(df: pd.DataFrame) -> Tuple[BytesIO, BytesIO, BytesIO]:
    csv_buf = BytesIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    xlsx_buf = BytesIO()
    try:
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Transactions", index=False)
    except Exception:
        xlsx_buf = BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Transactions", index=False)
    xlsx_buf.seek(0)

    json_buf = BytesIO(json.dumps(json.loads(df.to_json(orient="records")), ensure_ascii=False, indent=2).encode("utf-8"))
    json_buf.seek(0)

    return csv_buf, xlsx_buf, json_buf


# ===================== UI ======================

def sidebar_user():
    st.sidebar.header("Autentificare")
    if "user" not in st.session_state:
        st.session_state["user"] = {}

    email = st.sidebar.text_input("Email (doar pentru identificare)", value=st.session_state["user"].get("email", "user@example.com"))
    name = st.sidebar.text_input("Nume", value=st.session_state["user"].get("name", "Silviu"))
    if st.sidebar.button("Setează utilizator"):
        st.session_state["user"] = {"id": email or "anonymous", "email": email, "name": name}
        st.sidebar.success("Utilizator setat!")

    st.sidebar.caption(
        "Dacă ai configurat **SUPABASE_URL** și **SUPABASE_ANON_KEY** în Render, "
        "datele se salvează în Supabase. Altfel, în CSV local."
    )

def import_section():
    st.subheader("Importă bon / extras (PDF, JPG, PNG)")

    up = st.file_uploader("Încarcă fișier", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=False)
    if not up:
        return

    with st.spinner("Rulez OCR..."):
        text = read_text_from_file(up)

    st.text_area("Text OCR (debug)", text, height=180)

    lines = split_clean_lines(text)
    if lines:
        st.write("Linii detectate:", len(lines))
        st.dataframe(pd.DataFrame({"line": lines}))
    else:
        st.warning("Nu am detectat linii relevante.")

    # propuneri
    m = guess_merchant(lines)
    d = guess_date(text) or date.today()
    total = guess_total(text) or 0.0
    curr = guess_currency(text)
    cat = classify_category(m)

    with st.form("confirm_tx"):
        col1, col2, col3 = st.columns(3)
        with col1:
            d2 = st.date_input("Data", value=d)
            val = st.number_input("Suma", value=float(total), step=0.01, format="%.2f")
        with col2:
            merchant = st.text_input("Comerciant", value=m)
            currency = st.selectbox("Monedă", ["RON", "LEI", "EUR", "USD"], index=["RON","LEI","EUR","USD"].index(curr if curr in ["RON","LEI","EUR","USD"] else "RON"))
        with col3:
            category = st.text_input("Categorie", value=cat)
            notes = st.text_input("Note", value="")

        submitted = st.form_submit_button("Adaugă tranzacția")
        if submitted:
            row = pd.DataFrame([{
                "date": d2,
                "merchant": merchant,
                "category": category,
                "amount": val,
                "currency": "RON" if currency == "LEI" else currency,
                "notes": notes,
                "raw": {"source": up.name, "lines": lines[:50]},
            }])
            appended = append_rows(row)
            st.success("Tranzacție adăugată ✅")
            st.session_state["last_added"] = appended.to_dict("records") if isinstance(appended, pd.DataFrame) else row.to_dict("records")


def quick_add():
    st.subheader("Adăugare rapidă")
    with st.form("quick_add_form"):
        c1, c2, c3, c4 = st.columns([1,2,1,1])
        d = c1.date_input("Data", value=date.today())
        merch = c2.text_input("Comerciant", value="")
        amt = c3.number_input("Suma", value=0.0, step=0.5, format="%.2f")
        cat = c4.text_input("Categorie", value="Other")
        col0, colx = st.columns([2,2])
        cur = col0.selectbox("Moneda", ["RON","LEI","EUR","USD"], index=0)
        note = colx.text_input("Note", value="")
        ok = st.form_submit_button("Adaugă")
        if ok:
            df = pd.DataFrame([{
                "date": d,
                "merchant": merch,
                "category": cat,
                "amount": amt,
                "currency": "RON" if cur == "LEI" else cur,
                "notes": note,
                "raw": {},
            }])
            append_rows(df)
            st.success("Salvat!")

def dashboard(df: pd.DataFrame):
    st.subheader("Dashboard")
    if df.empty:
        st.info("Nu sunt tranzacții încă.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # cheltuieli pe categorie
    cat = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    chart_cat = alt.Chart(cat).mark_bar().encode(
        x=alt.X("amount:Q", title="Suma"),
        y=alt.Y("category:N", sort="-x", title="Categorie"),
        tooltip=["category","amount"]
    ).properties(height=300)

    # evoluție în timp
    ts = df.groupby("month", as_index=False)["amount"].sum()
    chart_ts = alt.Chart(ts).mark_line(point=True).encode(
        x=alt.X("month:T", title="Luna"),
        y=alt.Y("amount:Q", title="Total"),
        tooltip=["month","amount"]
    ).properties(height=300)

    # top comercianți
    topm = (df.groupby("merchant", as_index=False)["amount"].sum()
              .sort_values("amount", ascending=False).head(10))
    chart_m = alt.Chart(topm).mark_bar().encode(
        x=alt.X("amount:Q", title="Suma"),
        y=alt.Y("merchant:N", sort="-x", title="Comerciant"),
        tooltip=["merchant","amount"]
    ).properties(height=300)

    c1, c2 = st.columns(2)
    c1.altair_chart(chart_cat, use_container_width=True)
    c2.altair_chart(chart_ts, use_container_width=True)
    st.altair_chart(chart_m, use_container_width=True)

def export_section(df: pd.DataFrame):
    st.subheader("Export")
    csv_buf, xlsx_buf, json_buf = export_buffers(df)
    st.download_button("Descarcă CSV", data=csv_buf, file_name="transactions.csv", mime="text/csv")
    st.download_button("Descarcă Excel", data=xlsx_buf, file_name="transactions.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Descarcă JSON", data=json_buf, file_name="transactions.json", mime="application/json")


# ===================== Main App ======================

def main():
    st.title("Aplicatie Cheltuieli – OCR + Supabase")
    sidebar_user()

    tab1, tab2, tab3 = st.tabs(["📥 Import", "➕ Adăugare rapidă", "📊 Dashboard & Export"])

    with tab1:
        import_section()

    with tab2:
        quick_add()

    with tab3:
        df = load_tx()
        st.write("Ultimele tranzacții")
        st.dataframe(df.sort_values(by=["date"], ascending=False, na_position="last"), use_container_width=True)
        dashboard(df)
        export_section(df)

    st.divider()
    colA, colB = st.columns([1,1])
    if colA.button("Șterge toate datele utilizatorului curent"):
        clear_all_data()
        st.success("Date șterse.")
    if colB.button("Reîncarcă"):
        st.rerun()


if __name__ == "__main__":
    main()
