# app.py — OCR bonuri cu învățare categorie, Supabase + CSV fallback, dashboard & export

import os, io, re, json, math
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt

# opționale (protejate la import)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -------------------- Config --------------------
st.set_page_config(page_title="Aplicatie Cheltuieli — OCR inteligent", layout="wide")

APP_DATA_DIR = Path(os.getenv("APP_DATA_DIR", "./data"))
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# CSV fallback
CSV_ITEMS = APP_DATA_DIR / "tx_items.csv"
CSV_CATMAP = APP_DATA_DIR / "category_learning.csv"

# Supabase
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

def sb_url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"

# -------------------- Regex & parsing helpers --------------------
# linii de metadate pe care le ignorăm în OCR (cap/coloane din extrase, info card etc.)
META_RE = re.compile(
    r"(Booking|Reference|Card|Apple\s*Pay|Exchange\s*rate|Comision|Commission|"
    r"TVA|VAT|Subtotal|Sub\-?total|Total\s*de\s*plata|Total|Sumă|Suma\s*plătită|"
    r"Rest|Cash|Bon|Fiscal|Ora|Time|CUI|CIF|TVA\s*%\s*|NIR|Casa|Bon\s*nr|Bon\s+fiscal)",
    re.IGNORECASE,
)
ONLY_QTY_LINE = re.compile(r"^\s*(?:x\s*)?\d+(?:[.,]\d+)?\s*(?:buc|pcs|x)?\s*$", re.IGNORECASE)

# preț la final de linie: "DORNA 2L .... 4,50" sau "Apa Dorna x2 9.00"
PRICE_TAIL_RE = re.compile(
    r"""(?P<name>.+?)\s+(?P<price>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})\s*(?:RON|LEI|lei|RON)?\s*$""",
    re.IGNORECASE,
)

DATE_RE = re.compile(r"(?:(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4}))|(\d{4}-\d{2}-\d{2})")
CUR_RE = re.compile(r"\b(RON|LEI|EUR|USD)\b", re.IGNORECASE)

def normalize_amount(s: str) -> Optional[float]:
    s = s.replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def get_user_id() -> str:
    u = st.session_state.get("user", {})
    return u.get("id") or u.get("email") or "anonymous"

def guess_date(text: str) -> Optional[date]:
    m = DATE_RE.search(text)
    if not m:
        return None
    if m.group(4):  # yyyy-mm-dd
        try:
            return datetime.strptime(m.group(4), "%Y-%m-%d").date()
        except Exception:
            return None
    d, mth, y = m.group(1), m.group(2), m.group(3)
    try:
        y = int(y)
        if y < 100: y += 2000
        return date(int(y), int(mth), int(d))
    except Exception:
        return None

def guess_currency(text: str) -> str:
    m = CUR_RE.search(text)
    if not m:
        return "RON"
    cur = m.group(1).upper()
    return "LEI" if cur == "LEI" else cur

def clean_line(s: str) -> str:
    s = re.sub(r"\s{2,}", " ", s.strip())
    return s

def line_is_meta(s: str) -> bool:
    if not s: return True
    if META_RE.search(s): return True
    if ONLY_QTY_LINE.match(s): return True
    return False

# -------------------- OCR --------------------
def ensure_tesseract_path():
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

def available_langs() -> List[str]:
    try:
        return pytesseract.get_languages(config="") if pytesseract else []
    except Exception:
        return []

def preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    if cv2 is None:
        return pil.convert("L")
    try:
        img = np.array(pil.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 11)
        return Image.fromarray(thr)
    except Exception:
        return pil.convert("L")

def ocr_text_from_pil(pil: Image.Image) -> str:
    if pytesseract is None:
        return ""
    ensure_tesseract_path()
    cfg = "--oem 3 --psm 6"
    candidates = ["ron+eng", "ron", "eng"]
    langs = available_langs()
    order = [c for c in candidates if all(x in langs for x in c.split("+"))] or candidates
    img = preprocess_for_ocr(pil)
    # încearcă în ordine
    for lang in order:
        try:
            return pytesseract.image_to_string(img, lang=lang, config=cfg)
        except Exception:
            continue
    return ""

# -------------------- Citire fișiere --------------------
def images_from_pdf(data: bytes) -> List[Image.Image]:
    pages = []
    if fitz is None:
        return pages
    with fitz.open(stream=data, filetype="pdf") as doc:
        for p in doc:
            pix = p.get_pixmap(alpha=False, dpi=220)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))))
    return pages

def read_text_from_upload(upload) -> Tuple[List[Image.Image], str]:
    name = upload.name.lower()
    if name.endswith(".pdf"):
        imgs = images_from_pdf(upload.getvalue())
        texts = [ocr_text_from_pil(im) for im in imgs]
        return imgs, "\n".join(texts)
    else:
        img = Image.open(upload).convert("RGB")
        return [img], ocr_text_from_pil(img)

# -------------------- Parsare linii (doar denumire + preț) --------------------
def extract_item_lines(text: str) -> pd.DataFrame:
    """
    Returnează DataFrame cu coloane:
      item_name, qty (implicit 1.0), unit_price, line_total, merchant_guess(optional), currency_guess(optional)
    Ignoră liniile meta/total/TVA. Caută preț la coada liniei.
    """
    items = []
    for raw in text.splitlines():
        s = clean_line(raw)
        if not s or line_is_meta(s):
            continue
        m = PRICE_TAIL_RE.search(s)
        if not m:
            continue
        name = clean_line(m.group("name"))
        price = normalize_amount(m.group("price"))
        if price is None:
            continue

        # detectează cantități simple „x2” din nume și le scoate din denumire
        qty = 1.0
        qmatch = re.search(r"(?:^|\s)x\s*(\d+(?:[.,]\d+)?)\b", name, re.IGNORECASE)
        if qmatch:
            qty = normalize_amount(qmatch.group(1)) or 1.0
            name = re.sub(r"(?:^|\s)x\s*\d+(?:[.,]\d+)?\b", "", name, flags=re.IGNORECASE).strip()

        items.append({
            "item_name": name[:120],
            "qty": qty,
            "unit_price": round(price/qty, 2) if qty and qty > 0 else price,
            "line_total": round(price, 2)
        })
    if not items:
        return pd.DataFrame(columns=["item_name","qty","unit_price","line_total"])
    df = pd.DataFrame(items)
    return df

# -------------------- „Învățare” categorie pe item --------------------
# cheie în map: item_name normalizat (litere mici + spații compacte)
def item_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())

def catmap_fetch() -> pd.DataFrame:
    if sb_enabled():
        try:
            url = sb_url("category_learning")
            params = {"user_id": f"eq.{get_user_id()}"}
            r = pd.read_json(
                io.BytesIO(
                    __import__("requests").get(url, headers=sb_headers(), params=params, timeout=15).content
                )
            )
            return r if not r.empty else pd.DataFrame(columns=["user_id","item_key","category"])
        except Exception as e:
            st.warning(f"Nu pot citi category_learning din Supabase: {e}")
    # CSV
    if not CSV_CATMAP.exists():
        return pd.DataFrame(columns=["user_id","item_key","category"])
    try:
        df = pd.read_csv(CSV_CATMAP)
        return df
    except Exception:
        return pd.DataFrame(columns=["user_id","item_key","category"])

def catmap_set(name: str, category: str):
    key = item_key(name)
    uid = get_user_id()
    if sb_enabled():
        try:
            url = sb_url("category_learning")
            payload = [{"user_id": uid, "item_key": key, "category": category}]
            r = __import__("requests").post(url, headers=sb_headers(),
                                            data=json.dumps(payload).encode("utf-8"),
                                            timeout=15)
            # dacă există deja, folosim upsert pe pk (definit în DB)
            if r.status_code not in (200, 201):
                # fallback: DELETE + INSERT
                __import__("requests").delete(url, headers=sb_headers(), params={"user_id": f"eq.{uid}", "item_key": f"eq.{key}"})
                __import__("requests").post(url, headers=sb_headers(),
                                            data=json.dumps(payload).encode("utf-8"),
                                            timeout=15)
            return
        except Exception as e:
            st.warning(f"Nu pot scrie în category_learning Supabase: {e}")
    # CSV
    m = catmap_fetch()
    m = m[~((m["user_id"]==uid) & (m["item_key"]==key))]  # remove old
    m = pd.concat([m, pd.DataFrame([{"user_id": uid, "item_key": key, "category": category}])], ignore_index=True)
    m.to_csv(CSV_CATMAP, index=False)

def catmap_apply(df_items: pd.DataFrame) -> pd.DataFrame:
    m = catmap_fetch()
    if m.empty or df_items.empty:
        df_items["category"] = df_items.get("category", pd.Series(["Other"]*len(df_items)))
        return df_items
    m = m[m["user_id"]==get_user_id()]
    mapping = dict(zip(m["item_key"], m["category"]))
    cats = []
    for _, row in df_items.iterrows():
        k = item_key(row.get("item_name",""))
        cats.append(mapping.get(k, "Other"))
    out = df_items.copy()
    out["category"] = cats
    return out

# -------------------- Persistență item-level --------------------
def ensure_items_csv():
    if not CSV_ITEMS.exists():
        pd.DataFrame(columns=[
            "date","merchant","currency","item_name","qty","unit_price","line_total","category","notes"
        ]).to_csv(CSV_ITEMS, index=False)

def items_load() -> pd.DataFrame:
    if sb_enabled():
        try:
            url = sb_url("tx_items")
            r = __import__("requests").get(url, headers=sb_headers(),
                                           params={"user_id": f"eq.{get_user_id()}","order":"date.desc,created_at.desc"},
                                           timeout=20)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data) if data else pd.DataFrame(columns=[
                "date","merchant","currency","item_name","qty","unit_price","line_total","category","notes"
            ])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            for c in ["qty","unit_price","line_total"]:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            st.warning(f"Supabase indisponibil (citire), revin la CSV: {e}")
    ensure_items_csv()
    try:
        df = pd.read_csv(CSV_ITEMS)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        for c in ["qty","unit_price","line_total"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception:
        ensure_items_csv()
        return pd.read_csv(CSV_ITEMS)

def items_append(df_new: pd.DataFrame) -> None:
    uid = get_user_id()
    df = df_new.copy()

    # normalize columns
    for c in ["date","merchant","currency","item_name","qty","unit_price","line_total","category","notes"]:
        if c not in df.columns: df[c] = None
    df["currency"] = df["currency"].fillna("RON")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Supabase
    if sb_enabled():
        try:
            payload = df.copy()
            payload["user_id"] = uid
            payload["date"] = payload["date"].astype(str)
            r = __import__("requests").post(
                sb_url("tx_items"), headers=sb_headers(),
                data=payload.to_json(orient="records", force_ascii=False).encode("utf-8"),
                timeout=25
            )
            r.raise_for_status()
            return
        except Exception as e:
            st.warning(f"Supabase indisponibil (scriere), salvez în CSV: {e}")

    # CSV
    ensure_items_csv()
    old = items_load()
    all_df = pd.concat([old, df], ignore_index=True)
    all_df.to_csv(CSV_ITEMS, index=False)

def items_clear_all():
    uid = get_user_id()
    if sb_enabled():
        try:
            __import__("requests").delete(sb_url("tx_items"), headers=sb_headers(),
                                          params={"user_id": f"eq.{uid}"}, timeout=20)
            return
        except Exception as e:
            st.warning(f"Nu am putut șterge din Supabase: {e}")
    ensure_items_csv()
    pd.DataFrame(columns=[
        "date","merchant","currency","item_name","qty","unit_price","line_total","category","notes"
    ]).to_csv(CSV_ITEMS, index=False)

# -------------------- Export --------------------
def export_buffers(df: pd.DataFrame) -> Tuple[io.BytesIO, io.BytesIO, io.BytesIO]:
    csv_buf = io.BytesIO(); df.to_csv(csv_buf, index=False); csv_buf.seek(0)

    xlsx_buf = io.BytesIO()
    try:
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Items")
    except Exception:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name="Items")
    xlsx_buf.seek(0)

    json_buf = io.BytesIO(json.dumps(json.loads(df.to_json(orient="records")),
                                     ensure_ascii=False, indent=2).encode("utf-8"))
    json_buf.seek(0)
    return csv_buf, xlsx_buf, json_buf

# -------------------- UI --------------------
def sidebar_user():
    st.sidebar.header("Autentificare")
    if "user" not in st.session_state:
        st.session_state["user"] = {}
    email = st.sidebar.text_input("Email (ID utilizator)", value=st.session_state["user"].get("email", "user@example.com"))
    name = st.sidebar.text_input("Nume", value=st.session_state["user"].get("name", "Utilizator"))
    if st.sidebar.button("Setează utilizator"):
        st.session_state["user"] = {"id": email or "anonymous", "email": email, "name": name}
        st.sidebar.success("Utilizator setat.")
    st.sidebar.caption("Cu SUPABASE_URL + SUPABASE_ANON_KEY în Render, datele rămân în cloud. Altfel, CSV local.")

def tab_import_ocr():
    st.subheader("📥 Import OCR (PDF/JPG/PNG) — doar denumire + preț per linie")
    upl = st.file_uploader("Încarcă bon / extras", type=["pdf","jpg","jpeg","png"])
    if not upl:
        return
    imgs, text = read_text_from_upload(upl)
    # preview imagini
    with st.expander("Previzualizare bon / pagini"):
        cols = st.columns(min(3, max(1, len(imgs))))
        for i, im in enumerate(imgs):
            cols[i % len(cols)].image(im, caption=f"Pagina {i+1}", use_column_width=True)
    with st.expander("Text OCR (debug)"):
        st.text_area("OCR", text, height=180)

    # parsare linii
    items = extract_item_lines(text)
    if items.empty:
        st.warning("Nu am găsit linii cu denumire + preț. Verifică claritatea pozei.")
        return

    # metadate
    d = guess_date(text) or date.today()
    curr = guess_currency(text)
    merchant = st.text_input("Comerciant", value="")
    items = catmap_apply(items)

    st.write("Linii detectate (editează înainte de salvare):")
    edited = st.data_editor(
        items.assign(date=d, merchant=merchant, currency=curr, notes=""),
        column_config={
            "item_name": st.column_config.TextColumn("Denumire"),
            "qty": st.column_config.NumberColumn("Cant.", step=0.25, format="%.2f"),
            "unit_price": st.column_config.NumberColumn("Preț unitar", format="%.2f"),
            "line_total": st.column_config.NumberColumn("Valoare linie", format="%.2f"),
            "category": st.column_config.TextColumn("Categorie"),
            "date": st.column_config.DateColumn("Data"),
            "merchant": st.column_config.TextColumn("Comerciant"),
            "currency": st.column_config.TextColumn("Monedă"),
            "notes": st.column_config.TextColumn("Note"),
        },
        num_rows="dynamic",
        use_container_width=True
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Salvează liniile"):
            # scriem itemele
            items_append(edited)
            # învățare categorie din ce ai editat (map pe item_name → category)
            for _, r in edited.iterrows():
                if str(r.get("category","")).strip():
                    catmap_set(str(r.get("item_name","")), str(r.get("category","")))
            st.success("Salvat. Categoria pentru denumiri se va propune automat data viitoare.")

    with c2:
        if st.button("🧹 Golește toate datele utilizatorului"):
            items_clear_all()
            st.success("Șters tot pentru utilizatorul curent.")

def tab_import_csv():
    st.subheader("🔄 Import CSV (linii item)")
    st.caption("Aștept coloane: date, merchant, currency, item_name, qty, unit_price, line_total, category, notes (numele pot fi mapate).")
    upl = st.file_uploader("Încarcă CSV", type=["csv"], key="csv_up")
    if not upl:
        return

    try:
        df = pd.read_csv(upl)
    except Exception as e:
        st.error(f"Nu pot citi CSV: {e}")
        return

    # mapare coloane
    expected = ["date","merchant","currency","item_name","qty","unit_price","line_total","category","notes"]
    options = {c: st.selectbox(f"Mapează '{c}'", ["(none)"] + list(df.columns), index=(df.columns.get_loc(c)+1 if c in df.columns else 0)) for c in expected}
    if st.button("✅ Importă"):
        out = pd.DataFrame()
        for tgt, src in options.items():
            out[tgt] = df[src] if src != "(none)" and src in df.columns else None
        # tipuri
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
        for c in ["qty","unit_price","line_total"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["currency"] = out["currency"].fillna("RON")
        items_append(out)
        st.success(f"Am importat {len(out)} linii.")

def tab_dashboard_export():
    st.subheader("📊 Dashboard & Export")
    df = items_load()
    if df.empty:
        st.info("Nu sunt date încă.")
        return

    # filtre rapide
    c1, c2, c3 = st.columns(3)
    with c1:
        y_min = int(df["date"].dropna().min().year) if df["date"].notna().any() else date.today().year
        y_max = int(df["date"].dropna().max().year) if df["date"].notna().any() else date.today().year
        years = st.slider("Ani", min_value=y_min, max_value=y_max, value=(y_min, y_max))
    with c2:
        merch_filter = st.text_input("Filtru comerciant conține", "")
    with c3:
        cat_filter = st.text_input("Filtru categorie conține", "")

    f = df.copy()
    if f["date"].notna().any():
        f = f[(f["date"].dt.year >= years[0]) & (f["date"].dt.year <= years[1])]
    if merch_filter.strip():
        f = f[f["merchant"].fillna("").str.contains(merch_filter.strip(), case=False, na=False)]
    if cat_filter.strip():
        f = f[f["category"].fillna("").str.contains(cat_filter.strip(), case=False, na=False)]

    st.write("Tabel tranzacții (item-level):")
    st.dataframe(f.sort_values(by=["date"], ascending=False, na_position="last"), use_container_width=True)

    # Pie chart pe categorii
    cat = f.groupby("category", as_index=False)["line_total"].sum().sort_values("line_total", ascending=False)
    pie = alt.Chart(cat).mark_arc().encode(
        theta="line_total:Q",
        color=alt.Color("category:N", legend=None),
        tooltip=["category","line_total"]
    ).properties(width=300, height=300)

    bar_cat = alt.Chart(cat).mark_bar().encode(
        x=alt.X("line_total:Q", title="Total"),
        y=alt.Y("category:N", sort="-x", title="Categorie"),
        tooltip=["category","line_total"]
    ).properties(height=300)

    # Serie în timp (sumă pe zi)
    t = f.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    ts = t.groupby("date", as_index=False)["line_total"].sum()
    line = alt.Chart(ts).mark_line(point=True).encode(
        x=alt.X("date:T", title="Dată"),
        y=alt.Y("line_total:Q", title="Total/zi"),
        tooltip=["date","line_total"]
    ).properties(height=300)

    c1, c2, c3 = st.columns([1,1,1])
    c1.altair_chart(pie, use_container_width=True)
    c2.altair_chart(bar_cat, use_container_width=True)
    c3.altair_chart(line, use_container_width=True)

    st.subheader("⬇️ Export")
    csv_b, xlsx_b, json_b = export_buffers(f)
    st.download_button("CSV", data=csv_b, file_name="tx_items.csv", mime="text/csv")
    st.download_button("Excel", data=xlsx_b, file_name="tx_items.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("JSON", data=json_b, file_name="tx_items.json", mime="application/json")

# -------------------- Main --------------------
def main():
    st.title("Optimizare Cheltuieli — OCR inteligent & învățare categorie")
    sidebar_user()

    tab1, tab2, tab3 = st.tabs(["🧾 Import OCR", "📄 Import CSV", "📊 Dashboard & Export"])
    with tab1: tab_import_ocr()
    with tab2: tab_import_csv()
    with tab3: tab_dashboard_export()

if __name__ == "__main__":
    main()
