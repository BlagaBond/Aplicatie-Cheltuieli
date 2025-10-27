# -*- coding: utf-8 -*-
"""
💼 Budget App 
Compatibil: Windows + Python 3.12

Funcționalități:
- OCR imagini & PDF (Tesseract + PyMuPDF)
- AI: detectare Reduceri & propunere categorie (scikit-learn)
- Parsing linii bon (evită TOTAL/Card/Apple Pay; păstrează Delivery/Service fee ca cheltuieli)
- Import Money Manager CSV
- Dashboard: bar & pie pe categorii + metrici
- Editor CRUD pentru transactions.csv (edit, șterge, normalizează semnele, deduplicate)
"""

# ================== IMPORTURI ==================
import streamlit as st
st.set_page_config(page_title="Budget OCR + AI", layout="wide")
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import altair as alt
from typing import Dict, Any, List
from dataclasses import dataclass

import pandas as pd
import yaml
import uuid
import io
import re

# Regex-uri folosite în parsarea liniilor de pe bon
META_RE = re.compile(r"\b(total|tva|card|visa|mastercard|rest|cash|change|apple|google|ramburs|plata|receipt|bon|fiscal)\b", re.I)
ONLY_QTY_LINE = re.compile(r"^\s*\d+(?:[.,]\d+)?\s*(?:buc|kg|l|pcs)?\s*[x×*]\s*\d+[.,]\d{2}(?:\s+\d+[.,]\d{2})?\s*$", re.I)

import unicodedata
from datetime import datetime, date
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import fitz  # PyMuPDF

# OCR deps
import pytesseract

# --------------- USER AUTH AND LOGIN ---------------
import os
import requests
from pathlib import Path  # <- asigură-te că importul există
# ==== PATCH 1: imports + helpers (TOP of app.py) ====
from __future__ import annotations
import uuid, re
from datetime import datetime, date as _date
import pandas as pd
import streamlit as st

# setează calea ta reală:
CSV_PATH = "transactions.csv"   # <- schimbă dacă folosești altă locație

SCHEMA_COLS = [
    "id","date","merchant","amount","currency","category","notes","source","created_at","ew"
]
DEFAULTS = {
    "id": lambda: uuid.uuid4().hex[:10],
    "date": lambda: datetime.utcnow().strftime("%Y-%m-%d"),
    "merchant": "",
    "amount": 0.0,
    "currency": "RON",
    "category": "uncategorized",
    "notes": "",
    "source": "manual",
    "created_at": lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    "ew": "unknown",  # essential / nonessential / unknown
}

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df.columns):
        df.columns = [str(c).strip().lower() for c in df.columns]

    alias = {
        "descriere":"merchant","merchant name":"merchant","details":"merchant",
        "sum":"amount","value":"amount","val":"amount",
        "data":"date","tip":"category","type":"category"
    }
    for k,v in alias.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)

    if "amount" in df.columns:
        df["amount"] = (
            df["amount"].astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    for col in SCHEMA_COLS:
        if col not in df.columns:
            default = DEFAULTS[col]
            df[col] = [default() if callable(default) else default]*len(df)

    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    except Exception:
        df["date"] = datetime.utcnow().strftime("%Y-%m-%d")

    df["currency"] = df["currency"].fillna("RON")
    df["category"] = df["category"].fillna("uncategorized")
    df["ew"] = df["ew"].fillna("unknown")
    return df[SCHEMA_COLS]

def read_transactions(path: str = CSV_PATH) -> pd.DataFrame:
    try:
        base = pd.read_csv(path)
    except FileNotFoundError:
        base = pd.DataFrame(columns=SCHEMA_COLS)
    return _ensure_columns(base)

def write_transactions(df: pd.DataFrame, path: str = CSV_PATH) -> None:
    df = _ensure_columns(df)
    df.to_csv(path, index=False)

def append_transactions(new_rows: pd.DataFrame, path: str = CSV_PATH) -> pd.DataFrame:
    base = read_transactions(path)
    nr = _ensure_columns(new_rows)
    all_rows = pd.concat([base, nr], ignore_index=True)
    # dedupe by (date, merchant, amount, currency)
    all_rows.drop_duplicates(subset=["date","merchant","amount","currency"], keep="first", inplace=True)
    write_transactions(all_rows, path)
    return all_rows

def flag_internal_transfers(df: pd.DataFrame, minutes_window=1440) -> pd.DataFrame:
    """Marchează perechi +/− cu aceeași sumă ca transfer intern (exclude din totaluri)."""
    if df.empty: 
        return df
    d = df.copy()
    d["abs_amt"] = d["amount"].abs().round(2)
    d["dt"] = pd.to_datetime(d["date"], errors="coerce")

    pos = d[d["amount"] > 0]
    neg = d[d["amount"] < 0]
    if pos.empty or neg.empty:
        d.drop(columns=["abs_amt","dt"], errors="ignore", inplace=True)
        return d

    pairs = pos.merge(neg, on=["abs_amt","currency"], suffixes=("_p","_n"))
    ok = (pairs["dt_p"] - pairs["dt_n"]).abs().dt.total_seconds().abs() <= minutes_window*60
    ids = set(pairs.loc[ok, "id_p"]).union(set(pairs.loc[ok, "id_n"]))

    if ids:
        d.loc[d["id"].isin(ids), "source"] = "transfer-internal"
    d.drop(columns=["abs_amt","dt"], errors="ignore", inplace=True)
    return d

# OCR line parser:  "NumeProdus 12,50"
LINE_RE = re.compile(r"^\s*(.+?)\s+([0-9]+(?:[.,][0-9]{1,2})?)\s*$", re.IGNORECASE)
def parse_receipt_text(text: str) -> pd.DataFrame:
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(("total","suma","sumă","tva")):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, price = m.group(1), m.group(2)
        price = float(price.replace(",", "."))
        rows.append({
            "merchant": name[:50],
            "amount": -price,           # negativ = cheltuială
            "currency": "RON",
            "category":"uncategorized",
            "notes":"",
            "source":"ocr-receipt",
            "ew":"unknown",
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "id": uuid.uuid4().hex[:10],
        })
    return _ensure_columns(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=SCHEMA_COLS)
# --- CONSTANTE & PATHS DE BAZĂ (trebuie să fie DEFINITE ÎNAINTE de init_user_csv!) ---
BASE = Path(__file__).resolve().parent
# CSV_PATH preserved from init_user_csv(user)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
}

def get_user_by_username(username: str):
    url = f"{SUPABASE_URL}/rest/v1/users_auth"
    params = {"select": "id,username,password", "username": f"eq.{username}"}
    res = requests.get(url, headers=HEADERS, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()
    return data[0] if data else None

def create_user(username: str, password: str):
    """
    Creează un nou utilizator în tabela users_auth.
    Supabase nu returnează corp JSON în mod implicit la un INSERT, așa că
    adăugăm headerul Prefer:return=representation și verificăm conținutul.
    """
    url = f"{SUPABASE_URL}/rest/v1/users_auth"
    payload = {"username": username, "password": password}
    res = requests.post(
        url,
        headers={
            **HEADERS,
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        json=payload,
        timeout=10,
    )
    res.raise_for_status()
    return res.json() if res.content else None

def login_view():
    st.title("Autentificare")
    tab1, tab2 = st.tabs(["Intră", "Creează cont"])
    with tab1:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Parolă", type="password", key="login_p")
        if st.button("Intră"):
            user = get_user_by_username(u.strip())
            if not user or user["password"] != p:
                st.error("User sau parolă greșită")
            else:
                st.session_state["user"] = {"id": user["id"], "username": user["username"]}
                st.rerun()
    with tab2:
        nu = st.text_input("Username nou", key="new_u")
        np = st.text_input("Parolă nouă", type="password", key="new_p")
        if st.button("Creează cont"):
            if not nu or not np:
                st.warning("Completează user și parolă.")
            elif get_user_by_username(nu.strip()):
                st.error("Username deja există.")
            else:
                create_user(nu.strip(), np)
                st.success("Cont creat. Acum intră în tab-ul 'Intră'.")

def require_login():
    if "user" not in st.session_state:
        login_view()
        st.stop()
    return st.session_state["user"]

# --- LOGIN + CSV per user ---
user = require_login()

def init_user_csv(current_user: dict) -> None:
    """
    Fiecare utilizator își are fișierul lui: transactions_<user_id>.csv
    Utilizator nou = fișier nou, gol.
    """
    global CSV_PATH
    if current_user and current_user.get("id"):
        CSV_PATH = BASE / f"transactions_{current_user['id']}.csv"
    else:
        CSV_PATH = BASE / "transactions.csv"

init_user_csv(user)

# Greeting for logged in user
st.write(f"Bun venit, {user['username']}!")

# ================== SETUP & PATHS ==================

BASE = Path(__file__).resolve().parent
# CSV_PATH preserved from init_user_csv(user)
CATS_PATH = BASE / "categories.yaml"

ML_DIR = BASE / "ml"
ML_DIR.mkdir(exist_ok=True)
LABELED_PATH = ML_DIR / "labels.csv"
DISC_MODEL_PATH = ML_DIR / "disc_model.pkl"
CAT_MODEL_PATH = ML_DIR / "cat_model.pkl"

# ================== CSV HELPERS ==================
def ensure_csv():
    if not CSV_PATH.exists():
        cols = ["id", "date", "merchant", "amount", "currency", "category", "notes", "source", "created_at"]
        
if CSV_PATH is None:
    raise RuntimeError("CSV_PATH nu e setat (verifică init_user_csv și să nu-l resetezi ulterior).")
pd.DataFrame({
    "id": pd.Series(dtype=object),
    "date": pd.Series(dtype=object),
    "merchant": pd.Series(dtype=object),
    "amount": pd.Series(dtype="float64"),
    "currency": pd.Series(dtype=object),
    "category": pd.Series(dtype=object),
    "notes": pd.Series(dtype=object),
    "source": pd.Series(dtype=object),
    "created_at": pd.Series(dtype=object),
}).to_csv(CSV_PATH, index=False, encoding="utf-8")


def append_rows(df: pd.DataFrame):
    exists = CSV_PATH.exists()
    if exists:
        df.to_csv(CSV_PATH, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")

def append_row(row: dict):
    append_rows(pd.DataFrame([row]))


# --- Ensure DataFrame has editor-friendly dtypes (avoid text<->float mismatches) ---
def coerce_editor_dtypes(df):
    import pandas as pd
    text_cols = ["notes", "merchant", "category", "currency", "source", "id"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(object)
            df[c] = df[c].where(df[c].notna(), "")
    if "amount" in df.columns:
        try:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        except Exception:
            pass
    return df

def load_tx():
    ensure_csv()
    tx = pd.read_csv(CSV_PATH)
    if not tx.empty:
        tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
        tx["created_at"] = pd.to_datetime(tx["created_at"], errors="coerce")
        tx["currency"] = tx["currency"].fillna("RON")
    tx = coerce_editor_dtypes(tx)
    return tx

def overwrite_tx(df: pd.DataFrame):
    """Scrie întregul DataFrame în transactions.csv, cu backup .bak (Windows-safe)."""
    df = df.copy()
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    if "amount" in df:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).round(2)
    if "currency" in df:
        df["currency"] = df["currency"].fillna("RON")

    tmp = CSV_PATH.with_suffix(".tmp.csv")
    bak = CSV_PATH.with_suffix(".bak.csv")
    df.to_csv(tmp, index=False, encoding="utf-8")
    if CSV_PATH.exists():
        try:
            CSV_PATH.replace(bak)
        except Exception:
            pass
    tmp.replace(CSV_PATH)

# ================== CATEGORIES ==================
def load_categories():
    expense_defaults = [
        "Food & Groceries","Restaurants & Coffee","Transport","Fuel",
        "Utilities","Shopping","Health","Household","Entertainment",
        "Livrare & taxe","Reduceri","Uncategorized","Other"
    ]
    income_defaults = ["Salary","Bonus","Freelance","Refund","Interest","Other Income"]
    cats = {}
    if CATS_PATH.exists():
        try:
            cats = yaml.safe_load(CATS_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            cats = {}
    exp_from_yaml = [k for k in cats.keys() if k and k.lower() != "income"]
    inc_from_yaml = []

    def uniq(seq):
        seen=set(); out=[]
        for x in seq:
            if x not in seen:
                out.append(x); seen.add(x)
        return out

    return uniq(expense_defaults + exp_from_yaml), uniq(income_defaults + inc_from_yaml), cats

def save_categories(cats: dict):
    try:
        CATS_PATH.write_text(yaml.safe_dump(cats, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return True
    except Exception:
        return False

def normalize_text(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()

# ================== IMPORT HELPERS ==================
def read_csv_auto_bytes(b: bytes):
    for enc in ["utf-8-sig", "utf-16", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(b), encoding="utf-8", errors="ignore")

def parse_amount_series(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.replace("\u00A0","", regex=False).str.replace(" ", "", regex=False)
    def fix(x):
        x = str(x)
        if x.count(",") == 1 and x.count(".") >= 1 and x.rfind(",") > x.rfind("."):
            x = x.replace(".", "").replace(",", ".")
        else:
            x = x.replace(",", ".")
        return x
    s = s.map(fix)
    return pd.to_numeric(s, errors="coerce")

def parse_date_series(s: pd.Series, fmt_hint: str|None):
    if fmt_hint == "DMY":
        dayfirst = True; yearfirst = False
    elif fmt_hint == "MDY":
        dayfirst = False; yearfirst = False
    elif fmt_hint == "YMD":
        dayfirst = False; yearfirst = True
    else:
        dayfirst = True; yearfirst = False
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, yearfirst=yearfirst)

# ================== OCR HELPERS ==================
def preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((1,1), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(thr)

def ocr_text_from_pil(pil_image: Image.Image) -> str:
    cfg = "--oem 3 --psm 6"
    try:
        return pytesseract.image_to_string(pil_image, lang="ron+eng", config=cfg)
    except pytesseract.TesseractNotFoundError:
        fallback = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(fallback).exists():
            pytesseract.pytesseract.tesseract_cmd = fallback
            return pytesseract.image_to_string(pil_image, lang="ron+eng", config=cfg)
        raise

def find_date(text: str):
    patterns = [r'(\d{2}[./-]\d{2}[./-]\d{4})', r'(\d{4}[./-]\d{2}[./-]\d{2})', r'(\d{2}[./-]\d{2}[./-]\d{2})']
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                from dateutil import parser as dateparser
                dt = dateparser.parse(m.group(1), dayfirst=True)
                return dt.date().isoformat()
            except Exception:
                pass
    return date.today().isoformat()

def find_total(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    total_candidates = []
    for ln in reversed(lines):
        if re.search(r'(total|sum[aă]|plată|plata|de\s*plătit|de\s*plata|amount due|payable)', ln, re.I):
            m = re.search(r'(-?\d+[.,]\d{2})', ln)
            if m:
                total_candidates.append(m.group(1))
    if not total_candidates:
        m = re.search(r'(-?\d+[.,]\d{2})', "\n".join(lines[::-1]))
        if m:
            total_candidates.append(m.group(1))
    if total_candidates:
        val = total_candidates[0].replace(",", ".")
        try:
            return round(float(val), 2)
        except:
            pass
    return None

def find_merchant(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "Unknown Merchant"
    blacklist = ("BON","FISCAL","RECEIPT","SC","S.C.","SRL","S.R.L.")
    for ln in lines[:6]:
        if any(b in ln.upper() for b in blacklist):
            continue
        if sum(c.isalpha() for c in ln) >= 3:
            return ln[:64]
    return lines[0][:64]

def load_images_from_upload(uploaded_file):
    data = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    name = (uploaded_file.name or "").lower()
    mime = (uploaded_file.type or "").lower()
    if name.endswith(".pdf") or "pdf" in mime:
        images = []
        doc = fitz.open(stream=data, filetype="pdf")
        for page in doc:
            pix = page.get_pixmap(dpi=200, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    else:
        return [Image.open(io.BytesIO(data)).convert("RGB")]

# ================== AI HELPERS ==================
def _load_labels() -> pd.DataFrame:
    if LABELED_PATH.exists():
        try:
            df = pd.read_csv(LABELED_PATH)
            for col in ["name","merchant","is_discount","category"]:
                if col not in df.columns:
                    df[col] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["name","merchant","is_discount","category"])

def _save_labels(df: pd.DataFrame):
    df.to_csv(LABELED_PATH, index=False, encoding="utf-8")

def _text_features(name: str, merchant: str, amount: float) -> str:
    s = f"{name} | merch:{merchant} | sign:{'neg' if amount<0 else 'pos'}"
    s = re.sub(r"\s+"," ",s).strip()
    return s

def train_discount_model():
    df = _load_labels()
    if df.empty:
        return None
    X = [_text_features(n,m,0.0) for n,m in zip(df["name"], df["merchant"])]
    y = df["is_discount"].astype(str).map(lambda v: 1 if str(v).lower() in ("true","1","yes") else 0)
    if y.nunique() < 2:
        return None
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(X,y); joblib.dump(pipe, DISC_MODEL_PATH); return pipe

def train_category_model():
    df = _load_labels()
    df = df[(df["is_discount"].astype(str).str.lower().isin(["false","0","no"])) & (df["category"].astype(str)!="")]
    if df.empty or df["category"].nunique() < 2:
        return None
    X = [_text_features(n,m,1.0) for n,m in zip(df["name"], df["merchant"])]
    y = df["category"].astype(str)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
        ("clf", LogisticRegression(max_iter=300))
    ])
    pipe.fit(X,y); joblib.dump(pipe, CAT_MODEL_PATH); return pipe

def load_discount_model():
    if DISC_MODEL_PATH.exists():
        try:
            return joblib.load(DISC_MODEL_PATH)
        except Exception:
            return None
    return None

def load_category_model():
    if CAT_MODEL_PATH.exists():
        try:
            return joblib.load(CAT_MODEL_PATH)
        except Exception:
            return None
    return None

def ml_predict_is_discount(name: str, merchant: str, amount: float, abstain_threshold=0.8):
    model = load_discount_model()
    if model is None:
        return None, 0.0
    X = [_text_features(name, merchant, amount)]
    proba = float(model.predict_proba(X)[0][1])  # clasa 1 = discount
    if proba >= abstain_threshold:
        return True, proba
    if (1.0 - proba) >= abstain_threshold:
        return False, 1.0 - proba
    return None, proba  # nesigur -> se abține

def ml_predict_category(name: str, merchant: str, amount: float, abstain_threshold=0.6):
    model = load_category_model()
    if model is None:
        return None, {}
    X = [_text_features(name, merchant, amount)]
    proba = model.predict_proba(X)[0]
    labels = model.classes_
    winners = {labels[i]: float(proba[i]) for i in np.argsort(proba)[::-1][:5]}
    top_label = max(winners, key=winners.get)
    if winners[top_label] >= abstain_threshold:
        return top_label, winners
    return None, winners

def log_labeled_examples(rows: pd.DataFrame, merchant: str):
    df = _load_labels()
    recs = []
    for _, r in rows.iterrows():
        name = str(r.get("name") or "")
        cat  = str(r.get("category") or "")
        amt  = float(r.get("amount") or 0.0)
        is_discount = (cat.strip().lower()=="reduceri") or bool(re.search(r"(discount|reducere)", name, re.I) and amt > 0)
        recs.append({"name": name, "merchant": merchant or "", "is_discount": is_discount, "category": cat})
    if recs:
        df = pd.concat([df, pd.DataFrame(recs)], ignore_index=True)
        _save_labels(df)

# ================== LINE-ITEMS EXTRACTION ==================
def auto_category_for_item(name: str, cats_dict: dict) -> str:
    if not name:
        return "Uncategorized"
    blob = re.sub(r"[^a-zăâîșț ]", "", normalize_text(name))
    for cat, kws in (cats_dict or {}).items():
        if not kws or cat.lower()=="income":
            continue
        for kw in kws:
            kw = normalize_text(kw)
            if kw and kw in blob:
                return cat
    builtin = [
        ("Fuel", ["motorina","benzina","diesel","fuel","petrol","omv","mol","lukoil"]),
        ("Food & Groceries", ["snack","chips","biscuit","paine","lapte","oua","lidl","kaufland","carrefour","mega"]),
        ("Restaurants & Coffee", ["cafea","coffee","restaurant","kfc","mcdonald","pizza","wolt"]),
        ("Transport", ["taxi","bolt","uber","bilet","tren","metrou"]),
        ("Health", ["farmacie","catena","helpnet","aspirina","vitamina"]),
        ("Livrare & taxe", ["delivery","livrare","service fee","taxa de livrare","comision","transport tax","handling"]),
    ]
    for cat, kws in builtin:
        for kw in kws:
            if kw in blob:
                return cat
    return "Uncategorized"
def extract_line_items(text: str, total_hint: float | None = None):
    """
    Extrage iteme, ignoră meta-liniile (TOTAL/Card/Apple Pay...),
    păstrează Delivery/Service fee ca cheltuieli și tratează discount ca venit.
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    def amount_at_end(ln: str):
        m = re.search(r"(-?\d+[.,]\d{2})\s*[A-Z]?\s*$", ln)
        return float(m.group(1).replace(",", ".")) if m else None

    def strip_amount(ln: str) -> str:
        return re.sub(r"(-?\d+[.,]\d{2})\s*[A-Z]?\s*$", "", ln).strip(" .:-")

    def parse_qty_unit_price(ln: str):
        """
        Prinde tipare:
        - 2 x 4,50
        - 1 buc x 3.00 3,00
        - 3*2.50 7,50
        Returnează (qty, unit_price, subtotal). Dacă subtotalul nu e în linie, îl calculează.
        """
        m = re.search(
            r"(?P<qty>\d+(?:[.,]\d+)?)\s*"
            r"(?:buc|kg|l|rola|pck|pz|pcs)?\s*"
            r"[x×*]\s*"
            r"(?P<unit>\d+[.,]\d{2})"
            r"(?:\s+(?P<subtotal>\d+[.,]\d{2}))?",
            ln,
            re.I,
        )
        if not m:
            return None
        qty = float(m.group("qty").replace(",", "."))
        unit_price = float(m.group("unit").replace(",", "."))
        if m.group("subtotal"):
            subtotal = float(m.group("subtotal").replace(",", "."))
        else:
            subtotal = round(qty * unit_price, 2)
        return qty, unit_price, subtotal

    def looks_meta(ln: str) -> bool:
        return bool(META_RE.search(ln))

    items = []
    lines = [norm(l) for l in text.splitlines() if norm(l)]
    i = 0
    while i < len(lines):
        ln = lines[i]

        # Delivery/fees
        if re.search(r"\b(delivery|livrare|service\s*fee|tax[ăa]\s*de\s*livrare|comision|transport)\b", ln, re.I):
            amt = amount_at_end(ln)
            if amt is not None:
                name = strip_amount(ln)
                items.append({"name": name, "amount": -abs(amt), "category": "Livrare & taxe"})
                i += 1
                continue

        # Stop înainte de secțiunile card terminal
        if re.search(r"\b(DETALII\s+TRANZACTI|TERMINAL|SUMA\s)\b", ln, re.I):
            break

        if looks_meta(ln) or re.search(r"\bTOTAL\b", ln, re.I):
            i += 1
            continue

        if ONLY_QTY_LINE.match(ln):
            i += 1
            continue

        amt = amount_at_end(ln)
        qty_info = parse_qty_unit_price(ln)

        # nu confunda o linie cu TOTAL-ul general
        if (amt is not None) and (total_hint is not None) and (abs(amt - float(total_hint)) <= 0.01):
            i += 1
            continue

        if qty_info:
            qty, unit_price, subtotal = qty_info

            # Fallback: articol complet pe același rând (ex: "2 x 3,50 7,00")
            if amt is not None and abs(amt - subtotal) <= 0.05:
                name_candidate = strip_amount(ln)
                if not re.search(r"\b(total|tva|card|apple|google|visa|mastercard|rest|ramburs)\b", name_candidate, re.I) and len(name_candidate) >= 2:
                    items.append({"name": name_candidate, "amount": subtotal})
                    i += 1
                    continue

            # Încearcă rândul următor pentru denumire/subtotal
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if not looks_meta(nxt):
                    nxt_amt = amount_at_end(nxt)
                    if nxt_amt is not None and abs(nxt_amt - subtotal) <= 0.05:
                        name = strip_amount(nxt)
                        if name and len(name) >= 2:
                            if re.search(r"\bdiscount|reducere\b", name, re.I):
                                subtotal = abs(subtotal)  # venit
                                items.append({"name": name, "amount": subtotal})
                            else:
                                items.append({"name": name, "amount": subtotal})
                            i += 2
                            continue
            i += 1
            continue

        # fallback simplu: "Nume produs .... 12,34"
        if amt is not None:
            name_candidate = strip_amount(ln)
            if not re.search(r"\b(total|tva|card|ing)\b", name_candidate, re.I) and len(name_candidate) >= 2:
                items.append({"name": name_candidate, "amount": amt})

        i += 1

    # deduplicate consecutive (name+amount)
    deduped = []
    for it in items:
        if not deduped or not (deduped[-1]["name"] == it["name"] and abs(deduped[-1]["amount"] - it["amount"]) < 0.01):
            deduped.append(it)

    if total_hint is not None:
        deduped = [x for x in deduped if abs(x["amount"] - float(total_hint)) > 0.01]

    return deduped
# ================== UI ==================
st.title("💼 Budget App — OCR + AI (categorii & discount)")
tabs = st.tabs(["🧾 Cheltuieli", "💰 Venituri", "📊 Dashboard", "📥 Import", "🧹 Editare tranzacții"])

exp_cats, inc_cats, cats_dict = load_categories()
tx = load_tx()

# ===== TAB 1: Expenses =====
with tabs[0]:
    st.header("🧾 Adaugă Cheltuială din bon (cu linii de produs)")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Upload bon (jpg/png/webp/pdf)", type=["png","jpg","jpeg","webp","pdf"], key="exp_upl")
        use_ocr = st.toggle("🔍 Auto-extrage din bon (OCR)", value=True)
        ocr_suggestion = {"merchant":"", "date":date.today().isoformat(), "amount":0.0, "text":""}
        items = []

        if uploaded and use_ocr:
            try:
                pages = load_images_from_upload(uploaded)
                st.image(pages[0], caption=f"Previzualizare bon (pagina 1 din {len(pages)})", use_column_width=True)

                texts = []
                for idx, im in enumerate(pages, start=1):
                    im_prep = preprocess_for_ocr(im)
                    t = ocr_text_from_pil(im_prep)
                    texts.append(t)

                txt = "\n".join(texts)
                ocr_suggestion["text"] = txt
                ocr_suggestion["date"] = find_date(txt)
                total_detected = find_total(txt)
                if total_detected is not None:
                    ocr_suggestion["amount"] = float(abs(total_detected))
                ocr_suggestion["merchant"] = find_merchant(txt)

                items = extract_line_items(txt, total_hint=total_detected)
                st.subheader("🧾 Linii detectate (editabile)")

                if not items:
                    st.warning("Nu am găsit linii clare. Completează manual în tabelul de mai jos.")
                    df_items = pd.DataFrame([{"name":"","amount":0.0,"category":"Uncategorized"}])
                else:
                    df_items = pd.DataFrame(items)
                    if "category" not in df_items.columns:
                        df_items["category"] = ""

                    merchant_for_ml = ocr_suggestion.get("merchant") or ""
                    for idx, row in df_items.iterrows():
                        nm = str(row["name"]); amt = float(row["amount"] or 0.0)
                        if not df_items.at[idx, "category"]:
                            guess_disc, _ = ml_predict_is_discount(nm, merchant_for_ml, amt)
                            if guess_disc is True:
                                df_items.at[idx, "category"] = "Reduceri"
                        if df_items.at[idx, "category"] in ("", "Uncategorized"):
                            guess_cat, _ = ml_predict_category(nm, merchant_for_ml, amt)
                            if guess_cat:
                                df_items.at[idx, "category"] = guess_cat
                            else:
                                df_items.at[idx, "category"] = auto_category_for_item(nm, cats_dict)

                    mask_disc = df_items["name"].str.contains(r"(discount|reducere)", case=False, na=False)
                    df_items.loc[mask_disc & (df_items["category"].str.strip() == ""), "category"] = "Reduceri"

                strict_mode = st.toggle("🔒 Modul strict (suma pe linii trebuie să egaleze totalul)", value=True)
                tol = st.number_input("Toleranță (RON)", min_value=0.0, max_value=10.0, value=0.50, step=0.10)
                total_input = st.number_input("Total bon (RON)", min_value=0.0, step=0.01, format="%.2f", value=float(ocr_suggestion.get("amount") or 0.0))

                edited = st.data_editor(
                    df_items, num_rows="dynamic", use_container_width=True,
                    column_config={
                        "name": st.column_config.TextColumn("Produs/linie"),
                        "amount": st.column_config.NumberColumn("Sumă (RON)", step=0.01, format="%.2f"),
                        "category": st.column_config.TextColumn("Categorie (poți scrie alta nouă)")
                    },
                    key="edit_items"
                )
                st.caption("💡 Dacă o categorie nu există, scrie numele dorit și o creăm automat la salvare.")

                sum_items = float(edited["amount"].sum()) if not edited.empty else 0.0
                diff = round(total_input - sum_items, 2)
                if abs(diff) <= tol:
                    st.success(f"✅ Suma pe linii ≈ total bon (diferență {diff:+.2f} RON)")
                else:
                    st.warning(f"⚠️ Suma pe linii ({sum_items:.2f} RON) diferă de total ({total_input:.2f} RON) cu {diff:+.2f} RON")

                can_save = (not strict_mode) or (abs(diff) <= tol)
                save_clicked = st.button("✅ Confirmă & Salvează toate liniile ca tranzacții")

                if save_clicked and not can_save:
                    st.error("Modul strict activ: ajustează sumele/totalul sau mărește toleranța ca să continui.")

                if save_clicked and can_save:
                    # învață cuvinte pentru categorii (mic dicționar)
                    new_cats = cats_dict.copy()
                    for _, r in edited.iterrows():
                        cat = (r.get("category") or "").strip()
                        prod = (r.get("name") or "").strip()
                        if cat and prod:
                            key = prod.split()[0].lower()
                            if cat not in new_cats or not isinstance(new_cats[cat], list):
                                new_cats[cat] = []
                            if key and key not in new_cats[cat]:
                                new_cats[cat].append(key)
                    save_categories(new_cats)

                    # log pentru AI
                    log_labeled_examples(edited, ocr_suggestion.get("merchant"))

                    # scrie tranzacțiile
                    for _, r in edited.iterrows():
                        row_amount = float(r.get("amount") or 0.0)
                        row_name = (r.get("name") or "")
                        row_cat  = (r.get("category") or "Uncategorized").strip() or "Uncategorized"

                        is_discount = (row_cat.lower()=="reduceri") or bool(re.search(r"(discount|reducere)", row_name, re.I))
                        if is_discount:
                            final_amount = abs(row_amount)  # venit
                            final_cat = "Reduceri"
                            final_source = "ocr-discount" if items else "manual-discount"
                        else:
                            final_amount = -abs(row_amount)  # cheltuială
                            final_cat = row_cat
                            final_source = "ocr-lineitems" if items else "manual-lineitems"

                        new_row = {
                            "id": uuid.uuid4().hex[:12],
                            "date": ocr_suggestion.get("date"),
                            "merchant": ocr_suggestion.get("merchant") or "Unknown",
                            "amount": final_amount,
                            "currency": "RON",
                            "category": final_cat,
                            "notes": row_name.strip()[:120],
                            "source": final_source,
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        append_row(new_row)

                    _ = train_discount_model()
                    _ = train_category_model()

                    st.success("✅ Am salvat toate liniile ca tranzacții")
                    st.balloons()

                with st.expander("📜 Text OCR (debug)", expanded=False):
                    st.text_area("Rezultat OCR", value=txt, height=200)

            except Exception as e:
                st.error(f"Eroare OCR: {e} — verifică Tesseract.")

    with col2:
        st.subheader("Adăugare rapidă (o singură sumă)")
        defaults = {"merchant":"", "date":date.today().isoformat(), "amount":0.0}
        if "ocr_suggestion" in locals() and ocr_suggestion.get("amount"):
            defaults = ocr_suggestion
        with st.form("exp_form_single", clear_on_submit=True):
            merchant = st.text_input("Comerciant", value=defaults["merchant"])
            try:
                d_default = pd.to_datetime(defaults["date"]).date()
            except Exception:
                d_default = date.today()
            dt_in = st.date_input("Data", value=d_default, key="date_single")
            amount = st.number_input("Sumă totală (RON)", min_value=0.0, step=0.1, format="%.2f", value=float(defaults["amount"]) )
            category = st.text_input("Categorie (dacă vrei totul într-una)", value="Uncategorized")
            notes = st.text_input("Notițe", placeholder="ex: fără detaliere pe linii")
            submitted = st.form_submit_button("💾 Salvează o singură tranzacție")
            if submitted:
                row = {
                    "id": uuid.uuid4().hex[:12],
                    "date": dt_in.isoformat(),
                    "merchant": merchant.strip() or "Unknown",
                    "amount": -abs(float(amount)),
                    "currency": "RON",
                    "category": category.strip() or "Uncategorized",
                    "notes": notes if notes else ("OCR total" if "ocr_suggestion" in locals() and ocr_suggestion.get("amount") else "manual"),
                    "source": "ocr-total" if "ocr_suggestion" in locals() and ocr_suggestion.get("amount") else "manual-expense",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                append_row(row)
                st.success("Cheltuială salvată ✔")

    st.divider()
    tx_local = load_tx()
    if not tx_local.empty:
        recent_exp = tx_local[tx_local["amount"] < 0].sort_values("date", ascending=False).head(50)
        st.subheader("📒 Ultimele cheltuieli")
        st.dataframe(recent_exp, use_container_width=True)
    else:
        st.info("Nu există încă tranzacții.")

# ===== TAB 2: Income =====
with tabs[1]:
    st.header("💰 Adaugă Venit")
    with st.form("inc_form", clear_on_submit=True):
        source_merchant = st.text_input("Sursă venit", placeholder="Salariu / Proiect freelancing / Refund...")
        dt_in = st.date_input("Data", value=date.today(), key="inc_date")
        amount = st.number_input("Sumă (RON)", min_value=0.0, step=0.1, format="%.2f", key="inc_amount")
        category = st.selectbox("Categorie (venit)", [*load_categories()[1]], index=0)
        notes = st.text_input("Notițe", placeholder="ex: luna curentă, proiect X")
        submitted = st.form_submit_button("💾 Salvează venitul")
        if submitted:
            row = {
                "id": uuid.uuid4().hex[:12],
                "date": dt_in.isoformat(),
                "merchant": source_merchant.strip() or "Unknown Income",
                "amount": abs(float(amount)),
                "currency": "RON",
                "category": category,
                "notes": notes,
                "source": "manual-income",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            append_row(row)
            st.success("Venit salvat ✔")

    st.divider()
    tx2 = load_tx()
    if not tx2.empty:
        recent_inc = tx2[tx2["amount"] > 0].sort_values("date", ascending=False).head(50)
        st.subheader("📒 Ultimele venituri")
        st.dataframe(recent_inc, use_container_width=True)
    else:
        st.info("Nu există încă tranzacții.")

# ===== TAB 3: Dashboard =====
with tabs[2]:
    st.header("📊 Dashboard & Balanță")
    tx = load_tx()
    if tx.empty:
        st.info("Adaugă tranzacții ca să vezi dashboardul.")
    else:
        tx["month"] = tx["date"].dt.to_period("M").astype(str)
        months = ["(toate)"] + sorted(tx["month"].dropna().unique().tolist())
        sel_month = st.selectbox("Selectează luna", months, index=0)

        dff = tx.copy()
        if sel_month != "(toate)":
            dff = dff[dff["month"] == sel_month]

        total_income = dff.loc[dff["amount"] > 0, "amount"].sum()
        total_expense = dff.loc[dff["amount"] < 0, "amount"].sum()
        balance = total_income + total_expense

        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Venituri", f"{total_income:,.2f} RON")
        c2.metric("🧾 Cheltuieli", f"{-total_expense:,.2f} RON")
        c3.metric("🧮 Balanță", f"{balance:,.2f} RON")

        st.divider()

        inc_month = dff[dff["amount"] > 0].groupby("month")["amount"].sum().reset_index().rename(columns={"amount":"Income"})
        exp_month = dff[dff["amount"] < 0].copy(); exp_month["amount"] = exp_month["amount"].abs()
        exp_month = exp_month.groupby("month")["amount"].sum().reset_index().rename(columns={"amount":"Expense"})
        merged = pd.merge(inc_month, exp_month, on="month", how="outer").fillna(0.0).sort_values("month")
        st.subheader("📈 Venituri vs Cheltuieli pe lună")
        if not merged.empty:
            st.line_chart(merged.set_index("month"))
        else:
            st.info("Nu sunt suficiente date pe luni pentru grafic.")

        st.subheader("📊 Cheltuieli pe categorii")
        exp_df = dff[dff["amount"] < 0].copy()
        if not exp_df.empty:
            exp_cat = (
                exp_df.assign(amount=lambda x: x["amount"].abs())
                      .groupby("category", as_index=False)["amount"].sum()
                      .sort_values("amount", ascending=False)
            )
            total_exp = float(exp_cat["amount"].sum())
            exp_cat["pct"] = (exp_cat["amount"] / total_exp * 100).round(2)

            view_mode = st.radio("Afișare", ["Coloane", "Cerc (pie)", "Ambele"], horizontal=True, index=2)

            bar = (
                alt.Chart(exp_cat).mark_bar().encode(
                    x=alt.X("category:N", sort="-y", title="Categorie"),
                    y=alt.Y("amount:Q", title="Sumă (RON)"),
                    color=alt.Color("category:N", legend=None, scale=alt.Scale(scheme="category20")),
                    tooltip=[alt.Tooltip("category:N", title="Categorie"),
                             alt.Tooltip("amount:Q", title="Sumă", format=".2f"),
                             alt.Tooltip("pct:Q", title="Procent", format=".2f")],
                ).properties(width="container", height=320)
            )

            pie = (
                alt.Chart(exp_cat).mark_arc().encode(
                    theta=alt.Theta(field="amount", type="quantitative"),
                    color=alt.Color(field="category", type="nominal", legend=alt.Legend(title="Categorie"),
                                    scale=alt.Scale(scheme="category20")),
                    tooltip=[alt.Tooltip("category:N", title="Categorie"),
                             alt.Tooltip("amount:Q", title="Sumă", format=".2f"),
                             alt.Tooltip("pct:Q", title="Procent", format=".2f")],
                ).properties(width="container", height=320)
            )

            if view_mode == "Coloane":
                st.altair_chart(bar, use_container_width=True)
            elif view_mode == "Cerc (pie)":
                st.altair_chart(pie, use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Coloane"); st.altair_chart(bar, use_container_width=True)
                with c2:
                    st.caption("Cerc (pie)"); st.altair_chart(pie, use_container_width=True)
        else:
            st.info("Nu există cheltuieli în intervalul selectat.")

        st.divider()
        st.subheader("📒 Toate tranzacțiile (după filtru)")
        st.dataframe(dff.sort_values("date", ascending=False), use_container_width=True)

# ===== TAB 4: Import Money Manager =====
with tabs[3]:
    st.header("📥 Import din Money Manager (CSV)")
    up = st.file_uploader("Alege fișierul CSV exportat", type=["csv"], key="mm_csv")
    colA, colB, colC = st.columns(3)
    with colA: currency = st.text_input("Monedă", value="RON")
    with colB: assume_expense = st.checkbox("Consideră toate ca cheltuieli dacă nu există Type", value=False)
    with colC: date_fmt = st.selectbox("Format dată", ["Auto (DMY)", "DMY", "MDY", "YMD"], index=0)

    if up is not None:
        try:
            raw = up.read(); df = read_csv_auto_bytes(raw)
            st.caption("Previzualizare CSV (primele 10 rânduri)")
            st.dataframe(df.head(10), use_container_width=True)

            cols_norm = [c.strip().lower() for c in df.columns]
            def pick(cands):
                for c in cands:
                    if c in cols_norm: return df.columns[cols_norm.index(c)]
                return None

            col_date   = pick(["date","data","transaction date","period"])
            col_amount = pick(["amount","sum","value","ron"])
            col_type   = pick(["type","transaction type","income/expense","income expense"])
            col_cat    = pick(["category","categorie"])
            col_note   = pick(["note","memo","remarks","description"])
            col_merch  = pick(["merchant","payee","store","name"])

            if not col_date or not col_amount:
                st.error("CSV-ul trebuie să conțină cel puțin coloanele Date/Period și Amount/RON.")
            else:
                hint = None if date_fmt.startswith("Auto") else date_fmt
                parsed_dates = parse_date_series(df[col_date], hint)
                amounts = parse_amount_series(df[col_amount])

                if col_type:
                    txt = df[col_type].astype(str).map(normalize_text).str.replace(r"[^a-z]+","", regex=True)
                    exp_mask = txt.str.contains(r"^exp|expense|chelt|debit|out|spend", regex=True, na=False)
                    inc_mask = txt.str.contains(r"^inc|income|venit|credit|in|earn", regex=True, na=False)
                    sign = np.where(exp_mask, -1, np.where(inc_mask, 1, np.sign(amounts).replace(0, 1)))
                else:
                    sign = -1 if assume_expense else np.sign(amounts).replace(0, 1)

                final_amount = (amounts.abs() * sign).round(2)
                merch = df[col_merch].fillna("").replace("", "Money Manager") if col_merch else "Money Manager"
                cat = df[col_cat] if col_cat else ""
                note = df[col_note] if col_note else ""

                out = pd.DataFrame({
                    "id": [f"mm{str(i).zfill(8)}" for i in range(len(df))],
                    "date": parsed_dates.dt.date.astype(str),
                    "merchant": merch,
                    "amount": final_amount,
                    "currency": currency,
                    "category": cat,
                    "notes": note,
                    "source": "import-moneymanager",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                out = out[pd.to_datetime(out["date"], errors="coerce").notna() & out["amount"].notna()]

                st.write(f"Rânduri valide pentru import: **{len(out)}**")
                if len(out) > 0 and st.button("✅ Importă în transactions.csv"):
                    append_rows(out)
                    st.success(f"Am importat {len(out)} rânduri ✔")
                    st.balloons()
        except Exception as e:
            st.error(f"Eroare la citirea CSV: {e}")

# ===== TAB 5: Editare tranzacții =====
with tabs[4]:
    st.header("🧹 Editare / ștergere tranzacții")
    df = load_tx()
    if df.empty:
        st.info("Nu există încă tranzacții.")
    else:
        view = df.copy()
        if "date" in view.columns:
            view["date"] = pd.to_datetime(view["date"], errors="coerce").dt.date
        if "amount" in view.columns:
            view["amount"] = pd.to_numeric(view["amount"], errors="coerce")
        if "currency" in view.columns:
            view["currency"] = view["currency"].fillna("RON")

        order_cols = ["id","date","merchant","amount","currency","category","notes","source","created_at"]
        view = view[[c for c in order_cols if c in view.columns]]

        st.caption("Editează celulele dorite (merchant, amount, category, notes, date). Poți adăuga și rânduri noi.")
        edited = st.data_editor(
            view, num_rows="dynamic", use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("date", format="YYYY-MM-DD"),
                "amount": st.column_config.NumberColumn("amount", step=0.01, format="%.2f"),
                "category": st.column_config.TextColumn("category"),
                "merchant": st.column_config.TextColumn("merchant"),
                "notes": st.column_config.TextColumn("notes"),
            },
            key="tx_editor"
        )

        st.divider()
        st.caption("Selectează rândurile pe care vrei să le ștergi:")
        options = {i: f"{edited.loc[i,'date']} • {edited.loc[i,'merchant']} • {edited.loc[i,'amount']:.2f} {edited.loc[i,'currency']}" for i in edited.index}
        to_delete = st.multiselect("Rânduri selectate", list(options.keys()), format_func=lambda k: options[k])

        c1, c2, c3, c4 = st.columns([1,1,1,2])

        if c1.button("💾 Salvează modificările"):
            overwrite_tx(edited)
            st.success("Modificările au fost salvate în transactions.csv.")
            st.rerun()

        if c2.button("🗑️ Șterge rândurile selectate", type="secondary", disabled=len(to_delete)==0):
            new_df = edited.drop(index=to_delete)
            overwrite_tx(new_df)
            st.success(f"Am șters {len(to_delete)} rânduri.")
            st.rerun()

        with c3:
            st.caption("Normalizare semne:")
            if st.button("🔁 Cheltuieli negative, Reduceri pozitive"):
                norm = edited.copy()
                is_disc = norm["category"].astype(str).str.lower().eq("reduceri")
                norm.loc[~is_disc, "amount"] = -norm.loc[~is_disc, "amount"].abs()
                norm.loc[is_disc, "amount"] = norm.loc[is_disc, "amount"].abs()
                overwrite_tx(norm)
                st.success("Am normalizat semnele.")
                st.rerun()

        with c4:
            if st.button("🧽 Elimină duplicate (date+merchant+amount)"):
                deduped = edited.sort_values("created_at").drop_duplicates(subset=["date","merchant","amount"], keep="last")
                removed = len(edited) - len(deduped)
                overwrite_tx(deduped)
                st.success(f"Am eliminat {removed} duplicate.")
                st.rerun()



# ==================== AI FINANCE STRATEGY (Lightweight) ====================
ESSENTIAL_CATS = {
    "Groceries": "essential",
    "Food": "essential",
    "Utilities": "essential",
    "Transport": "essential",
    "Health": "essential",
    "Housing": "essential",
    "Kids": "essential",
    "Pets": "essential",
    "Taxes": "essential",
    "Shopping": "wants",
    "Restaurants": "wants",
    "Entertainment": "wants",
    "Travel": "wants",
    "Subscriptions": "wants",
    "Hobby": "wants",
}

def _classify_row_for_ai(cat: str, merchant: str = "") -> str:
    if not isinstance(cat, str):
        return "unknown"
    return ESSENTIAL_CATS.get(cat, "unknown")

def _month_key_for_ai(ts):
    if pd.isna(ts):
        return None
    try:
        return pd.to_datetime(ts).strftime("%Y-%m")
    except Exception:
        return None

@dataclass
class StrategyInput:
    df: pd.DataFrame
    income_next: float
    rent: float
    loan: float
    ef_target_months: int = 3
    ef_current: float = 0.0
    extra_debt_payment: float = 0.0
    min_savings_pct: float = 0.10
    max_wants_pct: float = 0.30
    prefer_emergency_first: bool = True

@dataclass
class StrategyResult:
    allocations: pd.DataFrame
    narrative: str
    diagnostics: Dict[str, Any]

def _prepare_last_month_for_ai(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "date" in df.columns:
        df = df.copy()
        df["_ym"] = df["date"].apply(_month_key_for_ai)
    else:
        df = df.copy()
        df["_ym"] = None
    months = [m for m in df["_ym"].dropna().unique()]
    if not months:
        return df.iloc[0:0]
    last_month = sorted(months)[-1]
    mdf = df[df["_ym"] == last_month].copy()
    mdf["ew"] = [_classify_row_for_ai(c, m) for c, m in zip(mdf.get("category", ""), mdf.get("merchant", ""))]
    return mdf

def suggest_strategy(inp: StrategyInput) -> StrategyResult:
    df = _prepare_last_month_for_ai(inp.df.copy())

    tx_income_hist = df[df["amount"] > 0]["amount"].sum() if "amount" in df.columns else 0.0
    tx_expenses_hist = -df[df["amount"] < 0]["amount"].sum() if "amount" in df.columns else 0.0

    essential_hist = -df[(df["amount"] < 0) & (df["ew"] == "essential")]["amount"].sum() if "amount" in df.columns else 0.0
    essential_hist = -df[(df["amount"] < 0) & (df.get("ew", "") == "essential")]["amount"].sum()
    unknown_hist = tx_expenses_hist - (essential_hist + wants_hist)

    fixed = float(inp.rent) + float(inp.loan)
    variable_essentials = max(0.0, essential_hist)

    income = float(inp.income_next)
    baseline_needs = fixed + variable_essentials
    surplus = income - baseline_needs

    planned_wants_cap = min(inp.max_wants_pct * income, max(0.0, wants_hist))
    if surplus - planned_wants_cap < 0:
        planned_wants_cap = max(0.0, surplus)
    surplus_after_wants = max(0.0, surplus - planned_wants_cap)

    ef_target = inp.ef_target_months * max(1.0, baseline_needs - wants_hist)
    ef_gap = max(0.0, ef_target - float(inp.ef_current))

    min_savings = inp.min_savings_pct * income

    alloc = []
    alloc.append({"bucket": "Rent", "amount": round(inp.rent, 2), "why": "Locuință (fix)"})
    alloc.append({"bucket": "Loan (min)", "amount": round(inp.loan, 2), "why": "Rată credit (fix)"})
    alloc.append({"bucket": "Essentials (variable)", "amount": round(variable_essentials, 2), "why": "Media luna trecută"})
    alloc.append({"bucket": "Wants (capped)", "amount": round(planned_wants_cap, 2), "why": f"Plafon {int(inp.max_wants_pct*100)}% din venit"})

    remaining = income - sum(a["amount"] for a in alloc)

    ef_alloc = 0.0
    inv_alloc = 0.0
    debt_extra = float(inp.extra_debt_payment or 0.0)
    base_savings = max(min_savings, 0.0)

    if inp.prefer_emergency_first:
        ef_alloc = min(remaining, ef_gap, max(remaining - debt_extra, 0.0))
        rest_after_ef = max(0.0, remaining - ef_alloc - debt_extra)
        inv_alloc = rest_after_ef
    else:
        half = max(0.0, remaining - debt_extra) * 0.5
        ef_alloc = min(half, ef_gap)
        inv_alloc = max(0.0, remaining - debt_extra - ef_alloc)

    if ef_alloc + inv_alloc < base_savings and remaining - debt_extra > 0:
        bump = min(base_savings - (ef_alloc + inv_alloc), max(0.0, remaining - debt_extra - (ef_alloc + inv_alloc)))
        inv_alloc += bump

    if debt_extra > 0:
        debt_extra = min(debt_extra, max(0.0, income - sum(a["amount"] for a in alloc) - ef_alloc - inv_alloc))

    if ef_alloc > 0:
        alloc.append({"bucket": "Emergency Fund", "amount": round(ef_alloc, 2), "why": f"Țintă {inp.ef_target_months} luni"})
    if inv_alloc > 0:
        alloc.append({"bucket": "Investments / T-bills", "amount": round(inv_alloc, 2), "why": "După EF"})
    if debt_extra > 0:
        alloc.append({"bucket": "Loan (extra)", "amount": round(debt_extra, 2), "why": "Rambursare anticipată"})

    df_alloc = pd.DataFrame(alloc)
    df_alloc["pct_of_income"] = (df_alloc["amount"] / max(1.0, income)).round(4)

    bullets = []
    bullets.append(f"Venit estimat luna următoare: {income:,.0f} lei.")
    bullets.append(f"Fixe: chirie {inp.rent:,.0f} + rată {inp.loan:,.0f} = {fixed:,.0f} lei.")
    bullets.append(f"Esențiale variabile ~ {variable_essentials:,.0f} lei; wants plafonate la {int(inp.max_wants_pct*100)}% din venit.")
    bullets.append(f"Țintă fond de urgență: {inp.ef_target_months} luni → {ef_target:,.0f} lei; gap curent: {ef_gap:,.0f} lei.")
    bullets.append(f"Economii minime vizate: {int(inp.min_savings_pct*100)}% din venit.")
    if ef_alloc > 0:
        bullets.append(f"Aloc {ef_alloc:,.0f} lei către fondul de urgență, apoi {inv_alloc:,.0f} lei către investiții/T-bills.")
    else:
        bullets.append(f"Aloc {inv_alloc:,.0f} lei către investiții/T-bills (EF atins sau gap 0).")
    if debt_extra > 0:
        bullets.append(f"Rambursare anticipată: {debt_extra:,.0f} lei/lună.")

    narrative = " • " + "\n • ".join(bullets)

    diags = {
        "income_next": income,
        "baseline_needs": baseline_needs,
        "surplus_pre_wants": surplus,
        "surplus_after_wants": surplus_after_wants,
        "hist": {
            "income_last_month": tx_income_hist,
            "expenses_last_month": tx_expenses_hist,
            "essential_last_month": essential_hist,
            "wants_last_month": wants_hist,
            "unknown_last_month": unknown_hist,
        }
    }
    return StrategyResult(allocations=df_alloc, narrative=narrative, diagnostics=diags)




# ==================== UI: AI Strategy (beta) ====================
try:
    _tx_exists = tx is not None
except Exception:
    try:
        tx = load_tx()
    except Exception:
        tx = pd.DataFrame(columns=["id","date","merchant","amount","currency","category","notes","source","created_at"])

st.markdown("---")
st.header("AI Strategy (beta)")

c1, c2, c3 = st.columns(3)
with c1:
    _income_next = st.number_input("Venit luna următoare (lei)", min_value=0.0, value=20000.0, step=100.0)
    _ef_target_months = st.slider("Țintă fond urgență (luni)", 1, 12, 3)
with c2:
    _rent = st.number_input("Chirie (lei)", min_value=0.0, value=2500.0, step=50.0)
    _loan = st.number_input("Rată credit (lei)", min_value=0.0, value=1585.0, step=50.0)
with c3:
    _ef_current = st.number_input("Fond urgență existent (lei)", min_value=0.0, value=0.0, step=100.0)
    _extra_debt = st.number_input("Rambursare anticipată/lună (lei)", min_value=0.0, value=0.0, step=50.0)

_min_savings_pct = st.slider("Economii minime (% din venit)", 0, 50, 10) / 100.0
_max_wants_pct = st.slider("Plafon cheltuieli wants (% din venit)", 0, 70, 30) / 100.0
_prefer_ef_first = st.checkbox("Prioritizează fondul de urgență înaintea investițiilor", value=True)

if st.button("Generează strategie"):
    _si = StrategyInput(
        df=tx.copy(),
        income_next=_income_next,
        rent=_rent,
        loan=_loan,
        ef_target_months=_ef_target_months,
        ef_current=_ef_current,
        extra_debt_payment=_extra_debt,
        min_savings_pct=_min_savings_pct,
        max_wants_pct=_max_wants_pct,
        prefer_emergency_first=_prefer_ef_first,
    )
    _res = suggest_strategy(_si)
    st.markdown("#### Recomandări")
    st.write(_res.narrative)
    st.markdown("#### Alocări propuse")
    st.dataframe(_res.allocations, use_container_width=True)
    with st.expander("Diagnostic (detalii)"):
        st.json(_res.diagnostics)

