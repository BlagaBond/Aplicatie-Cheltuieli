
# sb_ocr_persist.py
# Drop-in overrides for OCR robustness + Supabase persistence for Aplicatie-Cheltuieli
# Usage (in app.py, after imports): 
#     from sb_ocr_persist import apply_overrides
#     apply_overrides(globals())

from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, re, json
import pandas as pd

# --- OCR override ---
def _ocr_text_from_pil_override(pytesseract, Image, pil_image):
    """Robust OCR: try ron+eng, then ron, then eng; return '' on failure."""
    cfg = "--oem 3 --psm 6"
    candidate_langs = ["ron+eng", "ron", "eng"]
    # Determine available langs (best-effort)
    langs_to_try = candidate_langs
    try:
        available = pytesseract.get_languages(config="")
        langs_to_try = [lang for lang in candidate_langs if all(l in available for l in lang.split("+"))] or candidate_langs
    except Exception:
        pass
    # Try sequence
    for lang in langs_to_try:
        try:
            return pytesseract.image_to_string(pil_image, lang=lang, config=cfg)
        except pytesseract.TesseractNotFoundError:
            # On Windows default install path
            fallback = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            try:
                import pathlib
                if pathlib.Path(fallback).exists():
                    pytesseract.pytesseract.tesseract_cmd = fallback
                    continue
            except Exception:
                pass
        except Exception:
            # try next language
            continue
    return ""

# --- Supabase helpers ---
def _sb_enabled(env=os.environ):
    return bool(env.get("SUPABASE_URL")) and bool(env.get("SUPABASE_ANON_KEY"))

def _sb_headers(env=os.environ):
    return {
        "apikey": env.get("SUPABASE_ANON_KEY", ""),
        "Authorization": f"Bearer {env.get('SUPABASE_ANON_KEY','')}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def _sb_table_url(table: str, env=os.environ):
    base = env.get("SUPABASE_URL", "").rstrip("/")
    return f"{base}/rest/v1/{table}"

def _sb_fetch_tx(user_id: str, table="transactions", env=os.environ) -> pd.DataFrame:
    import requests
    url = _sb_table_url(table, env)
    headers = _sb_headers(env)
    # filter by user_id; order by date desc
    params = {
        "user_id": f"eq.{user_id}",
        "order": "date.desc,created_at.desc"
    }
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["date","merchant","category","amount","currency","notes","raw"])
    df = pd.DataFrame(data)
    # normalize columns
    for col in ["date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df

def _sb_upsert_rows(user_id: str, df: pd.DataFrame, table="transactions", env=os.environ) -> pd.DataFrame:
    """Insert rows for user_id; returns inserted rows. Expects columns date, merchant, category, amount, currency, notes, raw."""
    import requests
    if df is None or df.empty:
        return df
    payload = df.copy()
    payload["user_id"] = user_id
    # Ensure expected columns exist
    for c in ["date","merchant","category","amount","currency","notes","raw"]:
        if c not in payload.columns:
            payload[c] = None
    # Convert date -> ISO
    if "date" in payload.columns:
        payload["date"] = pd.to_datetime(payload["date"], errors="coerce").dt.date.astype(str)
    # Ensure currency default
    payload["currency"] = payload["currency"].fillna("RON")
    # JSON serialize 'raw' cell-wise when dict/list
    if "raw" in payload.columns:
        payload["raw"] = payload["raw"].apply(lambda x: x if isinstance(x, (str, type(None))) else json.dumps(x, ensure_ascii=False))
    url = _sb_table_url(table, env)
    headers = _sb_headers(env)
    r = requests.post(url, headers=headers, data=payload.to_json(orient="records", force_ascii=False).encode("utf-8"), timeout=30)
    r.raise_for_status()
    out = r.json() if r.content else []
    return pd.DataFrame(out)

def apply_overrides(ns: dict):
    """Patch functions in the caller module namespace.
    - Replaces ocr_text_from_pil
    - Wraps load_tx and append_rows to use Supabase when configured.
    """
    # 1) OCR override
    def ocr_text_from_pil(pil_image):
        import pytesseract
        from PIL import Image
        return _ocr_text_from_pil_override(pytesseract, Image, pil_image)
    ns["ocr_text_from_pil"] = ocr_text_from_pil

    # 2) Persistence overrides
    # Keep references to originals (fallback to CSV)
    orig_load_tx = ns.get("load_tx")
    orig_append_rows = ns.get("append_rows")

    def load_tx():
        import streamlit as st
        if _sb_enabled() and "user" in st.session_state and st.session_state.get("user", {}).get("id"):
            try:
                uid = st.session_state["user"]["id"]
                return _sb_fetch_tx(uid)
            except Exception as e:
                st.warning(f"Supabase indisponibil, se revine la CSV. Motiv: {e}")
        # fallback
        if callable(orig_load_tx):
            return orig_load_tx()
        return pd.DataFrame(columns=["date","merchant","category","amount","currency","notes","raw"])
    ns["load_tx"] = load_tx

    def append_rows(df: pd.DataFrame):
        import streamlit as st
        # Save to Supabase if enabled
        if _sb_enabled() and "user" in st.session_state and st.session_state.get("user", {}).get("id"):
            try:
                uid = st.session_state["user"]["id"]
                inserted = _sb_upsert_rows(uid, df)
                # Return inserted (or df) so callers can refresh if needed
                return inserted if not inserted.empty else df
            except Exception as e:
                st.warning(f"Supabase indisponibil la scriere, se salveaza in CSV. Motiv: {e}")
        # Fallback to original append (CSV)
        if callable(orig_append_rows):
            return orig_append_rows(df)
        return df
    ns["append_rows"] = append_rows
