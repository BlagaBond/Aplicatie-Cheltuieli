
#!/usr/bin/env python3
# Import a CSV exported from the "Money Manager" app and append to transactions.csv.
# Usage:
#   python import_moneymanager_csv.py path\to\export.csv [--currency RON] [--assume-expense] [--date-format DMY|MDY|YMD]

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "transactions.csv"

def parse_args():
    p = argparse.ArgumentParser(description="Import Money Manager CSV into transactions.csv")
    p.add_argument("csv_path", help="Path to Money Manager CSV export")
    p.add_argument("--currency", default="RON", help="Currency code, default RON")
    p.add_argument("--assume-expense", action="store_true", help="If no Type field, treat all rows as expenses")
    p.add_argument("--date-format", choices=["DMY","MDY","YMD"], default=None, help="Force a date format if auto-parse fails")
    return p.parse_args()

def read_csv_auto(path):
    # try multiple encodings
    for enc in ["utf-8-sig", "utf-16", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="ignore")

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

def main():
    args = parse_args()
    in_path = Path(args.csv_path)
    if not in_path.exists():
        print("Input CSV not found:", in_path, file=sys.stderr); sys.exit(1)

    df = read_csv_auto(in_path)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    def first_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    col_date   = first_col(["date","data","transaction date"])
    col_type   = first_col(["type","transaction type"])
    col_amount = first_col(["amount","sum","value"])
    col_cat    = first_col(["category","categorie"])
    col_note   = first_col(["note","memo","remarks","description"])
    col_merch  = first_col(["merchant","payee","store","name"])

    if not col_date or not col_amount:
        print("CSV must contain at least Date and Amount columns.", file=sys.stderr)
        print("Detected columns:", list(df.columns), file=sys.stderr)
        sys.exit(1)

    # Parse dates
    parsed_dates = parse_date_series(df[col_date], args.date_format)

    # Amounts
    amounts = pd.to_numeric(
        df[col_amount].astype(str).str.replace("\u00A0","").str.replace(" ", "").str.replace(",", "."),
        errors="coerce"
    )

    # Determine sign
    if col_type:
        t = df[col_type].astype(str).str.lower()
        sign = np.where(t.str.contains("exp", na=False), -1,
                np.where(t.str.contains("inc", na=False), 1, np.sign(amounts).fillna(1)))
    else:
        if args.assume_expense:
            sign = -1
        else:
            sign = np.sign(amounts).replace(0, 1).fillna(1)

    final_amount = (amounts.abs() * sign).round(2)

    # Merchant/Payee
    if col_merch:
        merch = df[col_merch].fillna("").replace("", "Money Manager")
    else:
        merch = "Money Manager"

    # Category & Notes
    cat = df[col_cat] if col_cat else ""
    note = df[col_note] if col_note else ""

    out = pd.DataFrame({
        "id": [f"mm{str(i).zfill(8)}" for i in range(len(df))],
        "date": parsed_dates.dt.date.astype(str),
        "merchant": merch,
        "amount": final_amount,
        "currency": args.currency,
        "category": cat,
        "notes": note,
        "source": "import-moneymanager",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # Filter invalid
    out = out[pd.to_datetime(out["date"], errors="coerce").notna() & out["amount"].notna()]
    if out.empty:
        print("No valid rows to import.")
        return

    if CSV_PATH.exists():
        out.to_csv(CSV_PATH, mode="a", header=False, index=False, encoding="utf-8")
    else:
        out.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"Imported {len(out)} rows into transactions.csv")

if __name__ == "__main__":
    main()
