
#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "transactions.csv"

def parse_args():
    p = argparse.ArgumentParser(description="Flip positive amounts to EXPENSES (negative) with optional filters.")
    p.add_argument("--all-positive", action="store_true", help="Flip ALL positive amounts to negative (expenses).")
    p.add_argument("--merchant-contains", type=str, default=None, help="Only flip rows where merchant contains this text (case-insensitive).")
    p.add_argument("--date-from", type=str, default=None, help="Flip rows with date >= this (YYYY-MM-DD).")
    p.add_argument("--date-to", type=str, default=None, help="Flip rows with date <= this (YYYY-MM-DD).")
    p.add_argument("--source-contains", type=str, default=None, help="Only flip rows where source contains this text.")
    p.add_argument("--dry-run", action="store_true", help="Show what would change without saving.")
    return p.parse_args()

def main():
    args = parse_args()
    if not CSV_PATH.exists():
        print("transactions.csv not found next to this script.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("No data in transactions.csv")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    mask = df["amount"] > 0  # candidates to flip to expense
    if args.merchant_contains:
        mask &= df["merchant"].fillna("").str.contains(args.merchant_contains, case=False, na=False)
    if args.source_contains:
        mask &= df["source"].fillna("").str.contains(args.source_contains, case=False, na=False)
    if args.date_from:
        mask &= df["date"] >= pd.to_datetime(args.date_from, errors="coerce")
    if args.date_to:
        mask &= df["date"] <= pd.to_datetime(args.date_to, errors="coerce")

    if not args.all_positive and args.merchant_contains is None and args.source_contains is None and args.date_from is None and args.date_to is None:
        print("Nothing to do. Specify --all-positive or a filter.", file=sys.stderr)
        sys.exit(1)

    to_flip = df[mask]
    if to_flip.empty:
        print("No rows matched.")
        return

    print(f"Rows to flip: {len(to_flip)}")
    # Flip amounts
    df.loc[mask, "amount"] = -df.loc[mask, "amount"].abs()
    # Mark source
    df.loc[mask, "source"] = df.loc[mask, "source"].fillna("") + ";fix-sign"

    if args.dry_run:
        print(df[mask][["date","merchant","amount","category","source"]].head(20).to_string(index=False))
        print("Dry run complete. No changes saved.")
        return

    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print("Saved. Flipped the sign for positive rows matching filters.")

if __name__ == "__main__":
    main()
