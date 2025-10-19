#!/usr/bin/env python3
"""
Receipt OCR -> transactions.csv
- Offline OCR via Tesseract (install separately and ensure tesseract is in PATH)
- Heuristics to parse date, total, merchant
- Rule-based auto-categorization via categories.yaml
"""
import os, re, csv, sys, uuid, time, json
from datetime import datetime
from pathlib import Path

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image, ImageOps
import yaml

import cv2
import numpy as np
from dateutil import parser as dateparser

BASE = Path(__file__).resolve().parent
INBOX = BASE / "inbox"
ARCHIVE = BASE / "archive"
CSV_PATH = BASE / "transactions.csv"
CATS_PATH = BASE / "categories.yaml"

for p in [INBOX, ARCHIVE]:
    p.mkdir(exist_ok=True)

def load_categories():
    if CATS_PATH.exists():
        with open(CATS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def preprocess_for_ocr(img_path: Path) -> Image.Image:
    # Read with OpenCV, do grayscale, denoise, threshold, then convert to PIL
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # Adaptive threshold to handle various receipts
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    # Slight dilation to connect characters
    kernel = np.ones((1,1), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    pil_img = Image.fromarray(thr)
    return pil_img

def extract_text(pil_img: Image.Image) -> str:
    cfg = "--oem 3 --psm 6"  # LSTM OCR, assume a uniform block of text
    text = pytesseract.image_to_string(pil_img, lang="ron+eng", config=cfg)
    return text

def find_date(text: str):
    # Try multiple regexes common in RO receipts
    patterns = [
        r'(\d{2}[./-]\d{2}[./-]\d{4})',   # 31.12.2024 or 31-12-2024
        r'(\d{4}[./-]\d{2}[./-]\d{2})',   # 2024-12-31
        r'(\d{2}[./-]\d{2}[./-]\d{2})',   # 31-12-24
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                dt = dateparser.parse(m.group(1), dayfirst=True)
                return dt.date().isoformat()
            except Exception:
                pass
    # Fallback: today
    return datetime.now().date().isoformat()

def find_total(text: str):
    # Look for lines with TOTAL / SUMA / PLATA
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    total_candidates = []
    for ln in lines[::-1]:  # bottom-up
        if re.search(r'(total|sum[aă]|plată|plata|de_plătit|de\s*plătit)', ln, re.I):
            m = re.search(r'(-?\d+[.,]\d{2})', ln)
            if m:
                total_candidates.append(m.group(1))
    if not total_candidates:
        # any last number-ish
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
    # Heuristic: first non-empty line, uppercase-ish, without too many digits
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "Unknown Merchant"
    # remove receipt words
    blacklist = ("BON", "FISCAL", "RECEIPT")
    for ln in lines[:5]:
        if any(b in ln.upper() for b in blacklist):
            continue
        if sum(c.isalpha() for c in ln) >= 3:
            return ln[:64]
    return lines[0][:64]

def auto_category(merchant: str, text: str, cats: dict) -> str:
    blob = f"{merchant}\n{text}".lower()
    best = None
    for cat, kws in (cats or {}).items():
        for kw in (kws or []):
            kw = str(kw).lower()
            if kw and kw in blob:
                best = cat
                break
        if best:
            break
    return best or "Uncategorized"

def append_csv(row: dict):
    exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","date","merchant","amount","currency","category","notes","source","created_at"])
        if not exists:
            w.writeheader()
        w.writerow(row)

def process_file(path: Path):
    pil = preprocess_for_ocr(path)
    text = extract_text(pil)
    date_iso = find_date(text)
    total = find_total(text)
    merchant = find_merchant(text)
    cats = load_categories()
    category = auto_category(merchant, text, cats)
    row = {
        "id": uuid.uuid4().hex[:12],
        "date": date_iso,
        "merchant": merchant,
        "amount": total if total is not None else 0.0,
        "currency": "RON",
        "category": category,
        "notes": f"OCR from {path.name}",
        "source": "ocr",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_csv(row)
    # archive
    dest = ARCHIVE / path.name
    path.rename(dest)
    return row

def main():
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith((".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp",".pdf")):
        # single file path
        p = Path(sys.argv[1])
        if not p.exists():
            print("File not found:", p, file=sys.stderr)
            sys.exit(1)
        if p.suffix.lower() == ".pdf":
            print("PDF support: convert pages to images first (e.g., with pdftoppm).")
            sys.exit(1)
        row = process_file(p)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return

    # batch from inbox
    files = sorted(INBOX.glob("*"))
    if not files:
        print("Place images in:", INBOX)
        return
    results = []
    for f in files:
        if f.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"):
            try:
                row = process_file(f)
                results.append(row)
                print("OK:", f.name, "->", row["amount"], row["category"])
            except Exception as e:
                print("ERR:", f.name, e, file=sys.stderr)
    print(f"Imported {len(results)} receipts.")

if __name__ == "__main__":
    main()
