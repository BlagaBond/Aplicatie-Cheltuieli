# diag.py — rulează: python diag.py
import sys, platform, importlib
print("Python:", sys.version)
print("Platform:", platform.platform())

mods = [
    "streamlit",
    "pytesseract",
    "cv2",
    "PIL",
    "pandas",
    "yaml",
    "dateutil",
    "fitz",  # PyMuPDF
    "sklearn",
    "joblib",
    "altair",
    "numpy",
]

for m in mods:
    try:
        mod = importlib.import_module(m if m != "PIL" else "PIL.Image")
        ver = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        print(f"[OK] {m} -> {ver}")
    except Exception as e:
        print(f"[ERR] {m} -> {e}")

# Tesseract check
try:
    import pytesseract, shutil
    path = shutil.which("tesseract")
    print("Tesseract PATH:", path)
    try:
        v = pytesseract.get_tesseract_version()
        print("Tesseract version:", v)
    except Exception as e:
        print("Tesseract version read error:", e)
except Exception as e:
    print("pytesseract import error:", e)
