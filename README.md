# Budget OCR Starter Pack

## Ce ai aici
- `transactions.csv` — baza ta de date cu tranzacții
- `categories.yaml` — reguli simple de auto-categorizare
- `receipt_ingest.py` — script care citește poze cu bonuri și pune automat în `transactions.csv`
- `app.py` — interfață ușoară (Streamlit) pentru adăugare manuală/verificare
- `budget_dashboard.xlsx` — Excel cu foi gata (Transactions + Summary) și grafice auto-generate
- `requirements.txt` — dependențe Python

## Cum îl pui la treabă (varianta rapidă)
1) Instalează Python 3.10+
2) (Recomandat) Creează un venv, apoi:
```
pip install -r requirements.txt
```
3) Instalează Tesseract OCR (offline). Pe Windows: instalează de aici: https://github.com/UB-Mannheim/tesseract/wiki
   După instalare, asigură-te că `tesseract` este în PATH. (Opțional poți seta `pytesseract.pytesseract.tesseract_cmd` în script.)
4) Rulează OCR batch: pune imaginile în folderul `inbox/` (îl crează scriptul), apoi:
```
python receipt_ingest.py
```
   Fiecare bon importat apare în `transactions.csv`, iar imaginea e mutată în `archive/`.

## Interfață pentru verificare/adăugare manuală
```
streamlit run app.py
```
Se deschide în browser. Încarci poză, completezi câmpurile, Save → se adaugă în `transactions.csv`.
Pe aceeași pagină vezi un tabel cu ultimele tranzacții și grafice simple.

## Dashboard în Excel
Deschide `budget_dashboard.xlsx`. Are deja:
- grafic coloană pe categorii
- grafic linie pe lună
Apasă "Refresh" după ce actualizezi `transactions.csv` (poți re-rula generatorul din `build_excel_dashboard` dacă vrei să refaci fișierul).

## Notițe utile
- Regex-urile caută **TOTAL / SUMĂ / PLATĂ** și formate de dată RO.
- Dacă OCR nu prinde perfect, editezi din app sau direct în CSV.
- **Categorii**: adaugă cuvinte-cheie în `categories.yaml` ca să-ți învețe magazinele tale.
- PDF bonuri: convertește în imagini (ex: `pdftoppm -png input.pdf out`).

## Structura CSV
id,date,merchant,amount,currency,category,notes,source,created_at

Gata. Spor la organizat banii 💸
