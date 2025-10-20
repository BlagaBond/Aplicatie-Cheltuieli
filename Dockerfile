# Folosim 3.12 ca sa prindem wheel-uri precompilate pt PyMuPDF
FROM python:3.12-slim

# 1) Instalăm Tesseract + limba română
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ron \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Instalăm dependențele Python
COPY requirements.txt .
# Încercăm să luăm wheel-ul gata făcut pentru PyMuPDF; apoi restul
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --only-binary=:all: PyMuPDF==1.24.10 \
 || true
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copiem aplicația
COPY . .

# 4) Pornim Streamlit
ENV PYTHONUNBUFFERED=1
CMD bash -lc 'streamlit run app.py --server.port ${PORT:-10000} --server.address 0.0.0.0'
