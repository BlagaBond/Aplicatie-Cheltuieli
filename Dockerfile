FROM python:3.13-slim

# 1) Instalează Tesseract + limba română
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ron \
 && rm -rf /var/lib/apt/lists/*

# 2) Setează directorul de lucru
WORKDIR /app

# 3) Instalează toate dependențele Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copiază aplicația
COPY . .

# 5) Rulează Streamlit pe portul dat de Render
ENV PYTHONUNBUFFERED=1
CMD bash -lc 'streamlit run app.py --server.port ${PORT:-10000} --server.address 0.0.0.0'
