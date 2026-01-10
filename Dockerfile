# =========================
# Dockerfile CPU - Streamlit App (Whisper / NLP)
# =========================

FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifacts/models/deberta_dreaddit_best \
    MAX_LENGTH=128

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY app/ ./app/
# Copier les données (si existantes)
COPY data/ ./data/

# Copier le modèle
COPY artifacts/models/deberta_dreaddit_best/ ./artifacts/models/deberta_dreaddit_best/
RUN echo "Model files:" && ls -lah ./artifacts/models/deberta_dreaddit_best/

# Exposer le port utilisé par Hugging Face
EXPOSE 7860

# Lancer Streamlit
CMD ["streamlit", "run", "app/app_streamlit.py", "--server.port=7860", "--server.address=0.0.0.0"]
