# =========================
# Dockerfile CPU - Dreaddit DeBERTa + FastAPI
# =========================

FROM python:3.10-slim

# Variables Python et timeouts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifacts/models/deberta_dreaddit_best \
    MAX_LENGTH=128 \
    MODEL_LOAD_TIMEOUT=180

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

# Copier le modèle et vérifier son contenu
COPY artifacts/models/deberta_dreaddit_best/ ./artifacts/models/deberta_dreaddit_best/
RUN echo "Model files:" && ls -lah ./artifacts/models/deberta_dreaddit_best/

# Exposer le port FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=240s --retries=3 \
  CMD curl -f http://localhost:8000/liveness || exit 1

# Démarrage avec timeouts augmentés
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300", "--timeout-graceful-shutdown", "30"]