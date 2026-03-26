# Immagine base con Python 3.11.
FROM python:3.11-slim

# Directory di lavoro del container.
WORKDIR /app

# Variabili d'ambiente:
# - evita file .pyc inutili;
# - forza output immediato nei log;
# - centralizza la cache Hugging Face / Transformers dentro il container.    
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_CACHE=/app/.hf_cache

# Copia i file requirements.
COPY requirements-cpu.txt .
COPY requirements.txt .

# Installa le dipendenze specifiche per CPU e poi quelle generali.
RUN pip install --no-cache-dir -r requirements-cpu.txt && \
    pip install --no-cache-dir -r requirements.txt

# Copia il codice sorgente dell'applicazione.    
COPY src ./src

# Espone la porta usata dal servizio FastAPI
EXPOSE 8000

# Avvia l'API tramite uvicorn.
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]