FROM python:3.11-slim

WORKDIR /app

# Copia requirements e installa
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice e (quando esiste) il modello salvato
COPY src ./src
COPY artifacts/model ./artifacts/model

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
