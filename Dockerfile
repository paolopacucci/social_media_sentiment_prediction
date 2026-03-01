FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_CACHE=/app/.hf_cache

COPY requirements-cpu.txt .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements-cpu.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]