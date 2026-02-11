import os
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


MAX_LENGTH = 64
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
DEFAULT_MODEL_ID = "paolopacucci/sentiment-roberta"

app = FastAPI(title="Sentiment API", version="1.0")

tokenizer = None
model = None
model_id_loaded = None


# -------------------------
# Prometheus Metrics
# -------------------------
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions",
)

PREDICTIONS_BY_LABEL = Counter(
    "predictions_by_label_total",
    "Predictions grouped by predicted label",
    ["label"],
)

PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Prediction confidence score (max softmax prob)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latency of /predict endpoint in seconds",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
)


# -------------------------
# Schemas
# -------------------------
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


# -------------------------
# Model Loading
# -------------------------
def load_model_from_hf():
    global tokenizer, model, model_id_loaded

    model_id = os.getenv("MODEL_ID", DEFAULT_MODEL_ID).strip()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        use_safetensors=True,
    )
    model.eval()
    model_id_loaded = model_id


@app.on_event("startup")
def startup_event():
    try:
        load_model_from_hf()
    except Exception as e:
        # non crashiamo l'app: /health rimane disponibile e mostra model_loaded=false
        print(f"ERROR loading model from HF: {e}")


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    ok = (tokenizer is not None) and (model is not None)
    MODEL_LOADED.set(1 if ok else 0)
    return {"status": "ok", "model_loaded": ok, "model_id": model_id_loaded}


@app.get("/metrics")
def metrics():
    # Prometheus scrape endpoint
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    if req.text is None or req.text.strip() == "":
        raise HTTPException(status_code=400, detail="text must be a non-empty string")

    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="model not loaded")

    enc = tokenizer(
        req.text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        score = float(probs[pred_id].item())

    label = ID2LABEL[pred_id]

    # Metrics update
    PREDICTIONS_TOTAL.inc()
    PREDICTIONS_BY_LABEL.labels(label=label).inc()
    PREDICTION_SCORE.observe(score)
    PREDICT_LATENCY.observe(time.time() - start)

    return {"label": label, "score": score}
