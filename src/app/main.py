# Entrypoint dell'applicazione FastAPI.
# Questo file inizializza il servizio di serving, carica modello e tokenizer ed espone gli endpoint principali.

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.app import model_loader
from src.app.metrics import (
    MODEL_INFERENCE_SECONDS,
    PREDICTED_LABEL_TOTAL,
    PREDICTION_CONFIDENCE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.app.schemas import PredictItem, PredictRequest, PredictResponse

# Inizializza l'app FastAPI.
app = FastAPI(title="Sentiment API", version="1.0")


# Endpoint root (/) per rendere la homepage della Space più chiara.
# Senza questo endpoint, la pagina Space mostrava {"detail":"Not Found"} perché l’API esponeva solo /health, /metrics e /predict.
@app.get("/")
def root() -> dict:
    return {
        "message": "Sentiment RoBERTa API is running",
        "available_endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "predict": "/predict",
        },
    }


# Carica il modello all'avvio del servizio.
# A questo modo si evita l'inizializzazione ad ogni richiesta rendendo l'inferenza più efficiente.
@app.on_event("startup")
def startup_event() -> None:
    try:
        model_loader.load_model()
        print(f"[api] model loaded from: {model_loader.model_source}")
    except Exception as e:
        print(f"[api] ERROR loading model: {e}")


# Esegue check dell'API e restituisce anceh se il modello e il tokenizer sono caricati correttamente.
@app.get("/health")
def health() -> dict:
    ok = (model_loader.model is not None) and (model_loader.tokenizer is not None)
    return {
        "status": "ok",
        "model_loaded": ok,
        "model_source": model_loader.model_source,
    }


# Espone tutte le metriche Prometheus.
@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Endpoint principale di inferenza batch.
# Riceve una lista di testi, verifica se il modello è disponibile, esegue predizione e aggiorna le metriche.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    endpoint = "/predict"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()

    # Controlla che la batch contenga testi.
    if not req.texts:
        raise HTTPException(status_code=400, detail="Texts must be a non-empty list")

    # Misura la latenza endpoint/predict.
    start = time.perf_counter()

    # Verifica che modello e tokenizer siano caricati.
    if model_loader.model is None or model_loader.tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Esegue inference e restituisce id delle classi predette, label testuali, score/confidence, 
    # tempo di inferenza del modello e confidence media del modello.
    pred_ids, labels, scores, infer_s, avg_conf = model_loader.predict_texts(req.texts)

    # Aggiorna le metriche Prometheus osservate durante lea richiesta.
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    MODEL_INFERENCE_SECONDS.observe(infer_s)
    PREDICTION_CONFIDENCE.set(avg_conf)

    # Conta quante predizioni appartengono ad ogni label. 
    for lab in labels:
        PREDICTED_LABEL_TOTAL.labels(label=lab).inc()

    # Converte l'output del model_loader in una lista PredictItem così da poter essere correttamente usati dall'endpoint/predict.
    outputs = []
    for i, lab, sc in zip(pred_ids, labels, scores):
        outputs.append(PredictItem(label_id=int(i), label=str(lab), score=float(sc)))

    return PredictResponse(outputs=outputs)