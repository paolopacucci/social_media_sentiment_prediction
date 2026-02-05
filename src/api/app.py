from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = Path("artifacts/model")
MAX_LENGTH = 64
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


app = FastAPI(title="Sentiment API", version="1.0")

tokenizer = None
model = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


def load_model():
    global tokenizer, model

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}. Run training first to create artifacts/model."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        use_safetensors=True,
        local_files_only=True,
    )
    model.eval()


@app.on_event("startup")
def startup_event():
    # carica modello all'avvio
    try:
        load_model()
    except Exception as e:
        # non facciamo crashare tutto: health rimane disponibile
        print(f"ERROR loading model: {e}")


@app.get("/health")
def health():
    ok = (tokenizer is not None) and (model is not None)
    return {"status": "ok", "model_loaded": ok}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.text is None or req.text.strip() == "":
        raise HTTPException(status_code=400, detail="text must be a non-empty string")

    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="model not loaded (run training first)")

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

    return {"label": ID2LABEL[pred_id], "score": score}
