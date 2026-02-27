import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import HF_TOKEN, MAX_LENGTH, MODEL_ID

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

tokenizer = None
model = None
device = None
model_source = None


def load_model() -> None:
    global tokenizer, model, device, model_source

    if not MODEL_ID:
        raise ValueError("MODEL_ID is empty. Set it in config/.env.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN or None,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_source = MODEL_ID


def predict_texts(
    texts: list[str],
) -> tuple[list[int], list[str], list[float], float, float]:
    if tokenizer is None or model is None or device is None:
        raise RuntimeError("Model not loaded")

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**enc).logits
    infer_s = time.perf_counter() - t0

    probs = torch.softmax(logits, dim=-1)
    scores, pred_ids = torch.max(probs, dim=-1)

    pred_ids = pred_ids.cpu().tolist()
    scores = scores.cpu().tolist()
    labels = [ID2LABEL[i] for i in pred_ids]

    avg_conf = float(sum(scores) / len(scores)) if scores else 0.0
    return pred_ids, labels, scores, infer_s, avg_conf