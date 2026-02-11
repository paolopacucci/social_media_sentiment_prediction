import os
import time
import random
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from prometheus_client import Gauge, start_http_server

MODEL_ID = os.getenv("MODEL_ID", "paolopacucci/sentiment-roberta").strip()

DATA_PATH = Path(os.getenv("PERF_DATA_PATH", "data/processed/performance_monitoring_batch.csv"))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "60"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "64"))

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

PERF_ACC = Gauge("performance_accuracy", "Accuracy on performance monitoring mini-batch")
PERF_MACRO_F1 = Gauge("performance_macro_f1", "Macro F1 on performance monitoring mini-batch")
PERF_N = Gauge("performance_batch_size", "Batch size used for performance monitoring")


def load_processed_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed performance file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    return df

def predict_batch(model, tokenizer, texts):
    y_pred = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            chunk = texts[i : i + 16]
            enc = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            out = model(**enc)
            preds = torch.argmax(out.logits, dim=-1).cpu().numpy().tolist()
            y_pred.extend(preds)
    return y_pred


def main():
    print(f"[performance_exporter] Loading model from HF: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, use_safetensors=True)
    model.eval()

    df = load_processed_data()
    texts_all = df["text"].tolist()
    y_true_all = df["label"].map(LABEL2ID).tolist()

    print(f"[performance_exporter] Loaded {len(df)} rows from {DATA_PATH}")
    print("[performance_exporter] Serving metrics on http://0.0.0.0:8001/metrics")
    start_http_server(8001)

    while True:
        idx = random.sample(range(len(texts_all)), k=min(BATCH_SIZE, len(texts_all)))
        texts = [texts_all[i] for i in idx]
        y_true = [y_true_all[i] for i in idx]

        y_pred = predict_batch(model, tokenizer, texts)

        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")

        PERF_ACC.set(float(acc))
        PERF_MACRO_F1.set(float(mf1))
        PERF_N.set(int(len(texts)))

        print(f"[performance_exporter] acc={acc:.3f} macro_f1={mf1:.3f} n={len(texts)}")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
