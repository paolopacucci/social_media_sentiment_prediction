from pathlib import Path
import json

import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch


MODEL_DIR = Path("artifacts/model")
TEST_PATH = Path("data/split/test.csv")
REPORTS_DIR = Path("reports/baseline")

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    # 1) load test
    df = pd.read_csv(TEST_PATH)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].map(LABEL2ID).tolist()

    # 2) load model + tokenizer (usa safetensors se presente)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, use_safetensors=True)
    model.eval()

    # 3) predict (batch semplice)
    batch_size = 16
    y_pred = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=64,   # coerente con training
                return_tensors="pt",
            )
            outputs = model(**enc)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()
            y_pred.extend(preds)

    # 4) metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_name": "model",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "n_test": int(len(df)),
    }

    with open(REPORTS_DIR / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 5) confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)

    plt.figure(figsize=(6, 6))
    disp.plot(values_format="d")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()

    print("OK evaluation saved in reports/")


if __name__ == "__main__":
    main()
