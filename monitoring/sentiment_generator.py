import os
import time
import random
from pathlib import Path

import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://api:8000").strip()
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", "15"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
DATA_PATH = Path(os.getenv("SENTIMENT_DATA_PATH", "data/processed/sentiment_monitoring_batch_text.csv"))


def load_texts(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sentiment processed file: {path}")

    df = pd.read_csv(path)
    return df["text"].tolist()


def pick_batch(texts: list[str], batch_size: int) -> list[str]:
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be > 0")
    n = min(batch_size, len(texts))
    return random.sample(texts, k=n)


def send_predict_request(api_url: str, text: str) -> int:
    r = requests.post(f"{api_url}/predict", json={"text": text}, timeout=30)
    return r.status_code


def run_loop(texts: list[str]) -> None:
    print(f"[sentiment_generator] API_URL={API_URL} interval={INTERVAL_SECONDS}s batch={BATCH_SIZE}")
    print(f"[sentiment_generator] Loaded texts: {len(texts)} from {DATA_PATH}")

    while True:
        batch = pick_batch(texts, BATCH_SIZE)
        ok = 0
        for t in batch:
            try:
                status = send_predict_request(API_URL, t)
                if status == 200:
                    ok += 1
            except Exception as e:
                print(f"[sentiment_generator] ERROR: {e}")

        print(f"[sentiment_generator] sent={len(batch)} ok={ok}")
        time.sleep(INTERVAL_SECONDS)


def main():
    texts = load_texts(DATA_PATH)
    run_loop(texts)


if __name__ == "__main__":
    main()
