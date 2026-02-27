import time
import random
from pathlib import Path

import pandas as pd
import requests
from prometheus_client import Gauge, Counter, start_http_server

from src.data.prepare_data import prepare_sentiment_monitoring_data
from src.config import (
    API_URL,
    TEXT_COL,
    SENTIMENT_INTERVAL_SECONDS,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_METRICS_PORT,
    SENTIMENT_MONITORING_TEXT_PATH,
)

LABELS = ["negative", "neutral", "positive"]

LAST_BATCH_COUNT = Gauge("sentiment_last_batch_count", "Counts in last batch", ["label"])
LAST_BATCH_SIZE = Gauge("sentiment_last_batch_size", "Size of last batch")
LAST_BATCH_TS = Gauge("sentiment_last_batch_timestamp", "Unix ts of last batch")
BATCHES_TOTAL = Counter("sentiment_batches_total", "Total generated sentiment batches")


def load_processed_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed sentiment monitoring file: {path}")
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing required column: {TEXT_COL}")
    return df


def sample_batch(texts_all: list[str], batch_size: int) -> list[str]:
    k = min(batch_size, len(texts_all))
    return random.sample(texts_all, k=k)


def call_predict_batch(api_url: str, batch_texts: list[str]) -> dict:
    r = requests.post(f"{api_url}/predict", json={"texts": batch_texts}, timeout=30)
    r.raise_for_status()
    return r.json()


def reset_last_batch_gauges() -> None:
    for lab in LABELS:
        LAST_BATCH_COUNT.labels(label=lab).set(0)


def main() -> None:
    print(f"[sentiment_generator] API_URL={API_URL}")
    print(f"[sentiment_generator] DATA_PATH={SENTIMENT_MONITORING_TEXT_PATH}")
    print(f"[sentiment_generator] batch_size={SENTIMENT_BATCH_SIZE} every {SENTIMENT_INTERVAL_SECONDS}s")
    print(f"[sentiment_generator] metrics on :{SENTIMENT_METRICS_PORT}/metrics")

    start_http_server(SENTIMENT_METRICS_PORT)

    if not SENTIMENT_MONITORING_TEXT_PATH.exists():
        print("[sentiment_generator] processed file missing -> preparing it now")
        prepare_sentiment_monitoring_data()

    df = load_processed_data(SENTIMENT_MONITORING_TEXT_PATH)
    texts_all = df[TEXT_COL].astype(str).str.strip()
    texts_all = texts_all[texts_all != ""].tolist()

    if not texts_all:
        raise ValueError("No valid texts found")

    while True:
        batch_texts = sample_batch(texts_all, SENTIMENT_BATCH_SIZE)

        reset_last_batch_gauges()

        try:
            payload = call_predict_batch(API_URL, batch_texts)
            outs = payload.get("outputs", [])

            counts = {lab: 0 for lab in LABELS}
            for o in outs:
                lab = o.get("label")
                if lab in counts:
                    counts[lab] += 1

            for lab, c in counts.items():
                LAST_BATCH_COUNT.labels(label=lab).set(int(c))

            LAST_BATCH_SIZE.set(int(len(outs)))
            LAST_BATCH_TS.set(int(time.time()))
            BATCHES_TOTAL.inc()

            print(f"[sentiment_generator] last_batch n={len(outs)} counts={counts}")

        except Exception as e:
            print(f"[sentiment_generator] error calling API: {e}")

        time.sleep(SENTIMENT_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()