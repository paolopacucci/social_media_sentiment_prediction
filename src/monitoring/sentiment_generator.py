# Crea un flusso di batch di testi usando un dataset dedicato, 
# lo invia all'API e aggiorna le metriche Prometheus.

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

# Metriche del monitoring.
# - conteggi per label nell'ultimo batch
# - dimensione dell'ultimo batch
# - timestamp dell'ultimo batch elaborato
# - numero totale di batch generat
LAST_BATCH_COUNT = Gauge("sentiment_last_batch_count", "Counts in last batch", ["label"])
LAST_BATCH_SIZE = Gauge("sentiment_last_batch_size", "Size of last batch")
LAST_BATCH_TS = Gauge("sentiment_last_batch_timestamp", "Unix ts of last batch")
BATCHES_TOTAL = Counter("sentiment_batches_total", "Total generated sentiment batches")


# Carica il dataset preprocessato per simulare il flusso live e veriffica che sia presente solo la colonna text.
def load_processed_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed sentiment monitoring file: {path}")
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing required column: {TEXT_COL}")
    return df

 
# Estrare una batch casuale di testi dal dataset, simulando un input live.
def sample_batch(texts_all: list[str], batch_size: int) -> list[str]:
    k = min(batch_size, len(texts_all))
    return random.sample(texts_all, k=k)


# Invia la batch all'API di serving e restituisce il JSON completo.
def call_predict_batch(api_url: str, batch_texts: list[str]) -> dict:
    r = requests.post(f"{api_url}/predict", json={"texts": batch_texts}, timeout=30)
    r.raise_for_status()
    return r.json()

# Azzera i conteggi dell'ultimo batch prima di scrivere i nuovi valori.
def reset_last_batch_gauges() -> None:
    for lab in LABELS:
        LAST_BATCH_COUNT.labels(label=lab).set(0)


# Avvia il servizio di sentiment monitoring.
# Prepara il dataset se manca, espone metriche Prometheus. 
def main() -> None:
    print(f"[sentiment_generator] API_URL={API_URL}")
    print(f"[sentiment_generator] DATA_PATH={SENTIMENT_MONITORING_TEXT_PATH}")
    print(f"[sentiment_generator] batch_size={SENTIMENT_BATCH_SIZE} every {SENTIMENT_INTERVAL_SECONDS}s")
    print(f"[sentiment_generator] metrics on :{SENTIMENT_METRICS_PORT}/metrics")

    # Avvia endpoint da cui Prometheus leggerà le metriche.
    start_http_server(SENTIMENT_METRICS_PORT)

    # Se il dataset non esiste lo crea.
    if not SENTIMENT_MONITORING_TEXT_PATH.exists():
        print("[sentiment_generator] processed file missing -> preparing it now")
        prepare_sentiment_monitoring_data()

    # Carica il dataset preprocessato, converte i testi in stringhe ed esegue ulteriore pulizia dei testi. 
    df = load_processed_data(SENTIMENT_MONITORING_TEXT_PATH)
    texts_all = df[TEXT_COL].astype(str).str.strip()
    texts_all = texts_all[texts_all != ""].tolist()

    if not texts_all:
        raise ValueError("No valid texts found")

    while True:
        # Campiona una batch di testi dal dataset ripulito per simulare sentiment live.
        batch_texts = sample_batch(texts_all, SENTIMENT_BATCH_SIZE)

        # Resetta i valori delle gauge dell'ultima batch prima di aggiornarli con i nuovi risultati.
        reset_last_batch_gauges()

        try:
            # Invia la batch all'API e recupera il payload JSON della risposta.
            payload = call_predict_batch(API_URL, batch_texts)
            
            # Estrae dalla risposta JSON la lista delle predizioni della batch.
            outs = payload.get("outputs", [])

            # Conta quante predizioni della batch corrente appartengono a ciascuna label.
            counts = {lab: 0 for lab in LABELS}
            for o in outs:
                lab = o.get("label")
                if lab in counts:
                    counts[lab] += 1

            for lab, c in counts.items():
                LAST_BATCH_COUNT.labels(label=lab).set(int(c))

            # Aggiorna le metriche dell'ultima batch osservata e incrementa il numero totale di batch osservate.
            LAST_BATCH_SIZE.set(int(len(outs)))
            LAST_BATCH_TS.set(int(time.time()))
            BATCHES_TOTAL.inc()

            print(f"[sentiment_generator] last_batch n={len(outs)} counts={counts}")

        except Exception as e:
            print(f"[sentiment_generator] error calling API: {e}")

        time.sleep(SENTIMENT_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()