# Servizio di monitoring delle performance del modello.
# Campiona batch etichettati da un dataset dedicato, chiama l'endpoint/predict, 
# calcola accuracy e macro-F1 ed espone queste metriche a Prometheus.
# Può attivare il retraining se la macro-F1 supera la soglia impostata per un numero n volte consecutivo.

import random
import subprocess
import time

import pandas as pd

import requests
from prometheus_client import Counter, Gauge, start_http_server
from sklearn.metrics import accuracy_score, f1_score

from src.data.prepare_data import prepare_performance_monitoring_data
from src.config import (
    API_URL,
    PERFORMANCE_MONITORING_PATH,
    TEXT_COL,
    LABEL_COL,
    PERF_INTERVAL_SECONDS,
    PERF_BATCH_SIZE,
    PERF_METRICS_PORT,
    WINDOW_CONSECUTIVE,
    F1_THRESHOLD,
    TRIGGER_RETRAIN,
    RUN_RETRAIN,
)

# Metriche del monitoring.
# - macro-F1 sull'ultimo batch valutato
# - accuracy sull'ultimo batch valutato
# - numero totale di trigger di retraining avvenuti
F1_MACRO_WINDOW = Gauge("f1_macro_window", "Macro F1 on performance monitoring batch")
ACCURACY_WINDOW = Gauge("accuracy_window", "Accuracy on performance monitoring batch")
RETRAIN_TRIGGERS = Counter("retrain_triggers_total", "How many retrain triggers happened")

# Carica il dataset preprocessato e verifica che contenga sia il testo che la label reale.
def load_processed_data() -> pd.DataFrame:
    if not PERFORMANCE_MONITORING_PATH.exists():
        raise FileNotFoundError(f"Missing processed performance file: {PERFORMANCE_MONITORING_PATH}")
    df = pd.read_csv(PERFORMANCE_MONITORING_PATH)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Missing required columns: {TEXT_COL}, {LABEL_COL}")
    return df


# Invia una batch di testi all'API e restituisce gli id numerici delle classi predette dal modello.
def call_predict_batch(texts: list[str]) -> list[int]:
    r = requests.post(f"{API_URL}/predict", json={"texts": texts}, timeout=60)
    r.raise_for_status()
    outs = r.json()["outputs"]
    return [int(o["label_id"]) for o in outs]

# Avvia il retraining come processo separato.
def run_retrain() -> None:
    subprocess.check_call(["python", "-m", "src.training.train_retrain"])

# Funzione principale del servizio di monitoring.
def main() -> None:
    # All'avvio stampa tutte le informazioni e impostazioni del servizio. 
    print(f"[performance_exporter] API_URL={API_URL}")
    print(f"[performance_exporter] DATA_PATH={PERFORMANCE_MONITORING_PATH}")
    print(f"[performance_exporter] batch_size={PERF_BATCH_SIZE} every {PERF_INTERVAL_SECONDS}s")
    print(f"[performance_exporter] metrics on :{PERF_METRICS_PORT}/metrics")
    print(
        f"[performance_exporter] retrain: threshold={F1_THRESHOLD} "
        f"consecutive={WINDOW_CONSECUTIVE} trigger_enabled={TRIGGER_RETRAIN} "
        f"run_retrain={RUN_RETRAIN}"
    )

    
    # Avvia endpoint da cui Prometheus leggerà le metriche.
    start_http_server(PERF_METRICS_PORT)

    # Crea il file prepocessato se non esiste.
    if not PERFORMANCE_MONITORING_PATH.exists():
        print("[performance_exporter] processed file missing -> preparing it now")
        prepare_performance_monitoring_data()

    # Carica il dataset e mantiene in memoria in testi e label in due liste separate. 
    df = load_processed_data()
    texts_all = df[TEXT_COL].tolist()
    y_true_all = df[LABEL_COL].tolist()

    # Conta quante finestre consecutive vanno sotto soglia.
    consecutive_below = 0
 
    while True:

        # Seleziona casualmente gli inidici della batch corrente, questi vengono usati sia per i testi che le label reali,
        # così da mantenere allineati testi e relative label.
        idx = random.sample(range(len(texts_all)), k=min(PERF_BATCH_SIZE, len(texts_all)))
        batch_texts = [texts_all[i] for i in idx]
        y_true = [y_true_all[i] for i in idx]

        # Esegue inference tramite API e raccoglie le predizioni delle classi fornite dal modello. 
        try:
            y_pred = call_predict_batch(batch_texts)
        # Se la chiamata API fallisce il servizio non s'interrompe ma attende l'intervallo configurato e riprova al ciclo successivo.    
        except Exception as e:
            print(f"[performance_exporter] error calling API: {e}")
            time.sleep(PERF_INTERVAL_SECONDS)
            continue

        # Confronta le predizioni e le label reali della batch corrente per calcolare accuracy e macro-F1.        
        acc = float(accuracy_score(y_true, y_pred))
        mf1 = float(f1_score(y_true, y_pred, average="macro"))

        ACCURACY_WINDOW.set(acc)
        F1_MACRO_WINDOW.set(mf1)

        # Tiene traccia di quante batch consecutive restano sotto soglia e aggiorna la variabile.
        if mf1 <= F1_THRESHOLD:
            consecutive_below += 1
        else:
            consecutive_below = 0

        print(
            f"[performance_exporter] acc={acc:.3f} f1_macro={mf1:.3f} "
            f"(below={consecutive_below}/{WINDOW_CONSECUTIVE}, threshold={F1_THRESHOLD})"
        )

        # Se la soglia viene violata per n volte consecutive, registra il triggerr e, se configurato, lancia il retraining.
        if TRIGGER_RETRAIN and consecutive_below >= WINDOW_CONSECUTIVE:
            RETRAIN_TRIGGERS.inc()
            consecutive_below = 0

            if RUN_RETRAIN:
                print("[performance_exporter] retrain trigger condition met -> launching train_retrain")
                try:
                    run_retrain()
                    print("[performance_exporter] retrain complete")
                except Exception as e:
                    print(f"[performance_exporter] retrain failed: {e}")
            else:
                print("[performance_exporter] retrain trigger condition met -> retrain skipped (RUN_RETRAIN=False)")

        time.sleep(PERF_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()