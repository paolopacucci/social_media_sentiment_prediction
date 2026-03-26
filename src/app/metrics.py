# Metriche Prometheus usate dal servizio API.
# Sono definite in un modulo separato per mantenere più pulito main.py
# e centralizzare in un solo punto la configurazione delle metriche

from prometheus_client import Counter, Gauge, Histogram

# Conta il numero totale di richieste ricevute dall'API, distinguendole per endpoint
REQUEST_COUNT = Counter(
    "request_count",
    "Number of HTTP requests",
    ["endpoint"],
)

# Misura la latenza delle richieste HTTP per endpoint.
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

# Conta il totale di tutte le label predette.
PREDICTED_LABEL_TOTAL = Counter(
    "predicted_label_total",
    "Total predicted labels",
    ["label"],
)

# Misura il tempo di inferenza del modello.
MODEL_INFERENCE_SECONDS = Histogram(
    "model_inference_seconds",
    "Model inference time in seconds",
)

# Tiene traccia della confidence media dell'ultima batch predetta. 
PREDICTION_CONFIDENCE = Gauge(
    "prediction_confidence",
    "Average confidence of last prediction batch",
)