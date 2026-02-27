from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "request_count",
    "Number of HTTP requests",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

PREDICTED_LABEL_TOTAL = Counter(
    "predicted_label_total",
    "Total predicted labels",
    ["label"],
)

MODEL_INFERENCE_SECONDS = Histogram(
    "model_inference_seconds",
    "Model inference time in seconds",
)

PREDICTION_CONFIDENCE = Gauge(
    "prediction_confidence",
    "Average confidence of last prediction batch",
)