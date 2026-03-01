import os
from pathlib import Path

# Models
MODEL_BASE = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_ID = "paolopacucci/sentiment-roberta"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Columns
TEXT_COL = "text"
LABEL_COL = "label"

# Raw datasets
RAW_SENTIMENT_DATASET = Path("data/raw/sentiment_dataset_raw.csv")
RAW_SENTIMENT_MONITORING_BATCH = Path("data/raw/sentiment_monitoring_batch_raw.csv")
RAW_PERFORMANCE_MONITORING_BATCH = Path("data/raw/performance_monitoring_batch_raw.csv")

# Output folders
PREPROCESSED_DIR = Path("data/preprocessed")
SPLIT_DIR = Path("data/split/train")
SPLIT_DIR_RETRAIN = Path("data/split/retrain")

ARTIFACT_MODEL_DIR = Path("/content/drive/MyDrive/social_media_reputation/artifacts/model")
ARTIFACT_MODEL_DIR_RETRAIN = Path("artifacts/retrain_model")

# Derived paths
SENTIMENT_PREPROCESSED_PATH = PREPROCESSED_DIR / "sentiment_preprocessed.csv"
SENTIMENT_MONITORING_TEXT_PATH = PREPROCESSED_DIR / "sentiment_monitoring_preprocessed.csv"
PERFORMANCE_MONITORING_PATH = PREPROCESSED_DIR / "performance_monitoring_preprocessed.csv"

# Training
MAX_LENGTH = 96
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
FREEZE_BASE = True
UNFREEZE_LAST_N_LAYERS = 0
MAX_STEPS = 120

PUSH_AFTER_TRAIN = True
PUSH_AFTER_RETRAIN = False

# API
API_PORT = 8000
API_URL = f"http://api:{API_PORT}"

# Sentiment generator
SENTIMENT_INTERVAL_SECONDS = 15
SENTIMENT_BATCH_SIZE = 50
SENTIMENT_METRICS_PORT = 8002

# Performance exporter
PERF_INTERVAL_SECONDS = 15
PERF_BATCH_SIZE = 100
PERF_METRICS_PORT = 8001
WINDOW_CONSECUTIVE = 1
F1_THRESHOLD = 0.85
TRIGGER_RETRAIN = True
RUN_RETRAIN = False