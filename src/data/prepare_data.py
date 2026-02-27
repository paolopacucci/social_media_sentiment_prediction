from pathlib import Path

from src.config import (
    RAW_SENTIMENT_DATASET,
    SENTIMENT_PREPROCESSED_PATH, 
    RAW_PERFORMANCE_MONITORING_BATCH,
    PERFORMANCE_MONITORING_PATH,
    RAW_SENTIMENT_MONITORING_BATCH, 
    SENTIMENT_MONITORING_TEXT_PATH,
    TEXT_COL,
    LABEL_COL,
)

from src.data.data_utils import (
    load_raw_data,
    select_columns,
    clean_text,  
    clean_label_012,  
    validate_text,      
    validate_label_012,      
    save_csv,
)


def prepare_training_data() -> Path:
    df = load_raw_data(RAW_SENTIMENT_DATASET)
    df = select_columns(df, [TEXT_COL, LABEL_COL])
    df = clean_text(df, TEXT_COL)
    df = clean_label_012(df, LABEL_COL)

    validate_text(df, TEXT_COL)
    validate_label_012(df, LABEL_COL)

    save_csv(df, SENTIMENT_PREPROCESSED_PATH)
    print(f"Saved: {SENTIMENT_PREPROCESSED_PATH} ({len(df)})")
    return SENTIMENT_PREPROCESSED_PATH


def prepare_performance_monitoring_data() -> Path:
    df = load_raw_data(RAW_PERFORMANCE_MONITORING_BATCH)
    df = select_columns(df, [TEXT_COL, LABEL_COL])
    df = clean_text(df, TEXT_COL)
    df = clean_label_012(df, LABEL_COL)

    validate_text(df, TEXT_COL)
    validate_label_012(df, LABEL_COL)

    save_csv(df, PERFORMANCE_MONITORING_PATH)
    print(f"Saved: {PERFORMANCE_MONITORING_PATH} ({len(df)})")
    return PERFORMANCE_MONITORING_PATH


def prepare_sentiment_monitoring_data() -> Path:
    df = load_raw_data(RAW_SENTIMENT_MONITORING_BATCH)
    df = select_columns(df, [TEXT_COL])
    df = clean_text(df, TEXT_COL)
    validate_text(df, TEXT_COL)

    save_csv(df, SENTIMENT_MONITORING_TEXT_PATH)
    print(f"Saved: {SENTIMENT_MONITORING_TEXT_PATH} ({len(df)})")
    return SENTIMENT_MONITORING_TEXT_PATH