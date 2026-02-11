from pathlib import Path
import pandas as pd

from src.preprocessing.prepare_data import (
    load_raw_data,
    select_and_rename_columns,
    normalize_labels,
    validate_data,
    save_processed_data,
)

RAW_PATH = Path("data/raw/performance_monitoring_batch.csv")
OUT_PATH = Path("data/processed/performance_monitoring_batch.csv")


def main():
    df = load_raw_data(RAW_PATH)
    df = select_and_rename_columns(df)   # -> text,label
    df = normalize_labels(df)
    validate_data(df)
    save_processed_data(df, OUT_PATH)
    print(f"OK saved {OUT_PATH}")


if __name__ == "__main__":
    main()
