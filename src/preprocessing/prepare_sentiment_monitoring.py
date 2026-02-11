from pathlib import Path
import pandas as pd

from src.preprocessing.prepare_data import load_raw_data, save_processed_data

raw_data_path = Path("data/raw/sentiment_monitoring_batch.csv")
preprocessed_data_path = Path("data/processed/sentiment_monitoring_batch_text.csv")


def select_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        raise ValueError("Missing 'text' column")
    return df[["text"]]


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.strip()
    return df


def validate_text_data(df: pd.DataFrame) -> None:
    expected_columns = {"text"}
    if set(df.columns) != expected_columns:
        raise ValueError(f"Invalid columns, expected {expected_columns}, got {set(df.columns)}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isnull().any().any():
        raise ValueError("Dataset contains Null values")

    if (df["text"] == "").any():
        raise ValueError("Empty text values found")


def main():
    df = load_raw_data(raw_data_path)
    df = select_text_column(df)
    df = normalize_text(df)
    validate_text_data(df)
    save_processed_data(df, preprocessed_data_path)
    print(f"OK saved {preprocessed_data_path}")


if __name__ == "__main__":
    main()
