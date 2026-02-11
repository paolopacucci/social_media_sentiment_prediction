from pathlib import Path
import pandas as pd

raw_data_path = Path("data/raw/sentiment_dataset.csv")
preprocessed_data_path = Path("data/processed/sentiment_preprocessed.csv")

def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def select_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["text", "sentiment"]]
    df = df.rename(columns={"sentiment": "label"})
    return df

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].str.strip().str.lower()

    allowed_labels = {"positive", "neutral", "negative"}
    invalid_labels = set(df["label"].unique()) - allowed_labels
    if invalid_labels:
        raise ValueError(f"Invalid labels found: {invalid_labels}")
    return df

def validate_data(df: pd.DataFrame) -> None:
    expected_columns = {"text", "label"}
    if set(df.columns) != expected_columns:
        raise ValueError(f"Invalid columns, expected {expected_columns}, got {set(df.columns)}")
    
    if df.empty:
        raise ValueError(f"Dataset is empty")

    if df.isnull().any().any():
        raise ValueError("Dataset contains Null values")
    
    if (df["text"].str.strip() == "").any(): 
        raise ValueError("Empty text values found")

def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def main():
    df = load_raw_data(raw_data_path)
    df = select_and_rename_columns(df)
    df = normalize_labels(df)
    validate_data(df)
    save_processed_data(df, preprocessed_data_path)

if __name__ == "__main__":
    main()
