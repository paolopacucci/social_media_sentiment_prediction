# src/data/split_data.py
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    SENTIMENT_PREPROCESSED_PATH,
    SPLIT_DIR,
    PERFORMANCE_MONITORING_PATH,
    SPLIT_DIR_RETRAIN,
    LABEL_COL,
)


def split_dataframe(
    df: pd.DataFrame,
    label_col: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df[label_col],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df[label_col],
    )

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path} ({len(train_df)})")
    print(f"Saved: {val_path} ({len(val_df)})")
    print(f"Saved: {test_path} ({len(test_df)})")

    return out_dir


def split_training_data() -> Path:
    df = pd.read_csv(SENTIMENT_PREPROCESSED_PATH)
    train_df, val_df, test_df = split_dataframe(df, LABEL_COL)
    return save_splits(train_df, val_df, test_df, SPLIT_DIR)


def split_retraining_data() -> Path:
    df = pd.read_csv(PERFORMANCE_MONITORING_PATH)
    train_df, val_df, test_df = split_dataframe(df, LABEL_COL)
    return save_splits(train_df, val_df, test_df, SPLIT_DIR_RETRAIN)