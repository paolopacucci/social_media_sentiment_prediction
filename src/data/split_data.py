# Questo file contiene le funzioni neccesarie per eseguire gli split dei dataset di
# training iniziale e retraining.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    SENTIMENT_PREPROCESSED_PATH,
    SPLIT_DIR,
    PERFORMANCE_MONITORING_PATH,
    SPLIT_DIR_RETRAIN,
    TEST_DIR_SPLIT,
    LABEL_COL,
)


# Esegue split train/val/test per il training iniziale
def split_train_val_test_dataframe(
    df: pd.DataFrame,
    label_col: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Con il primo split si ottiene lo split train e uno split temporaneo.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df[label_col],
    )

    # Lo split temporaneo viene diviso per ottenere gli split val/test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df[label_col],
    )

    return train_df, val_df, test_df


# Esegue split train/val per il file di retraining.
def split_train_val_dataframe(
    df: pd.DataFrame,
    label_col: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df[label_col],
    )
    return train_df, val_df


# Salva gli split train/val nella cartella indicata.
def save_train_val_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Saved: {train_path} ({len(train_df)})")
    print(f"Saved: {val_path} ({len(val_df)})")

    return out_dir


# Salva lo split test condiviso da train e retrain.
def save_shared_test(
    test_df: pd.DataFrame,
    out_dir: str | Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_path = out_dir / "test.csv"
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {test_path} ({len(test_df)})")

    return out_dir


# Esegue lo split per il training iniziale, salva train/val nella cartella impostata e
# test separatamente da usare anche con il retraining come benchmark condiviso. 
def split_training_data() -> Path:
    df = pd.read_csv(SENTIMENT_PREPROCESSED_PATH)
    train_df, val_df, test_df = split_train_val_test_dataframe(df, LABEL_COL)

    save_train_val_splits(train_df, val_df, SPLIT_DIR)
    save_shared_test(test_df, TEST_DIR_SPLIT)

    return SPLIT_DIR

# Esegue lo split train/val per il file di retraining.
def split_retraining_data() -> Path:
    df = pd.read_csv(PERFORMANCE_MONITORING_PATH)
    train_df, val_df = split_train_val_dataframe(df, LABEL_COL)
    return save_train_val_splits(train_df, val_df, SPLIT_DIR_RETRAIN)