from pathlib import Path
import pandas as pd

from src.preprocessing.prepare_data import select_and_rename_columns, normalize_labels, validate_data
from src.preprocessing.split_data import split_data

def test_prepare_and_split_integration(tmp_path: Path):
    # 1) creo "raw" finto
    base_texts = ["a", "b", "c", "d", "e", "f"]
    base_labels = ["positive", "neutral", "negative", "positive", "neutral", "negative"]

    raw = pd.DataFrame({
        "text": base_texts * 5,
        "sentiment": base_labels * 5,
    })

    # 2) preparo come fa prepare_data.py
    df = select_and_rename_columns(raw)
    df = normalize_labels(df)
    validate_data(df)

    # 3) split come fa split_data.py
    train_df, val_df, test_df = split_data(df)

    # 4) controlli minimi
    assert set(train_df.columns) == {"text", "label"}
    assert set(val_df.columns) == {"text", "label"}
    assert set(test_df.columns) == {"text", "label"}
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
