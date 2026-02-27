import pandas as pd

from src.data.data_utils import (
    select_columns,
    clean_text,
    validate_text,
    clean_label_012,
    validate_label_012,
)
from src.data.split_data import split_dataframe

def test_prepare_and_split_integration() -> None:
    raw = pd.DataFrame({
        "text": ["a", "b", "c", "d", "e", "f"] * 5,
        "label": [0, 1, 2, 0, 1, 2] * 5,
        "junk": [123] * 30,
    })

    df = select_columns(raw, ["text", "label"])
    df = clean_text(df, "text")
    df = clean_label_012(df, "label")
    validate_text(df, "text")
    validate_label_012(df, "label")

    train_df, val_df, test_df = split_dataframe(df, "label")

    assert set(train_df.columns) == {"text", "label"}
    assert set(val_df.columns) == {"text", "label"}
    assert set(test_df.columns) == {"text", "label"}
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0