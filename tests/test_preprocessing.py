import pandas as pd
import pytest

from src.preprocessing.prepare_data import (
    select_and_rename_columns,
    normalize_labels,
    validate_data,
)


def test_select_and_rename_columns():
    df = pd.DataFrame({
        "text": ["a", "b"],
        "sentiment": ["positive", "negative"],
        "extra": [1, 2],
    })
    out = select_and_rename_columns(df)
    assert list(out.columns) == ["text", "label"]
    assert out.loc[0, "label"] == "positive"


def test_normalize_labels_ok():
    df = pd.DataFrame({
        "text": ["a", "b", "c"],
        "label": [" Positive ", "neutral", "NEGATIVE"],
    })
    out = normalize_labels(df)
    assert set(out["label"].unique()) == {"positive", "neutral", "negative"}


def test_normalize_labels_invalid_raises():
    df = pd.DataFrame({
        "text": ["a"],
        "label": ["mixed"],
    })
    with pytest.raises(ValueError):
        normalize_labels(df)


def test_validate_data_ok():
    df = pd.DataFrame({
        "text": ["hello"],
        "label": ["positive"],
    })
    validate_data(df)  # non deve lanciare


def test_validate_data_empty_text_raises():
    df = pd.DataFrame({
        "text": ["   "],
        "label": ["positive"],
    })
    with pytest.raises(ValueError):
        validate_data(df)
