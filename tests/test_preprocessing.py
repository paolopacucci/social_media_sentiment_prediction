import pandas as pd
import pytest

from src.data.data_utils import (
    select_columns,
    clean_text,
    validate_text,
    clean_label_012,
    validate_label_012,
)


def test_select_columns() -> None:
    df = pd.DataFrame({
        "text": ["a", "b"],
        "label": [0, 1],
        "extra": [1, 2],
    })

    out = select_columns(df, ["text", "label"])

    assert list(out.columns) == ["text", "label"]


def test_clean_text() -> None:
    df = pd.DataFrame({
        "text": ["  hello  ", "world   "],
        "label": [0, 1],
    })

    out = clean_text(df, "text")

    assert out.loc[0, "text"] == "hello"
    assert out.loc[1, "text"] == "world"


def test_validate_text_empty_raises() -> None:
    df = pd.DataFrame({
        "text": ["   "],
        "label": [1],
    })

    with pytest.raises(ValueError):
        validate_text(df, "text")


def test_clean_label_012() -> None:
    df = pd.DataFrame({
        "text": ["a", "b", "c"],
        "label": ["0", "1", "2"],
    })

    out = clean_label_012(df, "label")

    assert set(out["label"].unique()) == {0, 1, 2}


def test_validate_label_012_invalid_raises() -> None:
    df = pd.DataFrame({
        "text": ["a", "b"],
        "label": [0, 7],
    })

    with pytest.raises(ValueError):
        validate_label_012(df, "label")