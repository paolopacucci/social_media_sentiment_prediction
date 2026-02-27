from pathlib import Path
import pandas as pd

TEXT_COL = "text"
LABEL_COL = "sentiment"

def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def select_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Got: {list(df.columns)}")
    return df[cols].copy()

def clean_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    return df

def validate_text(df: pd.DataFrame, text_col: str) -> None:
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[text_col].isnull().any():
        raise ValueError("Null values found in text column")
    if (df[text_col].astype(str).str.strip() == "").any():
        raise ValueError("Empty text values found")

def clean_label_012(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype(int)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    return df

def validate_label_012(df: pd.DataFrame, label_col: str) -> None:
    lab = pd.to_numeric(df[label_col], errors="coerce")
    if lab.isnull().any():
        raise ValueError("Some labels are not numeric")
    invalid = set(lab.astype(int).unique()) - {0, 1, 2}
    if invalid:
        raise ValueError(f"Invalid label values found: {invalid}")