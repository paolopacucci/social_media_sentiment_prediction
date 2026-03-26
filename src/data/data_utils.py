# Funzioni condivise per la data pipeline.

from pathlib import Path
import pandas as pd


# Carica file csv non processato dal path indicato e verifica che esista.
def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

# Salva un DataFrame in CSV creando la cartella di destinazione se non esiste.
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# Seleziona la colonne specificate e verifica che siano tutte presenti.
def select_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Got: {list(df.columns)}")
    return df[cols].copy()

# Pulisce la colonna di testo rimuovendo righe con valori null, converte in stringa, rimuova spazi esterni ed elimina i testi vuoti.
def clean_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)
    return df

# Verifica che il dataset non sia il vuoto e che la colonna testo non contenga valori null o stringhe vuote.
def validate_text(df: pd.DataFrame, text_col: str) -> None:
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[text_col].isnull().any():
        raise ValueError("Null values found in text column")
    if (df[text_col].astype(str).str.strip() == "").any():
        raise ValueError("Empty text values found")

# Converte la colonna label in formato numerico e mantiene solo lerighe con label convertibili in interi.
def clean_label_012(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    labels = pd.to_numeric(df[label_col], errors="coerce")
    df = df.loc[labels.notna()].copy()
    df[label_col] = labels[labels.notna()].astype(int)
    return df.reset_index(drop=True)

# Verifica che tutte le label siano numeriche e che appartengano all'insieme {0, 1. 2}
def validate_label_012(df: pd.DataFrame, label_col: str) -> None:
    lab = pd.to_numeric(df[label_col], errors="coerce")
    if lab.isnull().any():
        raise ValueError("Some labels are not numeric")
    invalid = set(lab.astype(int).unique()) - {0, 1, 2}
    if invalid:
        raise ValueError(f"Invalid label values found: {invalid}")