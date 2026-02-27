from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset

from sklearn.metrics import (accuracy_score, f1_score)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.config import (MAX_LENGTH)

def load_split(split_dir: str | Path, name: str) -> pd.DataFrame:
    split_dir = Path(split_dir)
    path = split_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}. Run split_data first.")
    return pd.read_csv(path)


def to_hf_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Dataset:
    return Dataset.from_pandas(df[[text_col, label_col]].rename(columns={
        text_col: "text",
        label_col: "label",
    }))


def build_tokenizer(model_base: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_base)


def tokenize_dataset(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")
    return ds


def build_model(
    model_base: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_base,
        num_labels=3,
        use_safetensors=True,
    )


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }