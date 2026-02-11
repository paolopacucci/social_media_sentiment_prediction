from pathlib import Path
import os
import json

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

TRAIN_PATH = Path("data/split/train.csv")
VAL_PATH = Path("data/split/val.csv")
OUTPUT_DIR = Path("/content/drive/MyDrive/social_media_reputation/artifacts/model")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "64"))


def load_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError("Found unmapped labels in dataset")
    return df


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]])


def build_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
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


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        use_safetensors=True,
    )
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main():
    train_df = map_labels(load_split_csv(TRAIN_PATH))
    val_df = map_labels(load_split_csv(VAL_PATH))

    tokenizer = build_tokenizer()

    train_ds = tokenize_dataset(to_hf_dataset(train_df), tokenizer)
    val_ds = tokenize_dataset(to_hf_dataset(val_df), tokenizer)

    model = build_model()

    # Parametri "più seri" ma non NASA: pensati per Colab T4
    args = TrainingArguments(
        output_dir="artifacts/tmp_training",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.06,
        evaluation_strategy="epoch",
        save_strategy="no",          # niente checkpoint pesanti
        logging_steps=50,
        fp16=True,                   # mixed precision su GPU NVIDIA
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    save_run_info()
    print(f"OK: saved final model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
