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
    DataCollatorWithPadding,
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

TRAIN_PATH = Path("data/split/train.csv")
VAL_PATH = Path("data/split/val.csv")

# Salva su Drive (come stai già facendo)
OUTPUT_DIR = Path("/content/drive/MyDrive/social_media_reputation/artifacts/model")
REPORTS_DIR = Path("/content/drive/MyDrive/social_media_reputation/reports/current")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["label"].map(LABEL2ID)
    if df["label"].isna().any():
        raise ValueError("Found unmapped labels")
    return df


def to_ds(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label"]])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main():
    max_length = int(os.getenv("MAX_LENGTH", "128"))

    train_df = map_labels(load_csv(TRAIN_PATH))
    val_df = map_labels(load_csv(VAL_PATH))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = to_ds(train_df).map(tok, batched=True)
    val_ds = to_ds(val_df).map(tok, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        use_safetensors=True,
    )
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    # padding dinamico (meglio di padding max_length fisso)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="artifacts/tmp_training",

        # Spingi: batch più grande (effettivo) con grad_accum
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,   # effettivo train batch ~32

        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=5,              # con early stop spesso si ferma prima
        warmup_ratio=0.06,

        # nel tuo transformers 5.0.0 è "eval_strategy" (non evaluation_strategy)
        eval_strategy="epoch",

        # salva SOLO il best model (un solo checkpoint best)
        save_strategy="epoch",
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        fp16=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_info = {
        "base_model": MODEL_NAME,
        "max_length": max_length,
        "output_dir": str(OUTPUT_DIR),
        "train_batch": 16,
        "grad_accum": 2,
        "epochs_max": 5,
        "lr": 2e-5,
    }
    with open(REPORTS_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"OK: saved model to {OUTPUT_DIR}")
    print("Best metrics:", trainer.state.best_metric)


if __name__ == "__main__":
    main()
