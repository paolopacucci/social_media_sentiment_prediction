from pathlib import Path
import os
import json
import random

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed,
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

TRAIN_PATH = Path("data/split/train.csv")
VAL_PATH   = Path("data/split/val.csv")

# salva su Drive (come fai già)
OUTPUT_DIR = Path("/content/drive/MyDrive/social_media_reputation/artifacts/model")
REPORTS_DIR = Path("/content/drive/MyDrive/social_media_reputation/reports/current")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

SEED = int(os.getenv("SEED", "42"))

def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def train_one(run_cfg: dict) -> dict:
    """
    run_cfg keys:
      - run_name
      - max_length
      - lr
      - epochs
      - train_bs
      - grad_accum
      - weight_decay
      - label_smoothing
      - warmup_ratio
      - scheduler
    """
    seed_everything(SEED)

    train_df = map_labels(load_csv(TRAIN_PATH))
    val_df   = map_labels(load_csv(VAL_PATH))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    max_length = int(run_cfg["max_length"])

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = to_ds(train_df).map(tok, batched=True)
    val_ds   = to_ds(val_df).map(tok, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds   = val_ds.remove_columns(["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        use_safetensors=True,
    )
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    run_name = run_cfg["run_name"]
    out_dir = Path("artifacts/tmp_training") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # transformers 5.0.0: usa eval_strategy (non evaluation_strategy)
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(run_cfg["train_bs"]),
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=int(run_cfg["grad_accum"]),

        learning_rate=float(run_cfg["lr"]),
        num_train_epochs=int(run_cfg["epochs"]),
        weight_decay=float(run_cfg["weight_decay"]),

        warmup_ratio=float(run_cfg["warmup_ratio"]),
        lr_scheduler_type=str(run_cfg["scheduler"]),

        # “NASA”: un po’ di regolarizzazione utile
        label_smoothing_factor=float(run_cfg["label_smoothing"]),
        max_grad_norm=1.0,

        eval_strategy="epoch",
        save_strategy="epoch",

        # salva solo 1 checkpoint (best)
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        fp16=True,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    best_metric = trainer.state.best_metric
    best_ckpt = trainer.state.best_model_checkpoint

    return {
        "run_name": run_name,
        "best_macro_f1": float(best_metric) if best_metric is not None else None,
        "best_checkpoint": best_ckpt,
        "cfg": run_cfg,
    }

def main():
    # 4 tentativi “mirati” (non a caso)
    trials = [
        # spesso 1e-5 è migliore del 2e-5 su finetuning sentiment
        {"run_name":"t1_lr1e-5_len128", "max_length":128, "lr":1e-5, "epochs":6, "train_bs":16, "grad_accum":2,
         "weight_decay":0.01, "label_smoothing":0.0, "warmup_ratio":0.06, "scheduler":"cosine"},

        {"run_name":"t2_lr1e-5_len192", "max_length":192, "lr":1e-5, "epochs":6, "train_bs":16, "grad_accum":2,
         "weight_decay":0.01, "label_smoothing":0.0, "warmup_ratio":0.06, "scheduler":"cosine"},

        # un po’ di smoothing può aiutare la generalizzazione
        {"run_name":"t3_lr2e-5_len128_ls", "max_length":128, "lr":2e-5, "epochs":5, "train_bs":16, "grad_accum":2,
         "weight_decay":0.01, "label_smoothing":0.05, "warmup_ratio":0.06, "scheduler":"cosine"},

        # scheduler linear spesso è più stabile
        {"run_name":"t4_lr2e-5_len128_lin", "max_length":128, "lr":2e-5, "epochs":5, "train_bs":16, "grad_accum":2,
         "weight_decay":0.01, "label_smoothing":0.0, "warmup_ratio":0.06, "scheduler":"linear"},
    ]

    results = []
    for cfg in trials:
        print("\n==============================")
        print("RUN:", cfg["run_name"], cfg)
        res = train_one(cfg)
        print("DONE:", res)
        results.append(res)

    # scegli best su val macro_f1
    results = [r for r in results if r["best_macro_f1"] is not None]
    best = max(results, key=lambda r: r["best_macro_f1"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "colab_trials.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(REPORTS_DIR / "colab_best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    # carica best checkpoint e salva in OUTPUT_DIR (finale)
    print("\nBEST:", best["run_name"], "macro_f1:", best["best_macro_f1"])
    print("Checkpoint:", best["best_checkpoint"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(best["best_checkpoint"], use_safetensors=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # info finale
    run_info = {
        "base_model": MODEL_NAME,
        "seed": SEED,
        "saved_to": str(OUTPUT_DIR),
        "best_run": best,
    }
    with open(REPORTS_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nOK: BEST model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
