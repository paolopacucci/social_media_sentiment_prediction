from pathlib import Path
import json

import torch
from transformers import (TrainingArguments, Trainer, EarlyStoppingCallback,)

from src.config import (
    MODEL_BASE,
    SPLIT_DIR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MODEL_ID,
    TEXT_COL, 
    LABEL_COL,
    ARTIFACT_MODEL_DIR,
    PUSH_AFTER_TRAIN, 
)
from src.data.prepare_data import prepare_training_data
from src.data.split_data import split_training_data
from src.training.push_to_hub import push_model_dir
from src.training.train_utils import (
    load_split,
    to_hf_dataset,
    build_tokenizer,
    tokenize_dataset,
    build_model,
    compute_metrics
)


def main() -> None:

    prepare_training_data()
    split_training_data()

    train_df = load_split(SPLIT_DIR, "train")
    val_df = load_split(SPLIT_DIR, "val")
    test_df = load_split(SPLIT_DIR, "test")

    tokenizer = build_tokenizer(MODEL_BASE)

    train_ds = tokenize_dataset(to_hf_dataset(train_df, TEXT_COL, LABEL_COL), tokenizer)
    val_ds = tokenize_dataset(to_hf_dataset(val_df, TEXT_COL, LABEL_COL), tokenizer)
    test_ds = tokenize_dataset(to_hf_dataset(test_df, TEXT_COL, LABEL_COL), tokenizer)

    model = build_model(MODEL_BASE)


    args = TrainingArguments(
        output_dir=ARTIFACT_MODEL_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",   
        save_total_limit=1,     
        logging_steps=50,
        fp16=torch.cuda.is_available(),                  
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
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    metrics = {
        "model_name": MODEL_ID,
        "test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
        "test_macro_f1": float(test_metrics.get("test_macro_f1", 0.0)),
    }

    ARTIFACT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = ARTIFACT_MODEL_DIR / "final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    trainer.save_model(str(ARTIFACT_MODEL_DIR))
    tokenizer.save_pretrained(str(ARTIFACT_MODEL_DIR))

    print(f"Saved model to: {ARTIFACT_MODEL_DIR}")
    print(f"Saved final metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if PUSH_AFTER_TRAIN:
        push_model_dir(ARTIFACT_MODEL_DIR, commit_prefix="initial-train")

if __name__ == "__main__":
    main()
