import json 

import torch
from transformers import (Trainer, TrainingArguments)

from src.config import (
    MODEL_ID,
    ARTIFACT_MODEL_DIR_RETRAIN,
    SPLIT_DIR_RETRAIN,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    MAX_STEPS,
    FREEZE_BASE,
    UNFREEZE_LAST_N_LAYERS,
    TEXT_COL, 
    LABEL_COL,
    PUSH_AFTER_RETRAIN,
)
from src.data.split_data import split_retraining_data
from src.training.push_to_hub import push_model_dir
from src.training.train_utils import (
    load_split,
    to_hf_dataset,
    build_tokenizer,
    tokenize_dataset,
    build_model,
    compute_metrics
)


def freeze_model(model) -> None:
    # freeze backbone
    for p in model.base_model.parameters():
        p.requires_grad = False

    # unfreeze last N layers if possible
    try:
        layers = model.base_model.encoder.layer
        n = len(layers)
        k = UNFREEZE_LAST_N_LAYERS
        if k < 0:
            k = 0
        if k > n:
            k = n
        for layer in layers[n - k:]:
            for p in layer.parameters():
                p.requires_grad = True
    except Exception:
        pass

    # classifier always trainable
    try:
        for p in model.classifier.parameters():
            p.requires_grad = True
    except Exception:
        pass


def main()-> None:
    split_retraining_data()
    
    ARTIFACT_MODEL_DIR_RETRAIN.mkdir(parents=True, exist_ok=True)

    train_df = load_split(SPLIT_DIR_RETRAIN, "train")
    val_df = load_split(SPLIT_DIR_RETRAIN, "val")
    test_df = load_split(SPLIT_DIR_RETRAIN, "test")

    tokenizer = build_tokenizer(MODEL_ID)
    model = build_model(MODEL_ID)

    if FREEZE_BASE:
        freeze_model(model)

    train_ds = tokenize_dataset(to_hf_dataset(train_df, TEXT_COL, LABEL_COL), tokenizer)
    val_ds = tokenize_dataset(to_hf_dataset(val_df, TEXT_COL, LABEL_COL), tokenizer)
    test_ds = tokenize_dataset(to_hf_dataset(test_df, TEXT_COL, LABEL_COL), tokenizer)

    # if MAX_STEPS > 0, it overrides epochs and caps training time
    max_steps_value = MAX_STEPS if MAX_STEPS and MAX_STEPS > 0 else -1

    args = TrainingArguments(
        output_dir=str(ARTIFACT_MODEL_DIR_RETRAIN),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps_value,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    metrics = {
        "model_name": MODEL_ID,
        "test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
        "test_macro_f1": float(test_metrics.get("test_macro_f1", 0.0)),
    }

    metrics_path = ARTIFACT_MODEL_DIR_RETRAIN / "retrain_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    trainer.save_model(str(ARTIFACT_MODEL_DIR_RETRAIN))
    tokenizer.save_pretrained(str(ARTIFACT_MODEL_DIR_RETRAIN))

    print("Saved model to:", ARTIFACT_MODEL_DIR_RETRAIN)
    print("Saved metrics to:", metrics_path)
    print("Test metrics:", test_metrics)

    if PUSH_AFTER_RETRAIN:
        push_model_dir(ARTIFACT_MODEL_DIR_RETRAIN, commit_prefix="retrain")

if __name__ == "__main__":
    main()