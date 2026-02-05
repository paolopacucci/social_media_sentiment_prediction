from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

TRAIN_PATH = Path("data/split/train.csv")
VAL_PATH = Path("data/split/val.csv")
OUTPUT_DIR = Path("artifacts/experiments/model_150_max_steps")

# Ordine ufficiale del modello:
# 0 = negative, 1 = neutral, 2 = positive
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # converti label testuale -> numerica (0/1/2)
    train_df["label"] = train_df["label"].map(LABEL2ID)
    val_df["label"] = val_df["label"].map(LABEL2ID)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, use_safetensors=True)
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=150,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
