# Training iniziale del modello effettuato con GPU su Colab.
# Il file prepara i dati, costrusice tokenizer e modello, esegue fine-tuning, valuta modello sul test codniviso con il retraining,
# salva metriche e artifact nel runtime e, se configurato, pubblica il modello su Hugging Face.

# Import librerie, variabili dal file config  e funzioni da utilities
from pathlib import Path
import json

import torch
from transformers import (TrainingArguments, Trainer, EarlyStoppingCallback,)

from src.config import (
    MODEL_BASE,
    SPLIT_DIR,
    TEST_DIR_SPLIT,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    MODEL_ID,
    TEXT_COL, 
    LABEL_COL,
    ARTIFACT_MODEL_DIR_TRAIN,
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
# Preprocessing dei dati, creazione e load degli split train/val/test per il training iniziale.
# Il test viene salvato sepratamente come split condviso per la valutazione del retraining.
    prepare_training_data()
    split_training_data()

    train_df = load_split(SPLIT_DIR, "train")
    val_df = load_split(SPLIT_DIR, "val")
    test_df = load_split(TEST_DIR_SPLIT, "test")

# Load del tokenizer RoBERTa
    tokenizer = build_tokenizer(MODEL_BASE)

# Conversione dei DataFrame in dataset Hugging Face compatibili con Trainer e tokenizzazione dei testi
    train_ds = tokenize_dataset(to_hf_dataset(train_df, TEXT_COL, LABEL_COL), tokenizer)
    val_ds = tokenize_dataset(to_hf_dataset(val_df, TEXT_COL, LABEL_COL), tokenizer)
    test_ds = tokenize_dataset(to_hf_dataset(test_df, TEXT_COL, LABEL_COL), tokenizer)

# Load del modello RoBERTa
    model = build_model(MODEL_BASE)


# Configurazione dei Training Arguments per il fine-tuning:
# Vengono definiti directory di salvataggio, batch per train/eval, learnign rate, numero di epoche,
# strategie di valutazione e salvataggio, logging e uso opzionale di fp16 su GPU.
# Valutazione e salvataggio avvengono ad ogni epoca, alla fine viene ricaricato il checkpoint migliore
# secondo la metrica macro_f1 sul validation test.
    args = TrainingArguments(
        output_dir=ARTIFACT_MODEL_DIR_TRAIN,
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

# Configurazione del Trainer.
# Il Trainer gestisce il ciclo di training e valutazione, ricevendo il modello, TrainingArguments,
# i dataset, la funzione per le metriche ed EarlyStopiingCallback.
# L'early stopping interrompe il training in anticipo se macro_f1 non migliora per il numero di volte consecutive definitite dalla patience,
# questo aiuta a limitare epoche inutili e a ridurre il rischio di over-fitting.
    trainer = Trainer(
        model=model,                
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

# Training.
    trainer.train()

# Valutazione modello con lo split "test" condiviso, e salvataggio metriche in un dizionario.
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    metrics = {
        "model_name": MODEL_ID,
        "test_accuracy": float(test_metrics.get("test_accuracy", 0.0)),
        "test_macro_f1": float(test_metrics.get("test_macro_f1", 0.0)),
    }

# La directory degli artifcats è locale al runtime che esegue il training(in questo caso Colab).
    ARTIFACT_MODEL_DIR_TRAIN.mkdir(parents=True, exist_ok=True)

# Salvataggio report.
    reports_dir = Path("reports/train")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "final_metrics.json"
   
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

# Salvataggio modello e tokenizer.
    trainer.save_model(str(ARTIFACT_MODEL_DIR_TRAIN))
    tokenizer.save_pretrained(str(ARTIFACT_MODEL_DIR_TRAIN))

    print(f"Saved model to: {ARTIFACT_MODEL_DIR_TRAIN}")
    print(f"Saved final metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

# Il Push del modello dal runtime corrente su Hugging Face è opzionale. 
# Se PUSH_AFTER_TRAIN=False, il modello resta dipsonibile nel filesystem del runtime corrente.
    if PUSH_AFTER_TRAIN:
        push_model_dir(ARTIFACT_MODEL_DIR_TRAIN, commit_prefix="initial-train")

if __name__ == "__main__":
    main()
