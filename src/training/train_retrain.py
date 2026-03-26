# Retraining "light" condizionale del modello effettuato all'interno del Docker.
# Il retrain viene triggerato dal file performace_exporter data da una condizione di superamento soglia della macro_f1.
# È stato pensato per effettuare un fine-tuning leggero adatto ad ambienti con CPU. 

# Import librerie, variabili dal file config  e funzioni da utilities
import json 
from pathlib import Path

import torch
from transformers import (Trainer, TrainingArguments)

from src.config import (
    MODEL_ID,
    ARTIFACT_MODEL_DIR_RETRAIN,
    SPLIT_DIR_RETRAIN,
    TEST_DIR_SPLIT,
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

# La funzione freeze della backbone serve a bloccare l'aggiornamento dei pesi durante il training,
# questo consente di ridurre la memoria richiesta e i tempi necessari per ogni epoce di addestramento.
# Questa scelta è stata fatta per rendere il progetto portabile e facilmente riproducibile in ambito didattico.
def freeze_model(model) -> None:
    # Congela tutti i paramentri della backbone
    for p in model.base_model.parameters():
        p.requires_grad = False

    try:
        # Recupera i layer dell'encoder per poter riattivare solo gli ultimi N.
        layers = model.base_model.encoder.layer
        n = len(layers)
        k = UNFREEZE_LAST_N_LAYERS

        # Mantiene k in un intervallo valido.
        # Se negativo diventa 0, se supera il numero di layer viene limitato a n.
        if k < 0:
            k = 0
        if k > n:
            k = n

        # Riattiva solo gli ultimi k layer dell'encoder.    
        for layer in layers[n - k:]:
            for p in layer.parameters():
                p.requires_grad = True
    except Exception:
        pass

    # Il classifier finale resta sempre trainabile
    try:
        for p in model.classifier.parameters():
            p.requires_grad = True
    except Exception:
        pass

#Funzione principale dove avvengono tutti i processi per il training, salvataggio e push su Hugging Face del modello.
def main()-> None:
    # Il retraining utilizza lo stesso dataset già prepocessato del performace exporter. 
    # Lo split crea i data frame train e val, il test verrà effettuato sullo split condiviso con il file di training iniziale.   
    split_retraining_data()
    
    ARTIFACT_MODEL_DIR_RETRAIN.mkdir(parents=True, exist_ok=True)

    train_df = load_split(SPLIT_DIR_RETRAIN, "train")
    val_df = load_split(SPLIT_DIR_RETRAIN, "val")
    test_df = load_split(TEST_DIR_SPLIT, "test")

    # Load del tokenizer e del modello precedentemente addestrato da Hugging Face.
    tokenizer = build_tokenizer(MODEL_ID)
    model = build_model(MODEL_ID)

    # Condizione di uso della funzione freeze
    if FREEZE_BASE:
        freeze_model(model)

    #Conversione dei DataFrame in dataset Hugging Face compatibili con Trainer e tokenizzazione dei testi
    train_ds = tokenize_dataset(to_hf_dataset(train_df, TEXT_COL, LABEL_COL), tokenizer)
    val_ds = tokenize_dataset(to_hf_dataset(val_df, TEXT_COL, LABEL_COL), tokenizer)
    test_ds = tokenize_dataset(to_hf_dataset(test_df, TEXT_COL, LABEL_COL), tokenizer)

    # max_steps viene usato come ulteriore misura per ridurre i tempi di retraining, infatti ha priorità sul numero di epoche di training.
    # Se il valore di MAX_STEPS è positivo, impone un limite al numero totale di step interrompendo il training anche se le epoche non sono terminate. 
    # Se il valoer di MAX_STEPS è negativo non veine imposto alcun limite esplicito e vengono usate il numero di epoche configurate.
    max_steps_value = MAX_STEPS if MAX_STEPS and MAX_STEPS > 0 else -1

    # Configurazione dei Training Arguments per il fine-tuning:
    # Alcuni parametri sono i medesimi utilizzati nel training principale, 
    # altri sono specifici per rendere il retraining facilmente riproducibile non puntando ad avere performance migliori.
    # Ciò che si diffrenzia maggiormente dal training iniziale sono il numero di epoche configurabili e l'uso di max_steps.
    args = TrainingArguments(
        output_dir=str(ARTIFACT_MODEL_DIR_RETRAIN),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,    
        learning_rate=LEARNING_RATE,  
        num_train_epochs=NUM_EPOCHS,                   
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        save_total_limit=1,        
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        max_steps=max_steps_value,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
    )

    #Configurazione del Trainer.
    #Il Trainer gestisce il ciclo di training e valutazione, ricevendo il modello, TrainingArguments,
    #i dataset, la funzione per le metriche.
    #Ho eseguito dei test di retraining e ho ritenuto non necessario implementare EarlyStoppingCallBack come fatto nel training,
    #il freeze del backbone e il limite opzionale su max_steps sono sufficienti per ottenere un fine-tuning in tempi accettabili.
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

    reports_dir = Path("reports/retrain")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "retrain_metrics.json"

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