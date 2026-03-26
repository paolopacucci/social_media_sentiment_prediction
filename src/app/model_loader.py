import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import HF_TOKEN, MAX_LENGTH, MODEL_ID

# Mappa gli id numerici delle classi nelle label testuali restituite dall'API.
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Inizializza variabili all'avvio del servizio evitando di doverle ricaricarle ad ogni richiesta.
tokenizer = None
model = None
device = None
model_source = None


# All'avvio dell'API carica il modello trainato da Hugging Face, lo imposta in modalità evaluation e lo sposta sul device disponibile.
def load_model() -> None:
    global tokenizer, model, device, model_source

    # Verifica che la repo su Hugging Face (MODEL_ID) sia specificata.
    if not MODEL_ID:
        raise ValueError("MODEL_ID is empty.")

    # Carica tokenizer e modello dalla repo su Hugging Face 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN or None,
    )
    # Il modello viene usato solo per per inferenza.
    model.eval()

    # Usa GPU se disponibile, altrimenti CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tiene traccia della sorgente del modello per health check.
    model_source = MODEL_ID


# Esegue l'inferenza batch su una lista di testi ricevuti dall'API
def predict_texts(
    texts: list[str],
) -> tuple[list[int], list[str], list[float], float, float]:
    # Se il tokenizer e il modello non sono caricati non si può eseguire l'inferenza.
    if tokenizer is None or model is None or device is None:
        raise RuntimeError("Model not loaded")

    # Tokenizza i testi e costruisce i tensori PyTorch.
    enc = tokenizer(
        texts,
        truncation=True,        # Limita lunghezza massima degli input.
        padding=True,           # Uniforma le sequenze per poter fare batch inference.
        max_length=MAX_LENGTH,  # Limita lunghezza massima degli input. 
        return_tensors="pt",    
    ).to(device)

    # Misura il tempo di inferenza del modello escludendo la tokenizzazione.
    t0 = time.perf_counter()
    # Disattiva il calcolo dei gradienti durante l'inferenza riducendo memoria usata e costo computazionale.
    with torch.no_grad():
        logits = model(**enc).logits
    infer_s = time.perf_counter() - t0

    # Converte i logits in probabilità con softmax e seleziona,
    # per ogni testo, la classe più probabile e il relativo score.
    probs = torch.softmax(logits, dim=-1)
    scores, pred_ids = torch.max(probs, dim=-1)

    # Riporta i risultati su CPU e li converte in liste Python in modo tale da poterli usare facilmente nel resto dell'applicazione.
    pred_ids = pred_ids.cpu().tolist()
    scores = scores.cpu().tolist()
    labels = [ID2LABEL[i] for i in pred_ids]

    # Calcola la confidence media della batch.
    avg_conf = float(sum(scores) / len(scores)) if scores else 0.0
    return pred_ids, labels, scores, infer_s, avg_conf