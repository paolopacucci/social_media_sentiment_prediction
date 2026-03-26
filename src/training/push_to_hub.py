from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

from src.config import HF_TOKEN, MODEL_ID

# La funzione pubblica su Hugging Face da una cartella locale gli artifacts di un modello.
def push_model_dir(model_dir: str | Path, commit_prefix: str = "model") -> None:
    # Verifica se il modello e il token esistono.
    if not MODEL_ID:
        raise ValueError("MODEL_ID is empty")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is empty")

    # Normalizza il path in oggetto Path e verifica che la directory locale esista davvero-
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    # Crea il client autenticato per interagire con Hugging Face Hub.
    api = HfApi(token=HF_TOKEN)

    # Crea la repo se non esiste già.
    api.create_repo(repo_id=MODEL_ID, exist_ok=True)

    # Crea commit descrittivo e timestamp per distinguere training iniziale e i futuri retraining.
    commit_message = f"{commit_prefix}: {datetime.utcnow().isoformat()}Z"

    # Carica la cartella completa degli artifacts nella repo del modello su Hugging Face. 
    api.upload_folder(
        repo_id=MODEL_ID,
        folder_path=str(model_dir),
        commit_message=commit_message,
    )

    print(f"Pushed to HF: {MODEL_ID}")
    print(f"Source dir: {model_dir}")
    print(f"Commit: {commit_message}")