from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

from src.config import HF_TOKEN, MODEL_ID


def push_model_dir(model_dir: str | Path, commit_prefix: str = "model") -> None:
    if not MODEL_ID:
        raise ValueError("MODEL_ID is empty")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is empty")

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=MODEL_ID, exist_ok=True)

    commit_message = f"{commit_prefix}: {datetime.utcnow().isoformat()}Z"
    api.upload_folder(
        repo_id=MODEL_ID,
        folder_path=str(model_dir),
        commit_message=commit_message,
    )

    print(f"Pushed to HF: {MODEL_ID}")
    print(f"Source dir: {model_dir}")
    print(f"Commit: {commit_message}")