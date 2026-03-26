import sys
from pathlib import Path

# Aggiunge la root del progetto al sys.path per permettere ai test di 
# importare correttamentei moduli della repository.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
