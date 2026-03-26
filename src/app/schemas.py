from pydantic import BaseModel


# Schema della richiesta batch inviata all'endpoint /predict.
class PredictRequest(BaseModel):
    texts: list[str]

# Schema di una singola predizione restituita dal modello.
class PredictItem(BaseModel):
    label_id: int
    label: str
    score: float

# Schema della risposta completa dell'endpoint /predict.
class PredictResponse(BaseModel):
    outputs: list[PredictItem]