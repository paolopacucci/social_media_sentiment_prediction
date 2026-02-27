from pydantic import BaseModel


class PredictRequest(BaseModel):
    texts: list[str]

class PredictItem(BaseModel):
    label_id: int
    label: str
    score: float

class PredictResponse(BaseModel):
    outputs: list[PredictItem]