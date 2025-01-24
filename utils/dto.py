from typing import List
from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: List[float]