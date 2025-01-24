from fastapi import FastAPI
from services.e5_service import E5Service
from services.xlmroberta_service import XLMRobertaService
from services.bartpho_service import BARTPhoService
from utils.dto import TextRequest, PredictionResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

e5Service = E5Service()
xlmRobertaService = XLMRobertaService()
bARTPhoService = BARTPhoService()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.post("/e5-predict", response_model=PredictionResponse)
def e5_predict(request: TextRequest):
  return e5Service.predict(request)

@app.post("/bartpho-predict", response_model=PredictionResponse)
def bartpho_predict(request: TextRequest):
  return bARTPhoService.predict(request)

@app.post("/xlm-roberta-predict", response_model=PredictionResponse)
def xlm_roberta_predict(request: TextRequest):
  return xlmRobertaService.predict(request)