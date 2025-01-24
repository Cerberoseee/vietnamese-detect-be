from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.dto import TextRequest, PredictionResponse
from utils.const import label_to_value

class XLMRobertaService:  
  def __init__(self):
    self.model_name = 'assets/xlm-roberta'
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=True)

  def predict(self, request: TextRequest):
    inputs = self.tokenizer(
      request.text,
      padding="max_length",
      truncation=True,
      max_length=256,
      return_tensors='pt'
    )
    with torch.no_grad():
      outputs = self.model(**inputs)
      probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
      predictions = torch.argmax(probs, dim=-1)

    labels = [self.model.config.id2label[idx.item()] for idx in predictions]
    probabilities = probs.tolist()
    return PredictionResponse(prediction=label_to_value[labels[0]], probability=probabilities[0])
