from fastapi import FastAPI
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class RequestPost(BaseModel):
   text:str


class CustomBert(nn.Module):
  def __init__(self, name_or_model_path="bert-base-uncased", n_classes=6):
    super(CustomBert, self).__init__()
    self.bert_pretrained = BertModel.from_pretrained(name_or_model_path)
    self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
    x = self.classifier(x.pooler_output)

    return x

model = CustomBert()
model.load_state_dict(torch.load("my_custom_bert_kk.pth", map_location=torch.device('cpu')))

def classifier_fn(text:str):
    labels = {0:"computer-science", 1:"physics", 2:"mathematics", 3:"quantitative-biology" ,4: "statistics", 5:"quantitative-finance"}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, padding="max_length", max_length=250, truncation=True, return_tensors="pt")

    output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    _ , pred = torch.max(output, dim=1)

    return labels[pred.item()]


@app.post("/predict")
def prediction(request: RequestPost):
   
   return {
      "prediction": classifier_fn(request.text)
      }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=4445)