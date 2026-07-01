import os  
import mlflow.pyfunc  
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI  
from pydantic import BaseModel  
from typing import Dict, List, Any  
  
app = FastAPI(title="Multifactor Portfolio Scoring API")  
  
class PredictionRequest(BaseModel):  
    # Expects a dict representing model inputs (klines_dict, ls_ratio_df, etc.)
    data: Dict[str, Any]  
  
model = None  
  
@app.on_event("startup")  
def load_model():  
    global model  
    model_uri = os.environ.get("MODEL_URI", "./model_artifacts")  
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")  
    if tracking_uri:  
        mlflow.set_tracking_uri(tracking_uri)  
    model = mlflow.pyfunc.load_model(model_uri)  
    print(f"Loaded MLflow PyFunc model from: {model_uri}")
  
@app.post("/predict")  
def predict(request: PredictionRequest):  
    res = model.predict(request.data)  
    return {"prediction": res}  
  
@app.get("/health")  
def health():  
    return {"status": "healthy"}  
