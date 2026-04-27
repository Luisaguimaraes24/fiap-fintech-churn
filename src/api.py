from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import json
import datetime

app = FastAPI(title="FIAP Fintech Churn API")

MODEL_NAME = "ChurnModel"
MODEL_STAGE = "Staging"
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

class ClienteInput(BaseModel):
    idade: float
    renda: float
    satisfacao: int
    historico_meses: int
    num_produtos: int

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE}

@app.post("/predict")
def predict(cliente: ClienteInput):
    try:
        df = pd.DataFrame([cliente.model_dump()])
        prob = model.predict_proba(df)[0][1]
        pred = int(prob >= 0.5)
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "input_features": cliente.model_dump(),
            "prediction": pred,
            "probability": round(prob, 4),
            "model_version": MODEL_STAGE
        }
        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        return {"churn": pred, "probability": round(prob, 4)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))