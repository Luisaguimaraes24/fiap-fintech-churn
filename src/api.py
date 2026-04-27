import json
import datetime
import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="FIAP Fintech Churn API", version="1.0")

def _guess_model_uri_from_mlruns(mlruns_root: str = "/app/mlruns") -> str | None:
    """
    Em Docker, o treino pode ter gerado apenas o artefato do *modelo registrado*
    em `mlruns/*/models/*/artifacts/` (e não o artefato `runs:/.../churn_model`).
    """
    try:
        candidates = []
        for root, _dirs, files in os.walk(mlruns_root):
            if "MLmodel" in files and root.endswith("/artifacts"):
                candidates.append(root)
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(os.path.join(p, "MLmodel")), reverse=True)
        return candidates[0]
    except Exception:
        return None


# Preferir `MODEL_URI` via env; se não vier, tentar inferir do `mlruns` dentro do container.
# Fallback final: `runs:/.../churn_model` se você realmente tiver esse artifact.
RUN_ID = os.getenv("RUN_ID", "7c975cc041e64347be6a9e0dbdcc2fd8")
MODEL_URI = os.getenv("MODEL_URI") or _guess_model_uri_from_mlruns() or f"runs:/{RUN_ID}/churn_model"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    print(f"Modelo carregado: {MODEL_URI}")
except Exception as e:
    print(f"Aviso: modelo não carregado ({e}). Use /health para verificar.")
    model = None

os.makedirs("logs", exist_ok=True)


class ClienteInput(BaseModel):
    Idade: float
    RendaMensal: float
    PercentualUtilizacaoLimite: float
    QtdTransacoesNegadas: float
    AnosDeRelacionamentoBanco: float
    JaUsouChequeEspecial: float
    QtdEmprestimos: float
    NumeroAtendimentos: float
    TMA: float
    IndiceSatisfacao: float
    Saldo: float
    CLTV: float
    CanalPref: str


@app.get("/health")
def health():
    return {
        "status": "ok" if model else "modelo_nao_carregado",
        "model_uri": MODEL_URI,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


@app.post("/predict")
def predict(cliente: ClienteInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não disponível")
    try:
        df = pd.DataFrame([cliente.model_dump()])
        prob = float(model.predict_proba(df)[0][1])
        pred = int(prob >= 0.5)

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "input_features": cliente.model_dump(),
            "prediction": pred,
            "probability": round(prob, 4),
            "model_uri": MODEL_URI
        }
        with open("logs/predictions.jsonl", "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return {
            "churn": pred,
            "probability": round(prob, 4),
            "risco": "alto" if prob >= 0.7 else "médio" if prob >= 0.4 else "baixo"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))