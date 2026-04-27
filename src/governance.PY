# src/governance.py
import json
import sys
import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score
from preprocess import load_data
import mlflow.sklearn

FAIRNESS_THRESHOLD = 0.15  # máx. disparidade tolerada

def audit_fairness(model, X_test, y_test, sensitive_col="genero"):
    y_pred = model.predict(X_test)
    sensitive = X_test[sensitive_col]

    mf = MetricFrame(
        metrics={
            "recall":    recall_score,
            "precision": precision_score,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    disparidade_recall = mf.difference()["recall"]
    print("Métricas por grupo:")
    print(mf.by_group)
    print(f"\nDisparidade de recall: {disparidade_recall:.4f}")

    if disparidade_recall > FAIRNESS_THRESHOLD:
        print(f"BLOQUEADO: disparidade {disparidade_recall:.4f} > {FAIRNESS_THRESHOLD}")
        sys.exit(1)  # falha o pipeline CI/CD
    print("Auditoria aprovada.")
    return mf

def generate_model_card(metrics: dict, fairness_report: dict):
    card = {
        "model_name": "ChurnModel v1",
        "uso_pretendido": "Previsão de churn de clientes para ação preventiva",
        "atributos_sensiveis": ["genero", "localidade"],
        "limitacoes": [
            "Treinado com dados históricos — pode não refletir novos perfis",
            "Não usar como único critério de decisão"
        ],
        "metricas_gerais": metrics,
        "metricas_por_grupo": fairness_report,
        "lgpd": "Modelo auditado. Atributos sensíveis não usados como features."
    }
    with open("model_card.json", "w") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print("Model Card gerado em model_card.json")
    return card