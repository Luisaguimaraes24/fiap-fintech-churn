import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_data, build_preprocessor
from evaluate import compute_metrics

DATA_PATH = "data/raw/dados_clientes.csv"
EXPERIMENT = "fiap-fintech-churn"
RANDOM_STATE = 42
MIN_ACCURACY = 0.80


def train():
    X, y = load_data(DATA_PATH)
    print(f"Dataset carregado: {X.shape[0]} linhas, {X.shape[1]} features")
    print(f"Distribuição de churn: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ])

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)

        print("\nMétricas:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Gatekeeper — bloqueia registro se abaixo do limiar
        if metrics["accuracy"] < MIN_ACCURACY:
            print(f"\nBLOQUEADO: accuracy {metrics['accuracy']} < {MIN_ACCURACY}")
            raise ValueError("Modelo abaixo do limiar mínimo de acurácia.")

        mlflow.log_params({"n_estimators": 100, "max_depth": 8, "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(DATA_PATH)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, "churn_model",
            signature=signature,
            registered_model_name="ChurnModel"
        )

        # Salva métricas em JSON para o CI/CD ler
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\nModelo registrado no MLflow com sucesso!")


if __name__ == "__main__":
    train()