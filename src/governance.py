import json
import sys
import os
from functools import partial

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from fairlearn.metrics import MetricFrame
from sklearn.metrics import recall_score, precision_score
from preprocess import load_data_with_sensitive

FAIRNESS_THRESHOLD = 0.15
DATA_PATH = "data/raw/dados_clientes.csv"

UF_REGIAO = {
    "AC": "Norte",   "AM": "Norte",   "AP": "Norte",  "PA": "Norte",
    "RO": "Norte",   "RR": "Norte",   "TO": "Norte",
    "AL": "Nordeste","BA": "Nordeste","CE": "Nordeste","MA": "Nordeste",
    "PB": "Nordeste","PE": "Nordeste","PI": "Nordeste","RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste","GO": "Centro-Oeste","MS": "Centro-Oeste","MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul",     "RS": "Sul",     "SC": "Sul",
    "Sao Paulo":      "Sudeste",
    "Rio de Janeiro": "Sudeste",
    "Minas Gerais":   "Sudeste",
    "Bahia":          "Nordeste",
    "Rio Grande do Sul": "Sul",
    "Parana":         "Sul",
    "Pernambuco":     "Nordeste",
    "Ceara":          "Nordeste",
    "Goias":          "Centro-Oeste",
    "Maranhao":       "Nordeste",
}

recall_safe    = partial(recall_score,    zero_division=0)
precision_safe = partial(precision_score, zero_division=0)


def clean_sensitive(sensitive: pd.DataFrame) -> pd.DataFrame:
    sensitive = sensitive.copy()
    sensitive["Regiao"] = sensitive["UF"].map(UF_REGIAO)
    sensitive = sensitive.drop(columns=["UF"])
    return sensitive


def audit_fairness(model, X_test, y_test, sensitive):
    y_pred = model.predict(X_test)
    results = {}
    blocked = False

    for col in sensitive.columns:
        col_data = sensitive[col].copy()

        grupos_validos = [
            grupo for grupo in col_data.dropna().unique()
            if y_test[col_data == grupo].sum() >= 3
        ]

        mask = col_data.isin(grupos_validos)
        if mask.sum() == 0:
            print(f"\n[{col}] sem grupos validos — pulando")
            continue

        mf = MetricFrame(
            metrics={"recall": recall_safe, "precision": precision_safe},
            y_true=y_test[mask],
            y_pred=y_pred[mask],
            sensitive_features=col_data[mask]
        )

        disparidade = mf.difference()["recall"]
        status = "BLOQUEADO" if disparidade > FAIRNESS_THRESHOLD else "aprovado"

        results[col] = {
            "disparidade_recall": round(float(disparidade), 4),
            "status": status,
            "por_grupo": mf.by_group.round(4).to_dict()
        }

        print(f"\n[{col}] disparidade de recall: {disparidade:.4f} -> {status}")
        print(mf.by_group.to_string())

        if disparidade > FAIRNESS_THRESHOLD:
            blocked = True

    if blocked:
        print(f"\nAUDITORIA FALHOU: disparidade > {FAIRNESS_THRESHOLD}. Modelo nao registrado.")
        sys.exit(1)

    print("\nAuditoria de fairness aprovada.")
    return results


def generate_model_card(metrics: dict, fairness_report: dict):
    card = {
        "model_name": "ChurnModel v1",
        "uso_pretendido": "Previsao de churn de clientes para acao preventiva de retencao",
        "atributos_sensiveis_auditados": ["Sexo", "Regiao"],
        "atributos_sensiveis_excluidos_do_modelo": True,
        "metodologia_auditoria": (
            "UFs agrupadas por regiao geografica para garantir significancia estatistica. "
            "Grupos com menos de 3 amostras positivas no conjunto de teste sao excluidos. "
            "Metrica de disparidade: diferenca maxima de recall entre grupos (threshold: 15%)."
        ),
        "limitacoes": [
            "Treinado com dados historicos - pode nao refletir perfis futuros",
            "Nao usar como unico criterio de decisao de relacionamento com o cliente",
            "Auditoria regional - variacoes intra-regionais por estado nao sao capturadas"
        ],
        "metricas_gerais": metrics,
        "auditoria_fairness": fairness_report,
        "lgpd": "Atributos sensiveis (Sexo, UF) nao utilizados como features do modelo."
    }
    with open("model_card.json", "w") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print("\nModel Card salvo em model_card.json")
    return card


def _load_model_from_mlruns(mlruns_root="mlruns"):
    """Carrega o modelo mais recente do mlruns sem depender do registry ou alias."""
    import mlflow.sklearn
    candidates = []
    for root, _dirs, files in os.walk(mlruns_root):
        if "MLmodel" in files and "artifacts" in root:
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {mlruns_root}")
    candidates.sort(key=lambda p: os.path.getmtime(os.path.join(p, "MLmodel")), reverse=True)
    print(f"Modelo carregado de: {candidates[0]}")
    return mlflow.sklearn.load_model(candidates[0])


if __name__ == "__main__":
    import mlflow.sklearn
    from evaluate import compute_metrics
    from sklearn.model_selection import train_test_split

    X, y, sensitive = load_data_with_sensitive(DATA_PATH)
    sensitive = clean_sensitive(sensitive)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    # Tenta alias local primeiro; se falhar, carrega pelo mlruns diretamente
    # Isso garante que funciona tanto localmente quanto no CI/CD
    try:
        model = mlflow.sklearn.load_model("models:/ChurnModel@champion")
        print("Modelo carregado via alias champion")
    except Exception:
        print("Alias nao encontrado, carregando pelo mlruns...")
        model = _load_model_from_mlruns()

    metrics = compute_metrics(model, X_test, y_test)
    fairness = audit_fairness(model, X_test, y_test, s_test)
    generate_model_card(metrics, fairness)