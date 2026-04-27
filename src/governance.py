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

# Agrupamento de UF por região — resolve o problema de estados com poucas amostras
# e atende o requisito de "Localidade" da diretoria de forma estatisticamente válida
UF_REGIAO = {
    "AC": "Norte",   "AM": "Norte",   "AP": "Norte",  "PA": "Norte",
    "RO": "Norte",   "RR": "Norte",   "TO": "Norte",
    "AL": "Nordeste","BA": "Nordeste","CE": "Nordeste","MA": "Nordeste",
    "PB": "Nordeste","PE": "Nordeste","PI": "Nordeste","RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste","GO": "Centro-Oeste","MS": "Centro-Oeste","MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul",     "RS": "Sul",     "SC": "Sul",
    # nomes por extenso que aparecem no CSV
    "São Paulo":      "Sudeste",
    "Rio de Janeiro": "Sudeste",
    "Minas Gerais":   "Sudeste",
    "Bahia":          "Nordeste",
    "Rio Grande do Sul": "Sul",
    "Paraná":         "Sul",
    "Pernambuco":     "Nordeste",
    "Ceará":          "Nordeste",
    "Goiás":          "Centro-Oeste",
    "Maranhão":       "Nordeste",
}

recall_safe    = partial(recall_score,    zero_division=0)
precision_safe = partial(precision_score, zero_division=0)


def clean_sensitive(sensitive: pd.DataFrame) -> pd.DataFrame:
    """
    Converte UF → Região para auditoria geográfica válida.
    Estados com poucas amostras são agrupados na mesma região,
    garantindo significância estatística e cobertura do requisito de Localidade.
    """
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

        # Filtra grupos com mínimo de 3 positivos reais no test set
        grupos_validos = [
            grupo for grupo in col_data.dropna().unique()
            if y_test[col_data == grupo].sum() >= 3
        ]

        mask = col_data.isin(grupos_validos)
        if mask.sum() == 0:
            print(f"\n[{col}] sem grupos válidos — pulando")
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

        print(f"\n[{col}] disparidade de recall: {disparidade:.4f} → {status}")
        print(mf.by_group.to_string())

        if disparidade > FAIRNESS_THRESHOLD:
            blocked = True

    if blocked:
        print(f"\nAUDITORIA FALHOU: disparidade > {FAIRNESS_THRESHOLD}. Modelo não registrado.")
        sys.exit(1)

    print("\nAuditoria de fairness aprovada.")
    return results


def generate_model_card(metrics: dict, fairness_report: dict):
    card = {
        "model_name": "ChurnModel v1",
        "uso_pretendido": "Previsão de churn de clientes para ação preventiva de retenção",
        "atributos_sensiveis_auditados": ["Sexo", "Regiao (Norte/Nordeste/Centro-Oeste/Sudeste/Sul)"],
        "atributos_sensiveis_excluidos_do_modelo": True,
        "metodologia_auditoria": (
            "UFs agrupadas por região geográfica para garantir significância estatística. "
            "Grupos com menos de 3 amostras positivas no conjunto de teste são excluídos. "
            "Métrica de disparidade: diferença máxima de recall entre grupos (threshold: 15%)."
        ),
        "limitacoes": [
            "Treinado com dados históricos — pode não refletir perfis futuros",
            "Não usar como único critério de decisão de relacionamento com o cliente",
            "Auditoria regional — variações intra-regionais por estado não são capturadas"
        ],
        "metricas_gerais": metrics,
        "auditoria_fairness": fairness_report,
        "lgpd": "Atributos sensíveis (Sexo, UF) não utilizados como features do modelo."
    }
    with open("model_card.json", "w") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print("\nModel Card salvo em model_card.json")
    return card


if __name__ == "__main__":
    import mlflow.sklearn
    from evaluate import compute_metrics
    from sklearn.model_selection import train_test_split

    X, y, sensitive = load_data_with_sensitive(DATA_PATH)
    sensitive = clean_sensitive(sensitive)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )

    model = mlflow.sklearn.load_model("models:/ChurnModel@champion")
    metrics = compute_metrics(model, X_test, y_test)
    fairness = audit_fairness(model, X_test, y_test, s_test)
    generate_model_card(metrics, fairness)