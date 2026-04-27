import json
import os
import pandas as pd
from scipy import stats

REF_PATH = "data/raw/dados_clientes.csv"
LOG_PATH = "logs/predictions.jsonl"
FEATURES = ["Idade", "RendaMensal", "IndiceSatisfacao", "Saldo", "PercentualUtilizacaoLimite"]
P_VALUE_THRESHOLD = 0.05


def load_logs():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Log nao encontrado: {LOG_PATH}")
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            rows.append(entry["input_features"])
    return pd.DataFrame(rows)


def detect():
    ref = pd.read_csv(REF_PATH)
    prod = load_logs()

    print(f"Dados de referencia: {len(ref)} amostras")
    print(f"Dados de producao (logs): {len(prod)} amostras")
    print("\n=== Relatorio de Drift (Kolmogorov-Smirnov) ===\n")

    drift_detected = False
    for feat in FEATURES:
        if feat not in ref.columns or feat not in prod.columns:
            print(f"  {feat:35s} [coluna ausente — pulando]")
            continue
        stat, p = stats.ks_2samp(ref[feat].dropna(), prod[feat].dropna())
        status = "DRIFT DETECTADO" if p < P_VALUE_THRESHOLD else "ok"
        print(f"  {feat:35s} KS={stat:.3f}  p={p:.4f}  [{status}]")
        if p < P_VALUE_THRESHOLD:
            drift_detected = True

    print()
    if drift_detected:
        print("ALERTA: drift significativo detectado em uma ou mais features.")
        print("Acao recomendada: acionar retreino do modelo com dados recentes.")
    else:
        print("Nenhum drift significativo detectado. Modelo estavel.")


if __name__ == "__main__":
    detect()
