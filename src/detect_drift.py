import json
import pandas as pd
from scipy import stats

REF_PATH = "data/raw/dados_clientes.csv"
LOG_PATH = "logs/predictions.jsonl"
FEATURES = ["idade", "renda", "satisfacao", "historico_meses"]
P_VALUE_THRESHOLD = 0.05

def load_logs():
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            rows.append(entry["input_features"])
    return pd.DataFrame(rows)

def detect():
    ref = pd.read_csv(REF_PATH)
    prod = load_logs()
    drift_detected = False
    print("=== Relatório de Drift (Kolmogorov-Smirnov) ===")
    for feat in FEATURES:
        if feat not in ref or feat not in prod:
            continue
        stat, p = stats.ks_2samp(ref[feat].dropna(), prod[feat].dropna())
        status = "DRIFT" if p < P_VALUE_THRESHOLD else "ok"
        print(f"  {feat:20s}  KS={stat:.3f}  p={p:.4f}  [{status}]")
        if p < P_VALUE_THRESHOLD:
            drift_detected = True
    if drift_detected:
        print("\nALERTA: drift detectado. Considere acionar retreino.")
        # Para retreino automático via GitHub Actions:
        # import subprocess
        # subprocess.run(["gh", "workflow", "run", "ml_pipeline.yml"])
    else:
        print("\nNenhum drift significativo detectado.")

if __name__ == "__main__":
    detect()