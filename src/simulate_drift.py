"""
Simula drift enviando dados artificialmente alterados para a API.
Útil para testar se o sistema de detecção está funcionando.
"""
import requests
import random
import json

API_URL = "http://localhost:8000/predict"

def simulate(n=200, drift=True):
    for i in range(n):
        cliente = {
            "idade":            random.gauss(55, 8) if drift else random.gauss(35, 8),
            "renda":            random.gauss(2500, 500) if drift else random.gauss(5000, 1200),
            "satisfacao":       random.randint(1, 2) if drift else random.randint(2, 5),
            "historico_meses":  random.randint(1, 6) if drift else random.randint(12, 60),
            "num_produtos":     1,
        }
        r = requests.post(API_URL, json=cliente, timeout=5)
        if i % 50 == 0:
            print(f"[{i+1}/{n}] {r.json()}")

if __name__ == "__main__":
    print("Simulando dados com drift...")
    simulate(n=200, drift=True)
    print("Concluído. Rode detect_drift.py para verificar.")