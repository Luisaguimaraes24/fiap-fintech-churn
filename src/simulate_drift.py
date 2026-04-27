import requests
import random

API_URL = "http://localhost:8001/predict"


def simulate(n=200, drift=True):
    ok = 0
    for i in range(n):
        cliente = {
            "Idade": random.gauss(62, 5) if drift else random.gauss(38, 8),
            "RendaMensal": random.gauss(2000, 400) if drift else random.gauss(5500, 1200),
            "PercentualUtilizacaoLimite": random.gauss(88, 5) if drift else random.gauss(55, 15),
            "QtdTransacoesNegadas": random.randint(5, 12) if drift else random.randint(0, 3),
            "AnosDeRelacionamentoBanco": random.randint(1, 3) if drift else random.randint(5, 20),
            "JaUsouChequeEspecial": 1 if drift else random.randint(0, 1),
            "QtdEmprestimos": random.randint(0, 1),
            "NumeroAtendimentos": random.randint(10, 20) if drift else random.randint(1, 5),
            "TMA": random.gauss(75, 10) if drift else random.gauss(10, 5),
            "IndiceSatisfacao": random.randint(1, 2) if drift else random.randint(3, 5),
            "Saldo": random.gauss(3000, 500) if drift else random.gauss(15000, 5000),
            "CLTV": random.randint(20, 40) if drift else random.randint(50, 90),
            "CanalPref": random.choice(["Email", "SMS", "Push", "App"]),
        }
        try:
            r = requests.post(API_URL, json=cliente, timeout=5)
            ok += 1
            if i % 50 == 0:
                print(f"[{i+1}/{n}] churn={r.json().get('churn')} prob={r.json().get('probability')}")
        except Exception as e:
            print(f"Erro na requisicao {i}: {e}")

    print(f"\nConcluido: {ok}/{n} requisicoes enviadas.")
    print("Rode agora: python src/detect_drift.py")


if __name__ == "__main__":
    print("Simulando drift — perfil alterado (clientes mais velhos, menor renda, insatisfeitos)")
    simulate(n=200, drift=True)
