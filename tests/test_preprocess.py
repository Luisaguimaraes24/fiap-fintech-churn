import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import load_data, build_preprocessor


def make_df(tmp_path):
    df = pd.DataFrame({
        "CodigoCliente": [1, 2, 3, 4],
        "Titulo": ["Sr.", "Sra.", "Sr.", "Sra."],
        "PrimeiroNome": ["João", "Maria", "Pedro", "Ana"],
        "UltimoNome": ["Silva", "Santos", "Lima", "Costa"],
        "Endereco": ["Rua A", "Rua B", "Rua C", "Rua D"],
        "Cidade": ["SP", "RJ", "MG", "BA"],
        "CEP": ["01000", "20000", "30000", "40000"],
        "Email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com"],
        "UFCompleto": ["São Paulo", "Rio de Janeiro", "Minas Gerais", "Bahia"],
        "PaisCompleto": ["Brasil"] * 4,
        "Pais": ["BR"] * 4,
        "Sexo": ["Masculino", "Feminino", "Masculino", "Feminino"],
        "UF": ["SP", "RJ", "MG", "BA"],
        "Idade": [30, None, 45, 22],
        "RendaMensal": [5000.0, 3200.0, None, 7800.0],
        "PercentualUtilizacaoLimite": [60.0, 80.0, 40.0, 90.0],
        "QtdTransacoesNegadas": [1, 3, 0, 5],
        "AnosDeRelacionamentoBanco": [5, 2, 10, 1],
        "JaUsouChequeEspecial": [0, 1, 0, 1],
        "QtdEmprestimos": [1, 0, 2, 0],
        "NumeroAtendimentos": [3, 7, 1, 10],
        "TMA": [5, 20, 3, 50],
        "IndiceSatisfacao": [4, 2, 5, 1],
        "Saldo": [10000.0, 500.0, 25000.0, 200.0],
        "CLTV": [70, 30, 90, 20],
        "CanalPref": ["App", "Email", "SMS", "Push"],
        "Churn": [0, 1, 0, 1],
    })
    path = str(tmp_path / "dados_clientes.csv")
    df.to_csv(path, index=False)
    return path


def test_load_data_remove_target(tmp_path):
    path = make_df(tmp_path)
    X, y = load_data(path)
    assert "Churn" not in X.columns


def test_load_data_remove_sensitive(tmp_path):
    path = make_df(tmp_path)
    X, y = load_data(path)
    assert "Sexo" not in X.columns
    assert "UF" not in X.columns


def test_load_data_remove_identifiers(tmp_path):
    path = make_df(tmp_path)
    X, y = load_data(path)
    assert "CodigoCliente" not in X.columns
    assert "Email" not in X.columns


def test_no_nulls_after_preprocessing(tmp_path):
    path = make_df(tmp_path)
    X, y = load_data(path)
    pre = build_preprocessor(X)
    X_out = pre.fit_transform(X)
    assert not np.any(np.isnan(X_out))


def test_no_inf_in_output(tmp_path):
    path = make_df(tmp_path)
    X, y = load_data(path)
    pre = build_preprocessor(X)
    X_out = pre.fit_transform(X)
    assert np.all(np.isfinite(X_out))


def test_target_is_binary(tmp_path):
    path = make_df(tmp_path)
    _, y = load_data(path)
    assert set(y.unique()).issubset({0, 1})