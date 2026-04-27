import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Colunas que não entram no modelo (identificadores, dados pessoais, geo-detalhado)
DROP_COLS = [
    "CodigoCliente", "Titulo", "PrimeiroNome", "UltimoNome",
    "Endereco", "Cidade", "CEP", "Email",
    "UFCompleto", "PaisCompleto", "Pais"
]

# Atributos sensíveis — usados APENAS na auditoria de fairness, não como features
SENSITIVE_COLS = ["Sexo", "UF"]

TARGET_COL = "Churn"


def load_data(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS + SENSITIVE_COLS, errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)
    return X, y


def load_data_with_sensitive(path: str):
    """Versão usada pela auditoria de fairness — retorna atributos sensíveis separados."""
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS, errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    sensitive = df[SENSITIVE_COLS].copy()
    X = df.drop(columns=[TARGET_COL] + SENSITIVE_COLS)
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)
    return X, y, sensitive


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return preprocessor