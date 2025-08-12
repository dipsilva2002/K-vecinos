import os
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

def ensure_dirs():
    for d in ["data/raw", "data/processed", "models", "reports/figures"]:
        os.makedirs(d, exist_ok=True)

def load_raw_wine_data(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url, sep=';')
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/winequality-red.csv", index=False)
    return df

def add_quality_label(df: pd.DataFrame) -> pd.DataFrame:
    if "quality" not in df.columns:
        raise ValueError("No se encontró la columna 'quality' en el dataset.")
    df = df.copy()
    def to_label(q: int) -> int:
        if q <= 5: return 0
        if q == 6: return 1
        return 2
    df["label"] = df["quality"].apply(to_label)
    return df

def save_processed(df: pd.DataFrame, name: str = "winequality_red_with_labels.csv") -> str:
    os.makedirs("data/processed", exist_ok=True)
    path = os.path.join("data", "processed", name)
    df.to_csv(path, index=False)
    return path
