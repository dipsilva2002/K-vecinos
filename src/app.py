from __future__ import annotations
import os
import argparse
import json
from typing import List, Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_utils import (
    ensure_dirs, load_raw_wine_data, add_quality_label, save_processed, FEATURES
)

MODEL_PATH = "models/knn_model.joblib"
METRICS_PATH = "models/metrics.json"
ACC_PLOT_PATH = "reports/figures/accuracy_vs_k.png"

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURES].values
    y = df["label"].values
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def build_pipeline(k: int = 5) -> Pipeline:
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])

def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

def tune_k(X_train, y_train, X_test, y_test, k_min=1, k_max=20):
    results = []
    best_k = None
    best_acc = -1.0
    for k in range(k_min, k_max + 1):
        model = build_pipeline(k)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append({"k": k, "accuracy": acc})
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k, pd.DataFrame(results)

def plot_accuracy_vs_k(df_results: pd.DataFrame, save_path: str = ACC_PLOT_PATH):
    plt.figure()
    plt.plot(df_results["k"], df_results["accuracy"], marker="o")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k (KNN)")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def predict_wine_quality(values: List[float], model_path: str = MODEL_PATH) -> str:
    if len(values) != len(FEATURES):
        raise ValueError(f"Se esperaban {len(FEATURES)} valores: {FEATURES}")
    model: Pipeline = joblib.load(model_path)
    pred = int(model.predict([values])[0])
    mapping = {0: "baja calidad", 1: "calidad media", 2: "alta calidad"}
    return f"Este vino probablemente sea de {mapping.get(pred, 'calidad desconocida')}"

def run_all(k_init: int = 5, k_min: int = 1, k_max: int = 20):
    ensure_dirs()
    df = load_raw_wine_data()
    print(f"Shape: {df.shape}")
    print("Columnas:", list(df.columns))
    print(df.head(3))
    df = add_quality_label(df)
    processed_path = save_processed(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_pipeline(k_init)
    model.fit(X_train, y_train)
    baseline_metrics = evaluate(model, X_test, y_test)
    best_k, df_results = tune_k(X_train, y_train, X_test, y_test, k_min, k_max)
    best_model = build_pipeline(best_k)
    best_model.fit(X_train, y_train)
    best_metrics = evaluate(best_model, X_test, y_test)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "best_k": int(best_k),
            "best_metrics": best_metrics
        }, f, indent=2)
    plot_accuracy_vs_k(df_results, ACC_PLOT_PATH)
    sample = df.sample(1, random_state=42)
    example = sample[FEATURES].iloc[0].tolist()
    print("Predicción:", predict_wine_quality(example))
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

def main():
    parser = argparse.ArgumentParser(description="Clasificador de calidad de vino tinto con KNN.")
    sub = parser.add_subparsers(dest="command")
    p_all = sub.add_parser("run-all")
    p_all.add_argument("--k-init", type=int, default=5)
    p_all.add_argument("--k-min", type=int, default=1)
    p_all.add_argument("--k-max", type=int, default=20)
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--values", nargs="+", type=float, required=True)
    args = parser.parse_args()
    if args.command == "run-all" or args.command is None:
        run_all(k_init=getattr(args, "k_init", 5),
                k_min=getattr(args, "k_min", 1),
                k_max=getattr(args, "k_max", 20))
    elif args.command == "predict":
        print(predict_wine_quality(args.values))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
