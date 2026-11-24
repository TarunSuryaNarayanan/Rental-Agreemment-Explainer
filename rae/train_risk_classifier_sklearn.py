# rae/train_risk_classifier_sklearn.py
"""
Train a lightweight scikit‑learn classifier for risk level detection.
It uses TF‑IDF vectorisation over the clause text and a LogisticRegression
classifier. The resulting model (and vectoriser) are pickled to the output
directory.

Usage:
    python -m rae.train_risk_classifier_sklearn \
        --train-path ./data/risk_training.csv \
        --output-dir ./models/risk-clf-sklearn
"""

import argparse
import os
import pathlib
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Train a simple TF‑IDF + LogisticRegression risk classifier")
    parser.add_argument("--train-path", type=str, required=True, help="CSV with clause_text,risk_label")
    parser.add_argument("--output-dir", type=str, default="./models/risk-clf-sklearn", help="Where to store the model")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path)
    df = df.dropna(subset=["clause_text", "risk_label"]).copy()
    df["risk_label"] = df["risk_label"].str.strip()

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=2000)),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")),
    ])

    pipeline.fit(df["clause_text"], df["risk_label"])

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = pathlib.Path(args.output_dir) / "pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
