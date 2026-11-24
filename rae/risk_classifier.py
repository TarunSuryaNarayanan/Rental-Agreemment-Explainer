# rae/risk_classifier.py
"""
Model‑based risk classifier.

The scikit‑learn pipeline (TF‑IDF + LogisticRegression) is trained with
`rae/train_risk_classifier_sklearn.py` and saved to
`models/risk-clf-sklearn/pipeline.pkl`.

`predict_risk(clause: str) -> str` returns one of:
Standard, Low, Medium, High, Critical
"""

import os
import pathlib
import pickle

# ----------------------------------------------------------------------
# Directory that holds the pickled pipeline – can be overridden via env var.
# ----------------------------------------------------------------------
MODEL_DIR = os.getenv("RISK_MODEL_DIR", "./models/risk-clf-sklearn")
PIPELINE_PATH = pathlib.Path(MODEL_DIR) / "pipeline.pkl"

if not PIPELINE_PATH.is_file():
    raise FileNotFoundError(
        f"Risk model not found at {PIPELINE_PATH}. "
        "Run `python -m rae.train_risk_classifier_sklearn` to create it."
    )

# Load once at import time – cheap for the tiny model.
with open(PIPELINE_PATH, "rb") as f:
    _pipeline = pickle.load(f)  # type: ignore[var‑annotated]

def predict_risk(clause: str) -> str:
    """
    Predict the risk level for a single clause.

    Parameters
    ----------
    clause: str
        Full clause text (or the extracted quote). Empty strings default to
        "Standard".

    Returns
    -------
    str
        One of "Standard", "Low", "Medium", "High", "Critical".
    """
    if not clause:
        return "Standard"
    # The pipeline expects an iterable of strings.
    try:
        prediction = _pipeline.predict([clause])
        return str(prediction[0])
    except Exception:  # safety net – never crash the checklist
        return "Standard"
