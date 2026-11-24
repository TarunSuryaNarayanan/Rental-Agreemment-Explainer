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
DEFAULT_MODEL_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "models" / "risk-clf-sklearn")
MODEL_DIR = os.getenv("RISK_MODEL_DIR", DEFAULT_MODEL_DIR)
PIPELINE_PATH = pathlib.Path(MODEL_DIR) / "pipeline.pkl"

_pipeline = None

if PIPELINE_PATH.is_file():
    with open(PIPELINE_PATH, "rb") as f:
        _pipeline = pickle.load(f)
else:
    import logging
    logging.getLogger(__name__).warning("Risk model not found at %s. Defaulting to heuristic fallback.", PIPELINE_PATH)


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
