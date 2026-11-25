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
# Directory that holds the pickled pipeline – resolved relative to repo root
# by default, and overridable via RISK_MODEL_DIR.
# ----------------------------------------------------------------------
_DEFAULT_MODEL_DIR = pathlib.Path(__file__).resolve().parent.parent / "models" / "risk-clf-sklearn"
MODEL_DIR = pathlib.Path(os.getenv("RISK_MODEL_DIR", str(_DEFAULT_MODEL_DIR)))
PIPELINE_PATH = MODEL_DIR / "pipeline.pkl"

_pipeline = None
if PIPELINE_PATH.is_file():
    try:
        with open(PIPELINE_PATH, "rb") as f:
            _pipeline = pickle.load(f)  # type: ignore[var-annotated]
    except Exception:
        _pipeline = None


def _heuristic_risk(clause: str) -> str:
    low = clause.lower()
    if any(k in low for k in ("sole discretion", "without prior notice", "without notice", "forfeited in full", "entire security deposit")):
        return "High"
    if any(k in low for k in ("curfew", "structural repairs", "mandatory painting", "6 months advance", "lock-in of 12")):
        return "Medium"
    return "Standard"


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
    if _pipeline is None:
        return _heuristic_risk(clause)
    # The pipeline expects an iterable of strings.
    try:
        prediction = _pipeline.predict([clause])
        return str(prediction[0])
    except Exception:  # safety net – never crash the checklist
        return _heuristic_risk(clause)
