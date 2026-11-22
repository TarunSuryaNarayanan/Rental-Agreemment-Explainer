import os

from rae.nlp.classifier import classify_clause
from rae.nlp.entailment import check_missing_nli


def test_nlp_classifier_fallback():
    # In mock or without weights, gracefully falls back to provided fallback function
    dummy_fallback = lambda txt: "deposit" if "deposit" in txt else "other"
    clause_type, conf = classify_clause("Pay refundable deposit of 50000", fallback_fn=dummy_fallback)
    assert clause_type == "deposit"
    assert conf > 0.0


def test_nlp_entailment_fallback():
    dummy_fallback = lambda doc: {"present": ["Lock-in period"], "missing": ["Notice period"]}
    res = check_missing_nli("Some agreement text", fallback_fn=dummy_fallback)
    assert "Lock-in period" in res["present"]
    assert "Notice period" in res["missing"]
