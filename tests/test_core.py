import json
import os
import pathlib

os.environ["RAE_MOCK"] = "1"

from rae.checklist import check_missing
from rae.guardrails import is_rental_agreement, scrub_verdicts
from rae.pipeline import analyze
from rae.splitter import split_clauses
from rae.verifier import quote_in_text

DOC = json.loads((pathlib.Path(__file__).parent.parent / "benchmark" / "benchmark.json").read_text())[0]["text"]


def test_split():
    assert len(split_clauses(DOC)) >= 7


def test_verifier():
    assert quote_in_text("lock-in period of 6 months", DOC)
    assert not quote_in_text("tenant may keep pets freely", DOC)


def test_guardrails():
    assert is_rental_agreement(DOC)
    assert not is_rental_agreement("Recipe for dosa: rice and lentils. " * 20)
    assert "verdict removed" in scrub_verdicts("This clause is illegal.")


def test_checklist():
    assert "Lock-in period" in check_missing(DOC)["present"]


def test_pipeline_mock():
    out = analyze(DOC)
    assert out["stats"]["n_clauses"] >= 7
