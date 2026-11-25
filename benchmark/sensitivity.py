"""Measure paraphrase sensitivity across INSTRUCTION_VARIANTS.

Usage:
    python -m benchmark.sensitivity
"""
import json
import pathlib
from rae.pipeline import analyze
from rae.prompts import INSTRUCTION_VARIANTS

DATA_PATH = pathlib.Path(__file__).parent / "benchmark.json"


def measure_sensitivity(doc_text: str, strategy: str = "few_shot") -> dict:
    runs = [analyze(doc_text, strategy=strategy, variant=v) for v in range(len(INSTRUCTION_VARIANTS))]
    n_clauses = len(runs[0]["clauses"])
    unanimous = 0
    for idx in range(n_clauses):
        labels = {r["clauses"][idx]["label"] for r in runs if idx < len(r["clauses"])}
        if len(labels) == 1:
            unanimous += 1
    return {
        "strategy": strategy,
        "variants_tested": len(INSTRUCTION_VARIANTS),
        "clauses": n_clauses,
        "unanimous_agreement": round(unanimous / max(1, n_clauses), 3),
    }


def main():
    docs = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for doc in docs:
        res = measure_sensitivity(doc["text"])
        print(f"{doc['id']}: {res['unanimous_agreement']:.1%} unanimous across {res['variants_tested']} paraphrases ({res['clauses']} clauses)")


if __name__ == "__main__":
    main()
