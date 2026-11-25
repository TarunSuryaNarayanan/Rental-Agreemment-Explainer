"""Compare prompting strategies on benchmark/benchmark.json.

Usage:
    python -m benchmark.run_benchmark
"""
import json
import pathlib
from rae.pipeline import analyze
from rae.prompts import STRATEGIES

DATA_PATH = pathlib.Path(__file__).parent / "benchmark.json"


def evaluate_strategy(docs: list[dict], strategy: str) -> dict:
    total, label_ok, verified = 0, 0, 0
    for doc in docs:
        res = analyze(doc["text"], strategy=strategy)
        gold_by_id = {g["clause_id"]: g for g in doc.get("gold_clauses", [])}
        for c in res.get("clauses", []):
            total += 1
            if c.get("verified"):
                verified += 1
            g = gold_by_id.get(c.get("clause_id"))
            if g and c.get("label") == g["label"]:
                label_ok += 1
    return {
        "strategy": strategy,
        "clauses": total,
        "label_accuracy": round(label_ok / max(1, total), 3),
        "quote_faithfulness": round(verified / max(1, total), 3),
    }


def main():
    docs = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"{'Strategy':<18} {'Clauses':>8} {'Label Acc':>10} {'Faithfulness':>13}")
    print("-" * 54)
    for strat in STRATEGIES:
        m = evaluate_strategy(docs, strat)
        print(f"{m['strategy']:<18} {m['clauses']:>8} {m['label_accuracy']:>10.1%} {m['quote_faithfulness']:>13.1%}")


if __name__ == "__main__":
    main()
