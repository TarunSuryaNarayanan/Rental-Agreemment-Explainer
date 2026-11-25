import json
import pathlib
from flask import Flask, jsonify, render_template, request

from rae.llm import active_backend
from rae.pipeline import analyze, compare_strategies
from rae.prompts import STRATEGIES, build_prompt

app = Flask(__name__)
VALID_STRATEGIES = set(STRATEGIES)
BENCHMARK_PATH = pathlib.Path(__file__).resolve().parent / "benchmark" / "benchmark.json"


@app.get("/")
def index():
    return render_template("index.html", strategies=STRATEGIES, backend=active_backend())


@app.get("/api/sample")
def api_sample():
    if BENCHMARK_PATH.is_file():
        samples = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        return jsonify(samples[0])
    return jsonify(error="No sample agreement found."), 404


@app.post("/api/prompt-preview")
def api_prompt_preview():
    data = request.get_json(force=True)
    strategy = data.get("strategy", "few_shot")
    clause = (data.get("clause") or "").strip() or (
        "3. LOCK-IN PERIOD: Both parties agree to a mandatory lock-in period of 6 months. "
        "If the Lessee vacates early, the entire Security Deposit shall stand forfeited."
    )
    if strategy not in VALID_STRATEGIES:
        return jsonify(error="Unknown strategy"), 400
    p = build_prompt(clause, strategy=strategy)
    return jsonify(strategy=strategy, system=p.system, user=p.user)


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    strategy = data.get("strategy", "few_shot")
    if not text:
        return jsonify(error="Please paste the agreement text."), 400
    if strategy not in VALID_STRATEGIES:
        return jsonify(error="Unknown strategy"), 400
    return jsonify(analyze(text, strategy))


@app.post("/api/compare")
def api_compare():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Please paste the agreement text."), 400
    return jsonify(compare_strategies(text))


if __name__ == "__main__":
    app.run(debug=True)
