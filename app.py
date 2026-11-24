import logging
import os
from pathlib import Path
import re

# Ensure local offline models and NLP features are enabled by default
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("USE_NLP", "1")
os.environ.setdefault("USE_LOCAL_MODELS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rae.app")

from flask import Flask, jsonify, render_template, request
from rae.pipeline import analyze

app = Flask(__name__)
STRATEGIES = {"local_model", "zero_shot", "persona", "template", "few_shot", "cot"}
AGREEMENTS_DIR = Path(__file__).resolve().parent / "agreements"


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.get("/")
def index():
    return render_template("index.html", strategies=sorted(STRATEGIES))


@app.get("/api/model-status")
def model_status():
    status = {}
    # 1. Explainer
    try:
        from rae.nlp.explainer import get_explainer
        m, t = get_explainer()
        status["explainer"] = "loaded" if m is not None else "unavailable"
    except Exception as e:
        status["explainer"] = f"error: {e}"

    # 2. Risk classifier
    try:
        from rae.risk_classifier import predict_risk
        r = predict_risk("sample clause")
        status["risk_classifier"] = "loaded" if r else "unavailable"
    except Exception as e:
        status["risk_classifier"] = f"error: {e}"

    # 3. Classifier
    try:
        from rae.nlp.classifier import get_classifier
        c = get_classifier()
        status["clause_classifier"] = "loaded" if c is not None else "unavailable"
    except Exception as e:
        status["clause_classifier"] = f"error: {e}"

    all_ready = all(v == "loaded" for v in status.values())
    return jsonify(ready=all_ready, models=status)



@app.get("/api/samples")
def get_samples():
    samples = []
    if AGREEMENTS_DIR.exists():
        for path in sorted(AGREEMENTS_DIR.glob("*.txt")):
            clean_name = re.sub(r"^agreement_\d+_", "", path.stem).replace("_", " ").title()
            samples.append({
                "id": path.name,
                "title": clean_name,
                "filename": path.name,
            })
    return jsonify(samples=samples)


@app.get("/api/sample/<filename>")
def get_sample_content(filename):
    if not filename.endswith(".txt") or "/" in filename or "\\" in filename:
        return jsonify(error="Invalid filename"), 400
    target = AGREEMENTS_DIR / filename
    if not target.is_file():
        return jsonify(error="File not found"), 404
    return jsonify(filename=filename, content=target.read_text(encoding="utf-8"))


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    strategy = data.get("strategy", "local_model")
    if not text:
        return jsonify(error="Please paste the agreement text."), 400
    if strategy not in STRATEGIES:
        strategy = "local_model"
    return jsonify(analyze(text, strategy))


if __name__ == "__main__":
    logger.info("Warming up local NLP models (Flan-T5, LegalBERT, Risk Classifier)...")
    try:
        from rae.nlp.explainer import get_explainer
        em, et = get_explainer()
        logger.info("Explainer model status: %s", "READY" if em is not None else "FAILED")
    except Exception as e:
        logger.warning("Explainer warmup error: %s", e)

    try:
        from rae.nlp.classifier import get_classifier
        clf = get_classifier()
        logger.info("Clause classifier status: %s", "READY" if clf is not None else "FAILED")
    except Exception as e:
        logger.warning("Classifier warmup error: %s", e)

    try:
        from rae.risk_classifier import predict_risk
        rk = predict_risk("sample")
        logger.info("Risk classifier status: %s", "READY" if rk else "FAILED")
    except Exception as e:
        logger.warning("Risk classifier warmup error: %s", e)

    app.run(debug=True)


