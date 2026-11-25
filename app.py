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



BUILTIN_SAMPLES = {
    "sample_bengaluru_balanced.txt": {
        "title": "Bengaluru Balanced 11-Month Lease",
        "content": (
            "RESIDENTIAL RENTAL AGREEMENT\n\n"
            "This Rental Agreement is made between the Lessor (Landlord) and the Lessee (Tenant) "
            "for the residential premises situated at Indiranagar, Bengaluru.\n\n"
            "1. MONTHLY RENT: The Lessee shall pay a monthly rent of Rs. 28,000 on or before the 5th day of each calendar month.\n\n"
            "2. SECURITY DEPOSIT: The Lessee has paid a refundable Security Deposit of Rs. 1,00,000, "
            "to be refunded within 30 days of vacating after deducting only actual damages beyond normal wear and tear.\n\n"
            "3. LOCK-IN PERIOD: Both parties agree to a mandatory lock-in period of 3 months from the commencement date.\n\n"
            "4. NOTICE PERIOD: After the lock-in period, either party may terminate this agreement by giving 1 month prior written notice.\n\n"
            "5. RENT INCREASE: Upon renewal after 11 months, the monthly rent may be enhanced by 5% with mutual written consent.\n\n"
            "6. MAINTENANCE AND REPAIRS: Minor day-to-day repairs shall be borne by the Lessee, while structural repairs shall be handled by the Lessor.\n\n"
            "7. LANDLORD ENTRY AND INSPECTION: The Lessor may inspect the premises with at least 24 hours prior written notice at a mutually convenient time.\n\n"
            "8. SUBLETTING: The Lessee shall not sublet or assign the premises without prior written consent of the Lessor.\n\n"
            "9. UTILITIES AND OTHER CHARGES: Electricity and water bills shall be paid by the Lessee as per actual meter readings.\n\n"
            "10. POLICE VERIFICATION: The Lessee shall complete police verification at the local police station within 15 days of occupancy.\n\n"
            "11. REGISTRATION AND STAMP DUTY: Stamp duty and registration charges shall be shared equally by both parties."
        ),
    },
    "sample_strict_landlord.txt": {
        "title": "Strict / Landlord-Favourable Lease (Risky)",
        "content": (
            "RESIDENTIAL LEASE AGREEMENT\n\n"
            "This Lease Agreement is executed between the Lessor and the Lessee for residential premises in Mumbai.\n\n"
            "1. MONTHLY RENT: The Lessee shall pay monthly rent of Rs. [Amount] on or before the [Day] of each month.\n\n"
            "2. SECURITY DEPOSIT: The Lessee shall pay Rs. 2,00,000 as Security Deposit. If the Lessee vacates before 12 months, "
            "the entire Security Deposit shall stand forfeited in full and the Lessor may claim extra losses at his sole discretion.\n\n"
            "3. LOCK-IN PERIOD: The Lessee agrees to a mandatory lock-in of 9 months during which early exit is prohibited.\n\n"
            "4. LANDLORD ENTRY AND INSPECTION: The Lessor may enter the premises at any time without notice for inspection or showing the flat.\n\n"
            "5. MAINTENANCE AND REPAIRS: The Lessee shall bear all interior, exterior, and structural repairs plus mandatory painting deductions.\n\n"
            "6. USE OF PREMISES: Strictly residential use; no visitors are permitted after 9 PM curfew.\n\n"
            "7. UTILITIES AND OTHER CHARGES: All electricity, water, and society charges shall be paid by the Lessee."
        ),
    },
}


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
    if not samples:
        for fname, info in BUILTIN_SAMPLES.items():
            samples.append({"id": fname, "title": info["title"], "filename": fname})
    return jsonify(samples=samples)


@app.get("/api/sample/<filename>")
def get_sample_content(filename):
    if not filename.endswith(".txt") or "/" in filename or "\\" in filename:
        return jsonify(error="Invalid filename"), 400
    target = AGREEMENTS_DIR / filename
    if target.is_file():
        return jsonify(filename=filename, content=target.read_text(encoding="utf-8"))
    if filename in BUILTIN_SAMPLES:
        return jsonify(filename=filename, content=BUILTIN_SAMPLES[filename]["content"])
    return jsonify(error="File not found"), 404


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


