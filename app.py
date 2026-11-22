from flask import Flask, jsonify, render_template, request

from rae.pipeline import analyze

app = Flask(__name__)
STRATEGIES = {"zero_shot", "persona", "template", "few_shot", "cot"}


@app.get("/")
def index():
    return render_template("index.html", strategies=sorted(STRATEGIES))


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    strategy = data.get("strategy", "few_shot")
    if not text:
        return jsonify(error="Please paste the agreement text."), 400
    if strategy not in STRATEGIES:
        return jsonify(error="Unknown strategy"), 400
    return jsonify(analyze(text, strategy))


if __name__ == "__main__":
    app.run(debug=True)
