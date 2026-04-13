from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from model import NLPModelService


app = Flask(__name__)
nlp_service = NLPModelService()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON payload."}), 400

    text = (payload.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter text before analysis."}), 400

    try:
        result = nlp_service.predict(text)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Prediction failed. Please try again shortly."}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
