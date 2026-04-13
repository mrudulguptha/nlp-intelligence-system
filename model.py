from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_lab.summarizer import extractive_frequency_summary  # noqa: E402
from nlp_lab.training import TrainedArtifacts, build_all_models  # noqa: E402


class NLPModelService:
    """Single-instance NLP inference service for Flask routes."""

    def __init__(self) -> None:
        self.artifacts: TrainedArtifacts = build_all_models()

    def predict(self, raw_text: str) -> dict[str, str]:
        cleaned_text = self.artifacts.text_preprocessor.preprocess_text(raw_text)
        if not cleaned_text.strip():
            raise ValueError("Text is empty after preprocessing. Please provide richer input.")

        best_sentiment_name = self.artifacts.sentiment_results[0].model_name
        best_sentiment_model = self.artifacts.sentiment_models[best_sentiment_name]

        sentiment_pred = int(best_sentiment_model.predict([cleaned_text])[0])
        spam_pred = int(self.artifacts.spam_model.predict([cleaned_text])[0])
        summary = extractive_frequency_summary(raw_text, sentence_limit=3)

        return {
            "sentiment": "Positive" if sentiment_pred == 1 else "Negative",
            "spam": "Spam" if spam_pred == 1 else "Not Spam",
            "summary": summary if summary else "Summary could not be generated for this text.",
            "sentiment_model": best_sentiment_name,
        }
