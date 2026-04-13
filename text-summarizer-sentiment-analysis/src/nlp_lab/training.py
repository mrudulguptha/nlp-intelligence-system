from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from .data_loader import load_sentiment_dataset, load_spam_dataset
from .models import ExperimentResult, run_sentiment_comparison, train_spam_detector
from .preprocessing import TextPreprocessor, ensure_nltk_assets


@dataclass
class TrainedArtifacts:
    sentiment_results: list[ExperimentResult]
    sentiment_models: dict[str, Pipeline]
    spam_result: ExperimentResult
    spam_model: Pipeline
    sentiment_dataset_note: str
    spam_dataset_note: str
    text_preprocessor: TextPreprocessor


def build_all_models() -> TrainedArtifacts:
    ensure_nltk_assets()
    preprocessor = TextPreprocessor()

    sentiment_texts, sentiment_labels, sentiment_note = load_sentiment_dataset()
    sentiment_results, sentiment_models = run_sentiment_comparison(
        sentiment_texts, sentiment_labels, preprocessor
    )

    spam_texts, spam_labels, spam_note = load_spam_dataset()
    spam_result, spam_model = train_spam_detector(spam_texts, spam_labels, preprocessor)

    return TrainedArtifacts(
        sentiment_results=sentiment_results,
        sentiment_models=sentiment_models,
        spam_result=spam_result,
        spam_model=spam_model,
        sentiment_dataset_note=sentiment_note,
        spam_dataset_note=spam_note,
        text_preprocessor=preprocessor,
    )
