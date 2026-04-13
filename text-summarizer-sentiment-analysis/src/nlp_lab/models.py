from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocessing import TextPreprocessor, build_tfidf_vectorizer


@dataclass
class ExperimentResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion: List[List[int]]


def _build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", classifier),
        ]
    )


def evaluate_classifier(
    model_name: str,
    classifier,
    texts: List[str],
    labels: List[int],
    preprocessor: TextPreprocessor,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ExperimentResult, Pipeline]:
    processed_texts = preprocessor.preprocess_corpus(texts)

    x_train, x_test, y_train, y_test = train_test_split(
        processed_texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model_pipeline = _build_pipeline(classifier)
    model_pipeline.fit(x_train, y_train)

    predictions = model_pipeline.predict(x_test)

    result = ExperimentResult(
        model_name=model_name,
        accuracy=accuracy_score(y_test, predictions),
        precision=precision_score(y_test, predictions, zero_division=0),
        recall=recall_score(y_test, predictions, zero_division=0),
        f1_score=f1_score(y_test, predictions, zero_division=0),
        confusion=confusion_matrix(y_test, predictions).tolist(),
    )

    return result, model_pipeline


def run_sentiment_comparison(
    texts: List[str], labels: List[int], preprocessor: TextPreprocessor
) -> Tuple[List[ExperimentResult], Dict[str, Pipeline]]:
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    outcomes = []
    fitted = {}
    for model_name, classifier in candidate_models.items():
        metrics, trained_model = evaluate_classifier(
            model_name=model_name,
            classifier=classifier,
            texts=texts,
            labels=labels,
            preprocessor=preprocessor,
        )
        outcomes.append(metrics)
        fitted[model_name] = trained_model

    outcomes.sort(key=lambda item: item.f1_score, reverse=True)
    return outcomes, fitted


def train_spam_detector(
    texts: List[str], labels: List[int], preprocessor: TextPreprocessor
) -> Tuple[ExperimentResult, Pipeline]:
    return evaluate_classifier(
        model_name="Multinomial Naive Bayes",
        classifier=MultinomialNB(),
        texts=texts,
        labels=labels,
        preprocessor=preprocessor,
    )
