import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List
from zipfile import BadZipFile

import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextPreprocessor:
    """Reusable text normalization helper used by all classification tasks."""

    language: str = "english"

    def __post_init__(self) -> None:
        self._stop_words = set(stopwords.words(self.language))
        self._lemmatizer = WordNetLemmatizer()

    def clean_and_tokenize(self, raw_text: str) -> List[str]:
        # Keep only alphabets and spaces to reduce noisy features.
        normalized = re.sub(r"[^a-zA-Z\s]", " ", raw_text.lower())
        tokens = word_tokenize(normalized)

        filtered = []
        for token in tokens:
            if token in self._stop_words or len(token) < 2:
                continue
            filtered.append(self._lemmatizer.lemmatize(token))
        return filtered

    def preprocess_text(self, raw_text: str) -> str:
        return " ".join(self.clean_and_tokenize(raw_text))

    def preprocess_corpus(self, corpus: Iterable[str]) -> List[str]:
        return [self.preprocess_text(text) for text in corpus]


def ensure_nltk_assets() -> None:
    """Download required NLTK resources once if they are missing or corrupted."""
    local_nltk_dir = Path.home() / ".nlp_intelligence_system" / "nltk_data"
    local_nltk_dir.mkdir(parents=True, exist_ok=True)

    # Prefer project-local data to avoid failures caused by broken global NLTK caches.
    if str(local_nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(local_nltk_dir))

    def _check_punkt() -> None:
        nltk.data.find("tokenizers/punkt")

    def _check_punkt_tab() -> None:
        nltk.data.find("tokenizers/punkt_tab")

    def _check_stopwords() -> None:
        stopwords.words("english")

    def _check_wordnet() -> None:
        wordnet.ensure_loaded()

    def _check_twitter_samples() -> None:
        twitter_samples.strings("positive_tweets.json")

    required_assets: list[tuple[str, Callable[[], None]]] = [
        ("punkt", _check_punkt),
        ("punkt_tab", _check_punkt_tab),
        ("stopwords", _check_stopwords),
        ("wordnet", _check_wordnet),
        ("twitter_samples", _check_twitter_samples),
    ]

    for package_name, validator in required_assets:
        try:
            validator()
        except (LookupError, OSError, BadZipFile):
            # Remove potentially broken package cache before retrying download.
            for stale_path in (
                local_nltk_dir / f"{package_name}.zip",
                local_nltk_dir / "corpora" / package_name,
                local_nltk_dir / "tokenizers" / package_name,
            ):
                if stale_path.is_file():
                    stale_path.unlink(missing_ok=True)
                elif stale_path.is_dir():
                    shutil.rmtree(stale_path, ignore_errors=True)

            nltk.download(
                package_name,
                download_dir=str(local_nltk_dir),
                quiet=True,
                raise_on_error=True,
                force=True,
            )
            validator()


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
