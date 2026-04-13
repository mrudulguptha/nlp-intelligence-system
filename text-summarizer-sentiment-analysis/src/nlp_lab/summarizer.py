from collections import Counter
from typing import List

from nltk.tokenize import sent_tokenize, word_tokenize


STOP_WORDS = {
    "a", "an", "the", "is", "are", "and", "or", "for", "to", "of", "in", "on",
    "with", "that", "this", "it", "as", "at", "by", "be", "from", "was", "were",
}


def extractive_frequency_summary(document: str, sentence_limit: int = 3) -> str:
    """Simple extractive summarizer based on token frequency and sentence scoring."""
    sentences = sent_tokenize(document)
    if not sentences:
        return ""

    cleaned_tokens = []
    for token in word_tokenize(document.lower()):
        if token.isalpha() and token not in STOP_WORDS:
            cleaned_tokens.append(token)

    if not cleaned_tokens:
        return ""

    frequencies = Counter(cleaned_tokens)

    sentence_scores = []
    for sentence in sentences:
        sentence_tokens = [tok.lower() for tok in word_tokenize(sentence) if tok.isalpha()]
        if not sentence_tokens:
            continue

        score = sum(frequencies.get(token, 0) for token in sentence_tokens)
        normalized_score = score / len(sentence_tokens)
        sentence_scores.append((sentence, normalized_score))

    if not sentence_scores:
        return ""

    ranked = sorted(sentence_scores, key=lambda item: item[1], reverse=True)
    selected_sentences = [item[0] for item in ranked[:sentence_limit]]

    # Keep the original narrative flow as much as possible.
    ordered_summary = [sentence for sentence in sentences if sentence in set(selected_sentences)]
    return " ".join(ordered_summary)
