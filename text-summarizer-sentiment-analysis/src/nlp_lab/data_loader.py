from __future__ import annotations

from typing import List, Tuple
from nltk.corpus import twitter_samples

SPAM_SAMPLES = [
    "Congratulations! You won a free vacation. Reply WIN now.",
    "Urgent: your account was selected for a cash prize, call now.",
    "Claim your reward today by clicking this limited-time offer.",
    "Free entry in our contest. Text YES to participate.",
    "Get rich quickly with this guaranteed investment opportunity.",
    "Hi, are we still meeting for the project discussion tomorrow?",
    "Please send me the report before noon.",
    "The class starts at 10 AM in room 204.",
    "Can you review my assignment draft tonight?",
    "Lunch at 1 PM works for me, see you then.",
]

SPAM_LABELS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def load_sentiment_dataset() -> Tuple[List[str], List[int], str]:
    """Load NLTK Twitter samples for a balanced positive/negative dataset."""
    positive_tweets = twitter_samples.strings("positive_tweets.json")
    negative_tweets = twitter_samples.strings("negative_tweets.json")

    texts = positive_tweets + negative_tweets
    labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

    description = (
        "Sentiment dataset: NLTK twitter_samples corpus containing 5,000 positive "
        "and 5,000 negative tweets."
    )
    return texts, labels, description


def load_spam_dataset() -> Tuple[List[str], List[int], str]:
    """Load a compact built-in SMS spam sample dataset."""
    description = (
        "Spam dataset: Compact in-repository SMS sample set with balanced ham/spam "
        "messages for lightweight training."
    )
    return SPAM_SAMPLES, SPAM_LABELS, description
