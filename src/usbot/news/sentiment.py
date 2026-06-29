"""Sentiment scoring + lightweight impact classification.

Default model: VADER (lexicon-based, keyless, light). Optional FinBERT when
``USBOT_SENTIMENT=finbert`` and transformers is installed. If neither VADER nor
FinBERT is available, falls back to a tiny built-in lexicon so the pipeline
never hard-fails.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from ..utils.logging import get_logger
from .model import NewsItem

log = get_logger(__name__)

# Keyword buckets for coarse impact category tagging.
_CATEGORY_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue", "quarter", "guidance", "beats", "misses"],
    "analyst": ["upgrade", "downgrade", "price target", "initiates", "rating", "analyst"],
    "legal": ["lawsuit", "investigation", "sec ", "fine", "settlement", "probe", "antitrust"],
    "product": ["launch", "unveils", "ai ", "product", "partnership", "contract", "patent"],
    "macro": ["fed", "inflation", "rates", "tariff", "jobs", "cpi", "recession"],
}

# Minimal fallback lexicon (only used if VADER & FinBERT both unavailable).
_POS = {"beat", "beats", "surge", "soar", "record", "upgrade", "growth", "strong",
        "wins", "raises", "outperform", "profit", "rally", "gains"}
_NEG = {"miss", "misses", "plunge", "drop", "downgrade", "lawsuit", "probe", "cut",
        "weak", "loss", "falls", "warning", "recall", "fraud", "bankruptcy"}


@dataclass
class SentimentResult:
    model: str
    scored: int


@lru_cache(maxsize=1)
def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception as exc:  # noqa: BLE001
        log.info("VADER unavailable (%s); using fallback lexicon", exc)
        return None


@lru_cache(maxsize=1)
def _finbert():
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as exc:  # noqa: BLE001
        log.warning("FinBERT unavailable (%s); falling back to VADER", exc)
        return None


def _lexicon_score(text: str) -> float:
    toks = set(text.lower().replace(",", " ").replace(".", " ").split())
    pos = len(toks & _POS)
    neg = len(toks & _NEG)
    if pos == neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _classify(text: str) -> str:
    low = text.lower()
    for cat, kws in _CATEGORY_KEYWORDS.items():
        if any(k in low for k in kws):
            return cat
    return "general"


def _label(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def score_sentiment(items: list[NewsItem], model: str | None = None) -> SentimentResult:
    """Annotate each NewsItem in place with sentiment, label and category."""
    model = (model or os.environ.get("USBOT_SENTIMENT", "vader")).lower()

    finbert = _finbert() if model == "finbert" else None
    vader = _vader() if finbert is None else None
    used = "finbert" if finbert else ("vader" if vader else "lexicon")

    for it in items:
        text = it.text
        if finbert is not None:
            try:
                out = finbert(text[:512])[0]
                lab = out["label"].lower()
                sc = out["score"]
                it.sentiment = sc if lab == "positive" else (-sc if lab == "negative" else 0.0)
            except Exception:  # noqa: BLE001
                it.sentiment = _lexicon_score(text)
        elif vader is not None:
            it.sentiment = vader.polarity_scores(text)["compound"]
        else:
            it.sentiment = _lexicon_score(text)
        it.label = _label(it.sentiment)
        it.category = _classify(text)

    return SentimentResult(model=used, scored=len(items))
