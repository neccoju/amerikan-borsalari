import pandas as pd

from usbot.config import get_secrets
from usbot.news import fetch_news, news_scores, score_sentiment
from usbot.news.model import NewsItem, dedup


def test_dedup_removes_duplicate_headlines():
    items = [
        NewsItem("AAPL", "Apple beats earnings"),
        NewsItem("AAPL", "Apple  beats   earnings"),  # whitespace variant
        NewsItem("AAPL", "Apple launches new product"),
    ]
    out = dedup(items)
    assert len(out) == 2


def test_sentiment_positive_vs_negative():
    # Wording chosen to be unambiguous across VADER / FinBERT / fallback lexicon.
    pos = NewsItem("X", "Strong gains and record profit; stock wins, outperform")
    neg = NewsItem("Y", "Weak results, big loss, lawsuit and downgrade; fraud probe")
    score_sentiment([pos, neg])
    assert pos.sentiment > neg.sentiment
    assert pos.sentiment > 0
    assert neg.sentiment < 0


def test_sentiment_category_classification():
    it = NewsItem("X", "Analyst upgrade: raises price target on strong demand")
    score_sentiment([it])
    assert it.category in ("analyst", "general")


def test_news_scores_neutral_when_no_news():
    series = news_scores({}, ["AAPL", "MSFT"])
    assert (series == 50.0).all()


def test_news_scores_positive_above_negative():
    pos = NewsItem("AAA", "record profit, surge, beats, upgrade")
    neg = NewsItem("BBB", "plunge, lawsuit, downgrade, loss")
    score_sentiment([pos, neg])
    series = news_scores({"AAA": [pos], "BBB": [neg]}, ["AAA", "BBB", "CCC"])
    assert series["AAA"] > series["BBB"]
    assert series["CCC"] == 50.0  # no news -> neutral


def test_fetch_news_graceful_without_keys(monkeypatch):
    for k in ("FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    secrets = get_secrets(dotenv_path="/nonexistent.env")
    res = fetch_news(["AAPL"], secrets)
    assert res.enabled is False
    assert res.total == 0
    assert "FINNHUB" in res.skip_reason or "ALPHA" in res.skip_reason
