"""Insider (SEC Form 4) trading signal — opportunistic cluster buys."""
from .ingest import InsiderResult, fetch_insider_trades
from .model import InsiderTrade, classify
from .score import insider_scores

__all__ = ["InsiderResult", "fetch_insider_trades", "InsiderTrade", "classify",
           "insider_scores"]
