"""Interactive dark-theme dashboard (site/index.html).

A finance-terminal style page (KPI cards, treemap heatmap, portfolio-vs-benchmark
curves, sector rotation / RRG, smart-money rotation proxy, holdings & signals,
monthly LLM review). Inspired by the UX of Finviz, Portfolio Visualizer, Koyfin,
TradingView, StockCharts and Yahoo Finance — own data and code. Generated
alongside the email report; never breaks it (caller wraps the build).
"""
from __future__ import annotations

from .builder import DashboardData, build_dashboard

__all__ = ["DashboardData", "build_dashboard"]
