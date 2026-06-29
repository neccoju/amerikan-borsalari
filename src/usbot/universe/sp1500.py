"""S&P 400 (MidCap) + S&P 600 (SmallCap) constituents for a broader universe.

Combined with the S&P 500 these give the S&P 1500 — a liquid, quality universe
that deliberately excludes microcap/penny junk (matching the design's "exclude
illiquid penny stocks" rule). Dynamic fetch from Wikipedia with a small static
fallback so the universe is always available without network.
"""
from __future__ import annotations

from ..utils.logging import get_logger

log = get_logger(__name__)

# Small static fallbacks (liquid mid/small caps) used only if dynamic fetch fails.
SP400_SEED = [
    "JBL", "DOCU", "WSM", "BURL", "RPM", "CSL", "EME", "WING", "DT", "PSTG",
    "FIX", "MANH", "CW", "ATR", "EXP", "THC", "GME", "AIT", "OLED", "SAIA",
]
SP600_SEED = [
    "SPSC", "MGEE", "ABCB", "BMI", "CALM", "AWR", "SHOO", "EXLS", "PRGS", "ENSG",
    "BOOT", "AEIS", "IDCC", "CARG", "FORM", "KTB", "MGY", "AWI", "PLXS", "HALO",
]


def _fetch_wiki_table(url: str, symbol_col: str = "Symbol") -> list[str]:
    import pandas as pd

    tables = pd.read_html(url)
    for tbl in tables:
        cols = {str(c).strip(): c for c in tbl.columns}
        col = cols.get(symbol_col) or cols.get("Ticker symbol") or cols.get("Ticker")
        if col is not None:
            syms = (tbl[col].astype(str).str.replace(".", "-", regex=False)
                    .str.strip().str.upper().tolist())
            return [s for s in syms if s and s != "NAN"]
    return []


def get_sp400(dynamic: bool = True) -> list[str]:
    if dynamic:
        try:
            syms = _fetch_wiki_table(
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
            if len(syms) >= 300:
                log.info("Loaded %d S&P 400 symbols dynamically", len(syms))
                return syms
        except Exception as exc:  # noqa: BLE001
            log.warning("S&P 400 fetch failed (%s); using seed", exc)
    return list(SP400_SEED)


def get_sp600(dynamic: bool = True) -> list[str]:
    if dynamic:
        try:
            syms = _fetch_wiki_table(
                "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
            if len(syms) >= 400:
                log.info("Loaded %d S&P 600 symbols dynamically", len(syms))
                return syms
        except Exception as exc:  # noqa: BLE001
            log.warning("S&P 600 fetch failed (%s); using seed", exc)
    return list(SP600_SEED)
