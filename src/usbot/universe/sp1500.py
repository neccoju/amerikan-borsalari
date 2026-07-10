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


def _fetch_wiki_table(url: str, symbol_col: str = "Symbol") -> tuple[list[str], dict[str, str]]:
    """(symbols, {symbol: GICS sector}) from the constituents table at ``url``."""
    from .wiki import read_wikipedia_tables

    tables = read_wikipedia_tables(url)
    for tbl in tables:
        cols = {str(c).strip(): c for c in tbl.columns}
        col = cols.get(symbol_col) or cols.get("Ticker symbol") or cols.get("Ticker")
        if col is None:
            continue
        syms_raw = (tbl[col].astype(str).str.replace(".", "-", regex=False)
                    .str.strip().str.upper().tolist())
        syms = [s for s in syms_raw if s and s != "NAN"]
        sectors: dict[str, str] = {}
        sec_col = next((c for c in tbl.columns if "GICS Sector" in str(c)), None)
        if sec_col is not None:
            for sym, sec in zip(syms_raw, tbl[sec_col].astype(str)):
                if sym and sym != "NAN" and sec and sec != "nan":
                    sectors[sym] = sec.strip()
        return syms, sectors
    return [], {}


_CACHE: dict[str, tuple[list[str], dict[str, str]]] = {}


def _constituents(name: str, url: str, min_count: int,
                  seed: list[str]) -> tuple[list[str], dict[str, str]]:
    if name in _CACHE:
        return _CACHE[name]
    try:
        syms, sectors = _fetch_wiki_table(url)
        if len(syms) >= min_count:
            log.info("Loaded %d %s symbols dynamically", len(syms), name)
            _CACHE[name] = (syms, sectors)
            return _CACHE[name]
    except Exception as exc:  # noqa: BLE001
        log.warning("%s fetch failed (%s); using seed", name, exc)
    return list(seed), {}


def get_sp400_constituents(dynamic: bool = True) -> tuple[list[str], dict[str, str]]:
    if not dynamic:
        return list(SP400_SEED), {}
    return _constituents("S&P 400",
                         "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                         300, SP400_SEED)


def get_sp600_constituents(dynamic: bool = True) -> tuple[list[str], dict[str, str]]:
    if not dynamic:
        return list(SP600_SEED), {}
    return _constituents("S&P 600",
                         "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
                         400, SP600_SEED)


def get_sp400(dynamic: bool = True) -> list[str]:
    return get_sp400_constituents(dynamic)[0]


def get_sp600(dynamic: bool = True) -> list[str]:
    return get_sp600_constituents(dynamic)[0]
