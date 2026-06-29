"""13F institutional-holdings ingestion via SEC EDGAR (edgartools, best-effort).

EDGAR is keyless but requires an identifying User-Agent. We use edgartools when
installed (optional ``[sec]`` extra). 13F reports holdings by CUSIP/issuer, so
ticker resolution is the hard part; we use edgartools' resolved ticker when
available and skip holdings we cannot map. Everything is wrapped so any failure
(missing package, network, schema drift) degrades to a clean skip — this signal
is an experimental, slow-moving confirmation layer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from .model import TRACKED_FUNDS, HoldingChange

log = get_logger(__name__)


@dataclass
class InstitutionalResult:
    changes: list[HoldingChange] = field(default_factory=list)
    enabled: bool = False
    skip_reason: str = ""
    funds_seen: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _ensure_identity() -> None:
    from edgar import set_identity

    identity = (os.environ.get("EDGAR_IDENTITY")
                or os.environ.get("SEC_USER_AGENT")
                or "usbot research contact@example.com")
    set_identity(identity)


def _holdings_by_ticker(thirteenf) -> dict[str, float]:
    """Best-effort {ticker: reported_value} from a ThirteenF object."""
    table = getattr(thirteenf, "infotable", None)
    if table is None:
        return {}
    try:
        df = table if hasattr(table, "columns") else None
        if df is None or df.empty:
            return {}
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("ticker")
        vcol = cols.get("value")
        if tcol is None or vcol is None:
            return {}
        out: dict[str, float] = {}
        for _, row in df.iterrows():
            tk = str(row[tcol]).strip().upper()
            if not tk or tk in ("", "NAN", "NONE"):
                continue
            try:
                out[tk] = out.get(tk, 0.0) + float(row[vcol])
            except (TypeError, ValueError):
                continue
        return out
    except Exception as exc:  # noqa: BLE001
        log.debug("infotable parse failed: %s", exc)
        return {}


def _classify(prev: float, curr: float) -> str:
    if prev <= 0 and curr > 0:
        return "new"
    if prev > 0 and curr <= 0:
        return "exited"
    if curr > prev * 1.10:
        return "increased"
    if curr < prev * 0.90:
        return "decreased"
    return "unchanged"


def _fund_changes(cik: str, fund: str, universe: set[str]) -> list[HoldingChange]:
    from edgar import Company

    filings = Company(cik).get_filings(form="13F-HR")
    # newest first; take the two most recent quarters
    recent = list(filings.latest(2)) if hasattr(filings, "latest") else list(filings)[:2]
    if len(recent) < 2:
        return []
    curr = _holdings_by_ticker(recent[0].obj())
    prev = _holdings_by_ticker(recent[1].obj())
    if not curr and not prev:
        return []
    out = []
    for sym in set(curr) | set(prev):
        if universe and sym not in universe:
            continue
        ct = _classify(prev.get(sym, 0.0), curr.get(sym, 0.0))
        if ct == "unchanged":
            continue
        out.append(HoldingChange(symbol=sym, fund=fund, change_type=ct,
                                 prev_value=prev.get(sym, 0.0),
                                 curr_value=curr.get(sym, 0.0)))
    return out


def fetch_institutional_changes(universe: list[str],
                                funds: dict[str, str] | None = None) -> InstitutionalResult:
    """Fetch 13F quarter-over-quarter changes for tracked funds. Graceful skip."""
    res = InstitutionalResult()
    try:
        _ensure_identity()
    except Exception as exc:  # noqa: BLE001
        res.skip_reason = f"edgartools unavailable: {exc}"
        log.info("Institutional skipped: %s", res.skip_reason)
        return res

    uni = {s.upper() for s in universe}
    funds = funds or TRACKED_FUNDS
    any_data = False
    for fund, cik in funds.items():
        try:
            changes = _fund_changes(cik, fund, uni)
            if changes:
                res.changes.extend(changes)
                res.funds_seen.append(fund)
                any_data = True
        except Exception as exc:  # noqa: BLE001
            res.errors.append(f"13F {fund}: {exc}")
            log.debug("13F %s failed: %s", fund, exc)

    if not any_data:
        res.enabled = False
        if not res.skip_reason:
            res.skip_reason = ("no resolvable 13F holdings (ticker mapping or filings "
                               "unavailable) — experimental signal")
    else:
        res.enabled = True
        log.info("Institutional: %d changes across %d funds",
                 len(res.changes), len(res.funds_seen))
    return res
