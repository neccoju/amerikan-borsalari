"""Fetch Wikipedia tables with a browser User-Agent.

pandas.read_html uses urllib's default UA, which Wikipedia rejects with HTTP 403
(notably from datacenter IPs like CI runners). Fetching via requests with a
descriptive browser UA returns 200, so the dynamic index-constituent lists work
in CI rather than silently falling back to the static seed.
"""
from __future__ import annotations

import io

from ..utils.logging import get_logger

log = get_logger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 usbot/0.1 (research)",
    "Accept": "text/html,application/xhtml+xml",
}


def read_wikipedia_tables(url: str):
    """Return the list of tables on a Wikipedia page (browser-UA fetch)."""
    import pandas as pd
    import requests

    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))
