"""13F institutional-holdings data model + tracked funds."""
from __future__ import annotations

from dataclasses import dataclass

# Curated notable institutions (name -> SEC CIK). Used to query 13F-HR filings.
# CIKs are stable identifiers; edit freely.
TRACKED_FUNDS: dict[str, str] = {
    "Berkshire Hathaway": "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Baillie Gifford": "0001088875",
    "Tiger Global": "0001167483",
    "Coatue": "0001135730",
    "Lone Pine": "0001061165",
    "Two Sigma": "0001179392",
}

# 13F is reported with a delay (up to 45 days after quarter end). Treat as a
# slow-moving confirmation signal, never a fast trigger.
CHANGE_TYPES = ("new", "increased", "decreased", "exited", "unchanged")


@dataclass
class HoldingChange:
    """One fund's quarter-over-quarter change in a single ticker."""

    symbol: str
    fund: str
    change_type: str            # one of CHANGE_TYPES
    prev_value: float = 0.0     # USD reported value last quarter
    curr_value: float = 0.0     # USD reported value this quarter

    @property
    def signed_weight(self) -> float:
        """Signal contribution: + for new/increase, - for exit/decrease."""
        if self.change_type == "new":
            return 1.0
        if self.change_type == "exited":
            return -1.0
        if self.change_type == "increased":
            return 0.6
        if self.change_type == "decreased":
            return -0.6
        return 0.0
