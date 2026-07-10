"""Phase 4: adaptive factor-weight learning (paper-only, look-ahead-safe)."""
from .factor_ic import compute_factor_ic, realized_returns, smooth_ic
from .online_weights import update_weights, normalize_weights

__all__ = ["compute_factor_ic", "realized_returns", "smooth_ic",
           "update_weights", "normalize_weights"]
