"""Command-line entrypoint: `usbot` or `python -m usbot`."""
from __future__ import annotations

import argparse
import json
import sys

from .orchestrator import run_daily
from .utils.dates import is_trading_day, market_status
from .utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="usbot", description="US market analysis & portfolio bot")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the daily pipeline")
    run_p.add_argument("--force", action="store_true",
                       help="Run even on non-trading days / treat as month-end")
    run_p.add_argument("--dry-run", action="store_true",
                       help="Do not send email; log report only")
    run_p.add_argument("--config-dir", default=None, help="Override config directory")

    sub.add_parser("status", help="Print today's market status and exit")

    bt = sub.add_parser("backtest", help="Run a look-ahead-safe momentum backtest")
    bt.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    bt.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    bt.add_argument("--benchmark", default="SPY")
    bt.add_argument("--top-n", type=int, default=10)
    bt.add_argument("--lookback", type=int, default=126, help="Momentum lookback (trading days)")
    bt.add_argument("--cost-bps", type=float, default=10.0)
    bt.add_argument("--trials", type=int, default=1,
                    help="Number of strategy configs explored — deflates the Sharpe "
                         "for multiple testing (López de Prado). 1 = no deflation.")
    bt.add_argument("--walk-forward", action="store_true",
                    help="Compare adaptive vs static factor weighting (Phase 4)")
    bt.add_argument("--composite", action="store_true",
                    help="Backtest the live technical-momentum composite (12-1 blend + "
                         "risk-adjusted momentum + drawdown) instead of single-lookback momentum")
    bt.add_argument("--config-dir", default=None)

    args = parser.parse_args(argv)

    if args.command in (None, "run"):
        run_daily(force=getattr(args, "force", False),
                  dry_run=getattr(args, "dry_run", False),
                  config_dir=getattr(args, "config_dir", None))
        return 0

    if args.command == "status":
        print(f"trading_day={is_trading_day()} status={market_status()}")
        return 0

    if args.command == "backtest":
        return _run_backtest_cmd(args)

    parser.print_help()
    return 1


def _run_backtest_cmd(args) -> int:
    setup_logging("INFO")
    from .backtest import BacktestConfig, momentum_weight_fn, run_backtest
    from .config import load_settings
    from .data import fetch_prices
    from .data.cache import Cache
    from .universe import build_universe

    settings = load_settings(args.config_dir)
    universe = build_universe(settings.settings)
    symbols = sorted(set(universe.symbols) | {args.benchmark})

    cache = Cache(settings.get("data", {}).get("cache_dir", "data/raw"))
    # Need long history for a multi-year backtest; pull max available.
    pdata = fetch_prices(symbols, period_days=3650, cache=cache)
    if not pdata.symbols:
        print("Backtest aborted: no price data could be fetched.")
        return 1

    cfg = BacktestConfig(start_date=args.start, end_date=args.end,
                         benchmark=args.benchmark, cost_bps=args.cost_bps)
    if getattr(args, "walk_forward", False):
        from .backtest import walk_forward_compare

        comp = walk_forward_compare(pdata.history, cfg, top_n=args.top_n,
                                    n_trials=int(args.trials))
        print(json.dumps(comp.summary(), indent=2))
        return 0
    if getattr(args, "composite", False):
        from .backtest import composite_momentum_weight_fn

        weight_fn = composite_momentum_weight_fn(top_n=args.top_n)
    else:
        weight_fn = momentum_weight_fn(top_n=args.top_n, lookback=args.lookback)
    result = run_backtest(pdata.history, weight_fn=weight_fn, config=cfg)
    from .backtest import deflated_sharpe_from_equity

    summary = result.summary()
    summary["deflated_sharpe"] = deflated_sharpe_from_equity(
        result.equity, n_trials=int(args.trials)).as_dict()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
