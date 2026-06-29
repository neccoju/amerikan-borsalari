"""Command-line entrypoint: `usbot` or `python -m usbot`."""
from __future__ import annotations

import argparse
import sys

from .orchestrator import run_daily
from .utils.dates import is_trading_day, market_status


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

    args = parser.parse_args(argv)

    if args.command in (None, "run"):
        force = getattr(args, "force", False)
        dry = getattr(args, "dry_run", False)
        cfg = getattr(args, "config_dir", None)
        run_daily(force=force, dry_run=dry, config_dir=cfg)
        return 0

    if args.command == "status":
        print(f"trading_day={is_trading_day()} status={market_status()}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
