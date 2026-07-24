"""Point d'entrée du bot de trading V4."""

from __future__ import annotations

import argparse

from trading_bot.engine import TradingEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent de paper trading CAC 40 V4")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Exécute un passage GitHub Actions puis quitte.",
    )
    return parser.parse_args()


def main() -> None:
    engine = TradingEngine()
    if parse_args().scheduled:
        engine.run_scheduled()
    else:
        engine.run_forever()


if __name__ == "__main__":
    main()
