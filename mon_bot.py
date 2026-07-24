"""Point d'entrée du bot de trading V4."""

from trading_bot.engine import TradingEngine

if __name__ == "__main__":
    TradingEngine().run_forever()
