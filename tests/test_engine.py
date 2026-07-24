from datetime import datetime
from zoneinfo import ZoneInfo

from trading_bot.config import Settings
from trading_bot.engine import TradingEngine
from trading_bot.state import StateStore
from trading_bot.telegram import TelegramNotifier


class NoopMarket:
    pass


class NoopNews:
    pass


def _engine(tmp_path):
    settings = Settings(
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "trades.csv",
    )
    return TradingEngine(
        settings=settings,
        market_data=NoopMarket(),
        news=NoopNews(),
        notifier=TelegramNotifier(None, None),
        store=StateStore(settings.state_path, settings.journal_path),
    )


def test_scan_slots_follow_0905_then_quarter_hours(tmp_path):
    engine = _engine(tmp_path)
    timezone = ZoneInfo("Europe/Paris")

    assert engine.scan_slot(datetime(2026, 7, 24, 9, 4, tzinfo=timezone)) is None
    assert engine.scan_slot(datetime(2026, 7, 24, 9, 7, tzinfo=timezone)) == "09:05"
    assert engine.scan_slot(datetime(2026, 7, 24, 9, 15, tzinfo=timezone)) == "09:15"
    assert engine.scan_slot(datetime(2026, 7, 24, 9, 44, tzinfo=timezone)) == "09:30"
    assert engine.scan_slot(datetime(2026, 7, 24, 16, 0, tzinfo=timezone)) == "16:00"
    assert engine.scan_slot(datetime(2026, 7, 24, 16, 1, tzinfo=timezone)) is None


def test_scheduled_run_stays_alive_while_position_is_open(tmp_path, monkeypatch):
    settings = Settings(
        state_path=tmp_path / "state.json",
        journal_path=tmp_path / "trades.csv",
        position_poll_seconds=60,
        scheduled_monitor_window_seconds=120,
    )
    engine = TradingEngine(
        settings=settings,
        market_data=NoopMarket(),
        news=NoopNews(),
        notifier=TelegramNotifier(None, None),
        store=StateStore(settings.state_path, settings.journal_path),
    )
    engine.state["position"] = {"ticker": "TEST.PA"}
    calls = []
    sleeps = []

    monkeypatch.setattr(engine, "run_once", calls.append)
    monkeypatch.setattr("trading_bot.engine.time_module.sleep", sleeps.append)

    engine.run_scheduled()

    assert len(calls) == 3
    assert sleeps == [60, 60]
    assert settings.state_path.exists()


def test_scheduled_run_exits_immediately_without_position(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    calls = []

    monkeypatch.setattr(engine, "run_once", calls.append)
    monkeypatch.setattr(
        "trading_bot.engine.time_module.sleep",
        lambda _: raise_unexpected_sleep(),
    )

    engine.run_scheduled()

    assert len(calls) == 1


def raise_unexpected_sleep():
    raise AssertionError("Aucune attente n'est attendue sans position.")
