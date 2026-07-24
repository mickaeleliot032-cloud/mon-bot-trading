from datetime import date

from trading_bot.state import StateStore


def test_new_day_keeps_capital_and_resets_daily_limits(tmp_path):
    store = StateStore(tmp_path / "state.json", tmp_path / "trades.csv")
    state = store.load(date(2026, 7, 23), 1000.0)
    state["capital"] = 1008.0
    state["entry_taken"] = True
    store.save(state)

    next_day = store.load(date(2026, 7, 24), 1000.0)

    assert next_day["capital"] == 1008.0
    assert next_day["daily_start_capital"] == 1008.0
    assert not next_day["entry_taken"]
    assert next_day["position"] is None
