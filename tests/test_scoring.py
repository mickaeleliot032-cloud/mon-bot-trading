from trading_bot.scoring import (
    combine_score,
    quantitative_score,
)


def strong_snapshot():
    return {
        "price": 103.0,
        "open": 102.0,
        "previous_close": 101.8,
        "return_open_pct": 0.98,
        "gap_pct": 0.20,
        "momentum_15m_pct": 0.35,
        "ema20": 102.2,
        "ema50": 101.7,
        "vwap": 102.4,
        "rsi14": 61.0,
        "atr_pct": 0.8,
        "volume_ratio": 1.7,
        "bars_today": 20,
        "timestamp": "2026-07-24T10:40:00+02:00",
    }


def test_strong_setup_reaches_signal_without_hard_news_gate():
    snapshot = strong_snapshot()
    quantitative, eligible, reasons = quantitative_score(
        snapshot, market_return_pct=0.2, max_absolute_gap_pct=4.0
    )
    score = combine_score(
        ticker="TEST.PA",
        name="Test",
        sector="Industrie",
        snapshot=snapshot,
        quantitative=quantitative,
        market=65,
        sector_score=65,
        news=50,
        eligible=eligible,
        reasons=reasons,
        thresholds=(65, 72, 80),
    )

    assert eligible
    assert score.final >= 72
    assert score.level in {"SIGNAL", "FORT"}


def test_excessive_gap_is_excluded_even_with_good_momentum():
    snapshot = strong_snapshot()
    snapshot["gap_pct"] = 4.5
    _, eligible, reasons = quantitative_score(
        snapshot, market_return_pct=0.2, max_absolute_gap_pct=4.0
    )

    assert not eligible
    assert any("gap excessif" in reason for reason in reasons)
