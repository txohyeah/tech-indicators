"""交易计划生成测试：金牛 / 复燃点 / 统一 K 线。"""

from __future__ import annotations

import pandas as pd

from tech_indicators.golden_bull_trading import build_golden_bull_trade_plan
from tech_indicators.kline_decision import build_unified_kline_trade_plan
from tech_indicators.reburn import reburn_risk_context, reburn_signal, reburn_strong_volume


def test_reburn_signal_returns_bool(daily_600519):
    signal = reburn_signal(daily_600519)
    assert isinstance(signal, bool)


def test_reburn_risk_context_structure(daily_600519):
    risk = reburn_risk_context(daily_600519, raw_channel_regime="unknown")
    assert isinstance(risk, dict)


def test_reburn_strong_volume_returns_bool(daily_600519):
    assert isinstance(reburn_strong_volume(daily_600519), bool)


def test_golden_bull_plan_structure(daily_600519):
    metrics = {"close": float(daily_600519.iloc[-1]["close"])}
    plan = build_golden_bull_trade_plan(metrics)
    assert isinstance(plan, dict)
    for key in ("action", "side", "signal_type", "target_position_pct", "candidates"):
        assert key in plan, f"missing key {key}"
    assert plan["side"] in ("buy", "sell", "hold", "none")


def test_unified_plan_structure(daily_600519):
    rating = {}
    plan = build_unified_kline_trade_plan(rating, daily_600519)
    assert isinstance(plan, dict)
    for key in ("action", "side", "signal_type", "target_position_pct", "stop_line_name", "reason", "candidates"):
        assert key in plan, f"missing key {key}"