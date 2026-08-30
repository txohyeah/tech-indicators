"""核心指标计算测试（金牛通道/评级、形态检查、短线信号）。"""

from __future__ import annotations

import pandas as pd

from tech_indicators.indicators import (
    check_golden_bull_channel,
    check_golden_bull_position_rating,
    check_ma_convergence,
    check_recent_limit_up,
    check_structure_rising,
    compute_golden_bull_lines,
    compute_indicators,
)


def test_compute_indicators_structure(daily_600519):
    ind = compute_indicators(daily_600519)
    for key in ("available", "history_days", "trade_date", "close", "pct_chg", "vol", "circ_mv", "ma5", "warnings"):
        assert key in ind, f"missing key {key}"
    assert ind["available"] is True
    assert ind["history_days"] == len(daily_600519)
    assert isinstance(ind["warnings"], list)


def test_compute_indicators_ma_bullish_type(daily_600519):
    ind = compute_indicators(daily_600519)
    assert ind["ma_bullish"] in (True, False, None)


def test_compute_golden_bull_lines_columns(daily_600519):
    lines = compute_golden_bull_lines(daily_600519)
    for col in ("golden_bull", "golden_bull_trend", "golden_bull_2", "channel_upper", "channel_lower"):
        assert col in lines.columns, f"missing column {col}"
    assert len(lines) == len(daily_600519)


def test_check_golden_bull_channel_output(daily_600519):
    result = check_golden_bull_channel(daily_600519)
    assert isinstance(result, dict)
    assert "passed" in result
    assert result["passed"] in (True, False)


def test_check_golden_bull_position_rating_output(daily_600519):
    result = check_golden_bull_position_rating(daily_600519)
    assert isinstance(result, dict)
    assert "passed" in result
    assert "ratings" in result
    assert result["passed"] in (True, False)


def test_check_structure_rising_output(daily_600519):
    result = check_structure_rising(daily_600519)
    assert isinstance(result, dict)
    assert "passed" in result


def test_check_ma_convergence_output(daily_600519):
    result = check_ma_convergence(daily_600519)
    assert isinstance(result, dict)
    assert "passed" in result


def test_check_recent_limit_up_output(daily_600519):
    result = check_recent_limit_up(daily_600519)
    assert isinstance(result, dict)
    assert "passed" in result


def test_empty_frame_returns_available_false():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "vol"])
    ind = compute_indicators(empty)
    assert ind["available"] is False
    assert "无 K 线数据" in ind["warnings"]


def test_short_frame_still_available_with_history_days():
    small = pd.DataFrame(
        {
            "trade_date": ["20240101", "20240102", "20240103"],
            "open": [1.0] * 3,
            "high": [1.2] * 3,
            "low": [0.9] * 3,
            "close": [1.1] * 3,
            "vol": [100.0] * 3,
        }
    )
    ind = compute_indicators(small)
    assert ind["available"] is True
    assert ind["history_days"] == 3