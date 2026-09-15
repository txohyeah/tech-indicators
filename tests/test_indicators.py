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


def _flat_tail_frame(flat_bars: int, bars: int = 220) -> pd.DataFrame:
    """构造末尾 ``flat_bars`` 根完全同价的行情（连续同价 = 价格零波动）。"""
    import numpy as np

    base = np.linspace(10.0, 20.0, bars - flat_bars)
    close = np.concatenate([base, np.full(flat_bars, base[-1])]) if flat_bars else base
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2020-01-01", periods=len(close), freq="B").strftime("%Y%m%d"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "vol": [1000.0] * len(close),
        }
    )


def test_check_golden_bull_channel_survives_flat_tail():
    """末尾连续同价 ≥7 根时不得抛 DataError（停牌补数、长期横盘会造出这种数据）。

    原实现写的是 ``denominator.replace(0, pd.NA)``：float64 列被 pd.NA 替换后会退化
    成 object dtype，而末尾连续同价恰好让分母整段为 0，紧接着的聚合就抛
    ``DataError: No numeric types to aggregate``。
    阈值是 7 根：分母经过两层 XMA(周期 6，半窗口 3)，第 7 根起窗口内全是 0。
    实测边界：6 根正常，7 根起抛错。
    """
    for flat in (0, 6, 7, 10, 30):
        result = check_golden_bull_channel(_flat_tail_frame(flat))
        assert isinstance(result, dict), f"末尾同价 {flat} 根时应当返回结果而不是抛错"