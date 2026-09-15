"""破位判据回归测试：`_cross_down` 必须以「前一交易日收盘价」为基准。

背景（2026-09-15 修复）
------------------------
原实现用「**当日开盘价**」做基准（``open > 上沿 且 收盘 < 上沿``）。跳空低开时
开盘价已经落在上沿之下，``open > 上沿`` 直接不成立 —— 真正的破位被漏判：

    上沿 99，昨收 100（在上方），今日跳空低开到 97、收 96
    旧判据：open(97) > 99 → False → 漏判
    新判据：prev_close(100) > 99 且 close(96) < 99 → 触发

同时按 2026-09-15 决策「只保留破位、不要贴价预警」，删除了 ``_close_is_near_upper``
（贴近上沿收阴的预警口径）。这组测试把两条都钉住，防止回退。
"""

from __future__ import annotations

import tech_indicators.golden_bull_trading as gbt
from tech_indicators.golden_bull_trading import _candle_context, _cross_down


def _candles(prev_close, open_, high, low, close):
    return {
        "prev_close": prev_close,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }


def test_gap_down_breakout_is_detected():
    """跳空低开破位必须判为破位（本次修复的核心回归点）。"""
    candles = _candles(prev_close=100.0, open_=97.0, high=97.5, low=95.5, close=96.0)
    assert _cross_down(candles, 99.0) is True


def test_high_open_then_reverse_is_detected():
    """高开在上沿之上、收盘跌破，同样算破位。"""
    candles = _candles(prev_close=100.0, open_=101.0, high=101.5, low=95.0, close=96.0)
    assert _cross_down(candles, 99.0) is True


def test_no_breakout_when_price_stays_above():
    """始终在上沿之上：不是破位。"""
    candles = _candles(prev_close=100.0, open_=101.0, high=102.0, low=100.5, close=101.0)
    assert _cross_down(candles, 99.0) is False


def test_no_breakout_when_already_below():
    """前一日收盘已在线下：不是「跌破」，不该重复报警。"""
    candles = _candles(prev_close=98.0, open_=97.5, high=98.0, low=96.0, close=96.5)
    assert _cross_down(candles, 99.0) is False


def test_no_breakout_when_closing_back_above():
    """盘中跌破但收盘收回线上：按「收盘价口径」不算破位。"""
    candles = _candles(prev_close=100.0, open_=100.5, high=101.0, low=95.0, close=99.5)
    assert _cross_down(candles, 99.0) is False


def test_missing_prev_close_is_failsafe():
    """数据缺失时不做破位假设（宁可不减仓，也不用错数据触发减仓）。"""
    candles = _candles(prev_close=None, open_=101.0, high=101.0, low=95.0, close=96.0)
    assert _cross_down(candles, 99.0) is False


def test_missing_line_price_is_failsafe():
    """上沿缺失时不触发。"""
    candles = _candles(prev_close=100.0, open_=97.0, high=97.5, low=95.5, close=96.0)
    assert _cross_down(candles, None) is False


def test_candle_context_exposes_prev_close():
    """破位判据依赖的 prev_close 必须由 metrics 透传进 candles。"""
    ctx = _candle_context(
        {"close": 96.0, "open": 97.0, "high": 97.5, "low": 95.5, "prev_close": 100.0}
    )
    assert ctx["prev_close"] == 100.0
    assert ctx["close"] == 96.0


def test_candle_context_prev_close_defaults_to_none():
    """metrics 里没有 prev_close 时不得抛异常，应退化为 None。"""
    ctx = _candle_context({"close": 96.0, "open": 97.0, "high": 97.5, "low": 95.5})
    assert ctx["prev_close"] is None


def test_near_upper_helper_is_removed():
    """贴价预警口径（_close_is_near_upper）已删除，防止被改回来。"""
    assert not hasattr(gbt, "_close_is_near_upper")
