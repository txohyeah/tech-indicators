"""金牛通道的因果性测试：causal=True 必须只看过去，causal=False 必须与通达信图一致。

这组测试的存在理由：2026-09-05 发现 `_tdx_xma` 是居中平均（第 i 根用了 i+1…i+12 的数据），
导致所有用它做的历史回测含未来函数。修好之后，用测试把这条底线钉住，防止以后又被改回去。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tech_indicators.ignition import golden_channel_state
from tech_indicators.indicators import (
    GOLDEN_BULL_XMA_PERIOD,
    _causal_double_xma,
    _tdx_xma,
    compute_golden_bull_lines,
)


def make_frame(bars: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50 + np.cumsum(rng.normal(0.0, 1.0, bars))
    high = close + np.abs(rng.normal(0.0, 0.8, bars))
    low = close - np.abs(rng.normal(0.0, 0.8, bars))
    return pd.DataFrame({
        "trade_date": pd.date_range("2020-01-01", periods=bars, freq="B").strftime("%Y%m%d"),
        "open": close, "high": high, "low": low, "close": close,
    })


FRAME = make_frame()


def test_causal_lines_are_truncation_invariant():
    """核心不变式：任意一根的值，等于"把数据截断到那一根再算"的末值 —— 即不看未来。"""
    full = compute_golden_bull_lines(FRAME, causal=True)
    for i in range(GOLDEN_BULL_XMA_PERIOD + 5, len(FRAME) - 1):
        head = FRAME.iloc[: i + 1]
        part = compute_golden_bull_lines(head, causal=True).iloc[-1]
        for column in ("channel_upper", "golden_bull_trend", "channel_lower"):
            assert full[column].iloc[i] == pytest.approx(part[column], rel=0, abs=1e-9), (column, i)


def test_causal_bear_flag_has_no_future_dependency():
    state = golden_channel_state(FRAME, causal=True)
    for i in range(40, len(FRAME) - 1):
        part = golden_channel_state(FRAME.iloc[: i + 1], causal=True).iloc[-1]
        assert bool(state["bear"].iloc[i]) == bool(part["bear"])


def test_appending_future_bars_does_not_move_causal_values():
    """在同一个前缀后面接不同的未来行情，因果版历史值必须一动不动。"""
    prefix = FRAME.iloc[:200]
    base = compute_golden_bull_lines(prefix, causal=True)["channel_upper"].iloc[-1]
    chart_base = compute_golden_bull_lines(prefix, causal=False)["channel_upper"].iloc[-1]
    def joined_with_dates(tail: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat([prefix, tail], ignore_index=True)
        # compute_golden_bull_lines 内部按 trade_date 排序，拼接段必须给连续日期，否则会被打乱
        out["trade_date"] = pd.date_range("2020-01-01", periods=len(out), freq="B").strftime("%Y%m%d")
        return out

    futures = [joined_with_dates(make_frame(bars=60, seed=seed)) for seed in (11, 12, 13)]
    futures.append(joined_with_dates(FRAME.iloc[200:]))
    for joined in futures:
        after = compute_golden_bull_lines(joined, causal=True)["channel_upper"].iloc[199]
        assert after == pytest.approx(base, rel=0, abs=1e-9)          # 因果版不受未来影响
    moved = any(abs(compute_golden_bull_lines(j, causal=False)["channel_upper"].iloc[199] - chart_base) > 1e-9
                for j in futures)
    assert moved, "图上模式对同一根的值竟然不随未来数据变化？需复查 _tdx_xma"


def test_chart_mode_is_not_truncation_invariant_by_design():
    """反向钉死：默认（图上）行为确实会随未来数据改变——这就是它不能用于回测的原因。"""
    full = compute_golden_bull_lines(FRAME, causal=False)["channel_upper"]
    changed = 0
    for i in range(60, len(FRAME) - 15):
        part = compute_golden_bull_lines(FRAME.iloc[: i + 1], causal=False)["channel_upper"].iloc[-1]
        if abs(full.iloc[i] - part) > 1e-9:
            changed += 1
    assert changed > 30, "图上模式竟然不含未来依赖？说明 _tdx_xma 被改过，需同步复查本测试"


def test_chart_mode_formula_matches_tdx_xma():
    """图上模式（causal=False）仍是通达信那套双重 XMA，逐值可复算。"""
    hx = _tdx_xma(_tdx_xma(FRAME["high"], GOLDEN_BULL_XMA_PERIOD), GOLDEN_BULL_XMA_PERIOD)
    lx = _tdx_xma(_tdx_xma(FRAME["low"], GOLDEN_BULL_XMA_PERIOD), GOLDEN_BULL_XMA_PERIOD)
    line_a = (hx - lx) + hx                       # golden_bull
    line_b = lx - (hx - lx)                       # golden_bull_trend（生命线）
    line_c = line_b.ewm(span=25, adjust=False).mean()
    expected = np.maximum(np.maximum(line_a.values, line_b.values), line_c.values)
    got = compute_golden_bull_lines(FRAME, causal=False)
    assert np.allclose(got["channel_upper"].values, expected, equal_nan=True)


def test_default_argument_is_causal():
    """默认参数必须是因果版（2026-09-15 决策）。

    默认值原为 ``causal=False``（图上模式，含未来函数）。问题在于
    ``check_golden_bull_channel`` 等**实盘判定**函数内部调用本函数时没有显式传参，
    等于拿"事后会被改写的线"做实时判断 —— 典型的未来函数。
    把默认值换成安全的一侧，是为了让"忘记传参"这个失误朝向不出错的方向。
    需要用图上一侧的场景（``chart.py`` 绘图）改为显式声明，见下一条测试。
    """
    got = compute_golden_bull_lines(FRAME)["channel_upper"]
    assert got.equals(compute_golden_bull_lines(FRAME, causal=True)["channel_upper"])
    assert not got.equals(compute_golden_bull_lines(FRAME, causal=False)["channel_upper"])


def test_chart_module_explicitly_requests_chart_mode():
    """chart.py 必须显式传 causal=False，否则画出来的线与通达信对不上。

    默认值改成 causal=True 之后，绘图这条唯一需要"图上模式"的路径必须自己声明，
    这条测试钉住它，防止以后被当成冗余参数删掉。
    """
    import inspect

    from tech_indicators import chart

    src = inspect.getsource(chart)
    assert "compute_golden_bull_lines(data, causal=False)" in src, (
        "chart.py 必须显式传 causal=False（画图用图上模式），否则线上图与通达信不一致"
    )


def test_causal_mode_warms_up_with_nan_then_becomes_finite():
    got = compute_golden_bull_lines(FRAME, causal=True)["channel_upper"]
    window = GOLDEN_BULL_XMA_PERIOD // 2 + 1                   # 单层窗口 13，套两层 → 首个有效下标 2*(13-1)
    first_valid = 2 * (window - 1)
    assert got.isna().iloc[:first_valid].all()
    assert np.isfinite(got.iloc[first_valid:].astype(float)).all()
