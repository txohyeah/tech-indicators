"""起爆点（ignition）模块测试：RSI 口径、一波一次、位置过滤、形态过滤、卖出状态机、交易计划。

用合成 K 线把每条规则单独打穿，不依赖真实行情是否恰好触发。
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import tech_indicators.ignition as ig

from tech_indicators.ignition import (
    IGNITION_TRIGGER,
    build_ignition_trade_plan,
    ignition_candle_filter,
    ignition_cross_signal,
    ignition_position_filter,
    ignition_position_state,
    ignition_rsi,
    ignition_signal,
    ignition_signal_breakdown,
    ignition_signal_series,
)


def frame(closes, opens=None, highs=None, lows=None, vols=None):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    highs = np.maximum(closes, opens) if highs is None else np.asarray(highs, dtype=float)
    lows = np.minimum(closes, opens) if lows is None else np.asarray(lows, dtype=float)
    vols = np.full(n, 1000.0) if vols is None else np.asarray(vols, dtype=float)
    return pd.DataFrame({
        "trade_date": [f"2026{1 + i // 28:02d}{1 + i % 28:02d}" for i in range(n)],
        "open": opens, "high": highs, "low": lows, "close": closes, "vol": vols,
    })


def deep_pullback_then_bounce(bounce_pct=0.09, bounce_day_vol=1.2):
    """70 根横盘在 100 → 45 根阴跌到 52（60 日窗口内回撤 >40%）→ 末根放量反弹。

    反弹 9% 时 RSI6 从 0 上穿 40、离 60 日低点约 9%，各项过滤同时满足。
    """
    closes = [100.0] * 70 + list(np.linspace(100, 52, 45))
    closes[-1] = closes[-2] * (1.0 + bounce_pct)
    opens = list(closes[:-1]) + [closes[-2]]
    highs = [max(o, c) for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.999 for o, c in zip(opens, closes)]
    vols = [1000.0] * (len(closes) - 1) + [1000.0 * bounce_day_vol]
    return frame(closes, opens, highs, lows, vols)


def gentle_ignition():
    """与 deep_pullback_then_bounce 同一段阴跌，但用三根 +2.5% 才把 RSI6 推过 40。

    起爆日涨幅 ≤5%，因此买点落在起爆当日（不需要延后），用于测普通 buy 与状态机。
    """
    closes = [100.0] * 70 + list(np.linspace(100, 52, 45))
    for _ in range(3):
        closes.append(closes[-1] * 1.025)
    opens = list(closes[:-1]) + [closes[-2]]
    highs = [max(o, c) for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.999 for o, c in zip(opens, closes)]
    return frame(closes, opens, highs, lows)


# ---------- RSI 口径 ----------
def test_rsi_extremes():
    up = frame(np.linspace(10, 60, 90))
    down = frame(np.linspace(60, 10, 90))
    assert ignition_rsi(up).iloc[-1] > 99.0
    assert ignition_rsi(down).iloc[-1] < 1.0


def test_rsi_matches_wilder_recursion_on_price_differences():
    """钉住口径：必须等于对"收盘价价差"做 Wilder 递推（SMA(X,N,1)）的结果。"""
    closes = np.array([10, 11, 10.5, 12, 13, 12.2, 14, 15.5, 14.8, 16, 17, 16.4] * 12, dtype=float)
    df = frame(closes)
    n = 6
    up, dn = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        up.append(max(d, 0.0)); dn.append(max(-d, 0.0))
    g = up[0]; l = dn[0]
    for i in range(1, len(up)):
        g = g + (up[i] - g) / n
        l = l + (dn[i] - l) / n
    expected = 100.0 * g / (g + l)
    assert float(ignition_rsi(df).iloc[-1]) == pytest.approx(expected, rel=1e-9)


def test_rsi_differs_from_percent_return_version_when_price_level_moves():
    """价差口径 ≠ 涨跌幅口径：价格量级漂移大时两者给出不同结果（通达信用前者）。"""
    closes = list(20 * (1.02 ** np.arange(120)))          # 一路指数上涨，价格量级翻倍
    closes = [c * (1 + 0.01 * ((-1) ** i)) for i, c in enumerate(closes)]
    df = frame(closes)
    s = pd.Series(df.close.values)
    diff_ret = s.diff() / s.shift(1) * 100.0              # 涨跌幅口径
    n = 6
    up = diff_ret.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-diff_ret.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    pct_based = float((100 * up / (up + dn)).iloc[-1])
    assert not np.isclose(float(ignition_rsi(df).iloc[-1]), pct_based, atol=1e-3)


# ---------- 一波只报一次 ----------
def test_cross_fires_exactly_once_while_above():
    df = deep_pullback_then_bounce()
    cross = ignition_cross_signal(df)
    rsi = ignition_rsi(df)
    assert bool(cross.iloc[-1]), "反弹日应当触发上穿"
    assert float(rsi.iloc[-2]) <= IGNITION_TRIGGER < float(rsi.iloc[-1])
    # 之后再连续收阳但不回落，不应重复触发
    after = list(df.close.values) + [df.close.values[-1] * (1.03 ** k) for k in range(1, 6)]
    df2 = frame(after)
    assert int(ignition_cross_signal(df2).sum()) == int(cross.sum()), "持续在上方期间不得重复上穿"


# ---------- 位置过滤 ----------
def test_position_filter_requires_deep_and_off_bottom():
    assert bool(ignition_position_filter(deep_pullback_then_bounce()).iloc[-1])


def test_position_filter_rejects_shallow_dip():
    closes = [100.0] * 60 + list(np.linspace(100, 92, 60)) + [96.0]
    assert not bool(ignition_position_filter(frame(closes)).iloc[-1])   # 只跌 8%，不够深


def test_position_filter_rejects_stuck_at_bottom():
    """贴着 60 日新低反弹不到 5%（下跌中继典型形态）必须被拒。"""
    closes = [100.0] * 60 + list(np.linspace(99, 55, 80))
    closes[-1] = closes[-2] * 1.02          # 离底仅 2%
    opens = list(closes[:-1]) + [closes[-2]]
    assert not bool(ignition_position_filter(frame(closes, opens)).iloc[-1])


# ---------- 形态过滤 ----------
def test_candle_filter_rejects_fake_bear():
    """收涨但收阴（假阴线）应被拒。"""
    df = deep_pullback_then_bounce()
    i = len(df) - 1
    df.loc[i, "open"] = df.close.values[i] * 1.05      # 大幅高开 → 收盘低于开盘但仍涨
    df.loc[i, "high"] = df.open.values[i]
    assert df.close.values[i] > df.close.values[i - 1]
    assert not bool(ignition_candle_filter(df).iloc[-1])


def test_candle_filter_rejects_long_upper_shadow():
    df = gentle_ignition()
    i = len(df) - 1
    body_top = max(df.close.values[i], df.open.values[i])
    df.loc[i, "high"] = body_top + 4.0 * (body_top - df.low.values[i])   # 上影占 80% > 3/4
    assert ignition_position_filter(df).iloc[-1]                     # 位置仍然合格
    assert not bool(ignition_candle_filter(df).iloc[-1])
    assert not bool(ignition_signal(df))


# ---------- 卖出状态机 ----------
def test_state_is_none_without_signal():
    df = frame([100.0] * 120)
    assert ignition_position_state(df) is None


def test_pending_entry_when_ignition_bar_gains_more_than_five_pct():
    """起爆日涨 9%（>5%）且它就是最后一根 → 买点待定，不能当成今天已建仓。"""
    df = deep_pullback_then_bounce()
    st = ignition_position_state(df)
    assert st["pending_entry"] is True
    assert st["entry_price"] is None and st["closed"] is False
    plan = build_ignition_trade_plan(df, current_position_pct=0.0)
    assert plan["action"] == "buy_next_bar"
    assert plan["signal_type"] == "ignition_entry_pending"


def test_entry_lands_next_bar_once_the_next_bar_exists():
    df = deep_pullback_then_bounce()
    nxt = list(df.close.values) + [df.close.values[-1] * 1.01]
    st = ignition_position_state(frame(nxt))
    assert st["pending_entry"] is False
    assert st["entry_price"] == pytest.approx(round(nxt[-1], 4), rel=1e-6)   # 成交价 = 次日收盘
    assert st["bars_held"] == 0


def test_stop_line_respects_ten_percent_floor():
    df = gentle_ignition()
    st = ignition_position_state(df)
    assert st is not None and not st["closed"]
    assert st["stop_line"] >= st["entry_price"] * (1.0 - 0.10) - 1e-9


def test_stop_loss_triggers_on_close_below_line():
    df = gentle_ignition()
    stop = ignition_position_state(df)["stop_line"]
    closes = list(df.close.values) + [stop * 0.97]
    st = ignition_position_state(frame(closes))
    assert st["closed"] and st["exit_reason"] == "stop_loss"


def test_closed_state_reports_exit_not_today():
    """已平仓的状态必须按平仓那一根来算持有天数与收益，不能拿今天的价糊上去。"""
    df = gentle_ignition()
    stop = ignition_position_state(df)["stop_line"]
    closes = list(df.close.values) + [stop * 0.97] + [999.0, 999.0]     # 平仓后又涨回去
    st = ignition_position_state(frame(closes))
    assert st["closed"] and st["exit_reason"] == "stop_loss"
    assert st["bars_held"] == 1
    assert st["exit_price"] == pytest.approx(stop * 0.97, rel=1e-6)
    assert st["unrealized_pct"] < 0                                      # 按平仓价算，不是按 999 算
    assert st["last_close"] == pytest.approx(999.0, rel=1e-6)            # 当前价仍如实给出


def test_profit_mode_switches_to_trailing_only(monkeypatch):
    """显式打开移动止盈：涨过 20% 进入利润奔跑，回撤总涨幅 35% 才离场。

    移动止盈默认是关的（15 槽组合层实测关闭的 C2 优于开启的 C3），所以要用 monkeypatch 打开。
    """
    monkeypatch.setattr(ig, "IGNITION_USE_TRAILING", True)
    df = gentle_ignition()
    entry = ignition_position_state(df)["entry_price"]
    closes = list(df.close.values) + [entry * 1.40, entry * 1.50, entry * 1.10]
    st = ignition_position_state(frame(closes))
    assert st["running"] is True
    assert st["closed"] and st["exit_reason"] == "trailing_take_profit"


def test_trailing_disabled_by_default():
    """默认配置（C2 定稿）：同样的深回落既不进利润奔跑、也不出场。"""
    df = gentle_ignition()
    entry = ignition_position_state(df)["entry_price"]
    closes = list(df.close.values) + [entry * 1.40, entry * 1.50, entry * 1.10]
    st = ignition_position_state(frame(closes))
    assert st["running"] is False and st["closed"] is False


def test_trailing_keeps_running_when_gain_small(monkeypatch):
    """（开关打开时）涨幅未达 20% 不进利润奔跑，小幅回落不应被移动止盈打掉。"""
    monkeypatch.setattr(ig, "IGNITION_USE_TRAILING", True)
    df = gentle_ignition()
    entry = ignition_position_state(df)["entry_price"]
    closes = list(df.close.values) + [entry * 1.10, entry * 1.06]
    st = ignition_position_state(frame(closes))
    assert st["running"] is False and st["closed"] is False


def test_new_ignition_point_reanchors_stop():
    """未进利润奔跑时出现新起爆点，买点与止损线都应随之下移/更新。"""
    df = gentle_ignition()
    first = ignition_position_state(df)
    base = list(df.close.values)
    closes = base + [base[-1] * (1 - 0.02 * k) for k in range(1, 11)] + [base[-1] * 0.8 * 1.025,
                                                                          base[-1] * 0.8 * 1.025 * 1.025,
                                                                          base[-1] * 0.8 * 1.025 ** 3]
    st = ignition_position_state(frame(closes))
    assert st["signal_date"] > first["signal_date"]
    assert st["entry_price"] < first["entry_price"]


# ---------- 交易计划 ----------
def test_plan_buys_on_signal_when_flat():
    df = gentle_ignition()
    assert ignition_signal(df), "该构造应同时满足信号/位置/形态三项"
    plan = build_ignition_trade_plan(df, current_position_pct=0.0)
    assert plan["action"] == "buy" and plan["side"] == "buy"
    assert plan["signal_type"] == "ignition_entry"
    assert plan["target_position_pct"] == pytest.approx(0.20)
    assert plan["stop_line_price"] is not None


def test_plan_holds_and_reports_stop_for_position():
    df = gentle_ignition()
    plan = build_ignition_trade_plan(df, current_position_pct=0.20)
    assert plan["action"] == "hold"
    assert plan["signal_type"].startswith("ignition_")
    assert any("stop line" in r for r in plan["reason"])


def test_breakdown_explains_why_not():
    df = frame([100.0] * 60 + list(np.linspace(100, 95, 60)) + [96.0])
    b = ignition_signal_breakdown(df)
    assert b["signal"] is False
    assert b["position_ok"] is False
    assert set(["rsi", "cross", "drawdown_pct", "off_bottom_pct", "upper_shadow_ratio"]) <= set(b)


def test_signal_series_index_aligned_with_input():
    df = deep_pullback_then_bounce()
    s = ignition_signal_series(df)
    assert len(s) == len(df) and list(s.index) == list(df.index)
    assert s.dtype == bool


def test_smoke_on_real_fixture(daily_600519):
    df = daily_600519
    assert isinstance(ignition_signal(df), bool)
    plan = build_ignition_trade_plan(df, current_position_pct=0.0)
    assert plan["ignition_breakdown"]["rsi"] is not None
    st = ignition_position_state(df)
    if st is not None:
        assert st["entry_price"] > 0 and st["bars_held"] >= 0


# ---------- 2026-09-05 因果版定稿契约 ----------
def test_default_exit_config_matches_validated_setup():
    """默认配置必须就是 15 槽组合层验证过的 C2，不许悄悄漂移。"""
    assert ig.IGNITION_UPPER_EXIT == "full"
    assert ig.IGNITION_USE_TRAILING is False
    assert ig.IGNITION_STOP_BARS == 30
    assert ig.IGNITION_STOP_PCT == 0.10
    assert ig.IGNITION_TRAIL_FRACTION == 0.35
    ig.validate_exit_config()          # 默认组合必须合法


def test_rolling_stop_excludes_current_bar():
    """止损窗口不含当根：昨低 51.5、今收 51.0 → 必须触发；若含当根（窗口=50.9）则永远打不到。"""
    df = gentle_ignition()
    entry = ignition_position_state(df)["entry_price"]
    # 起爆后 gently 回落到 51.8 并横盘 40 根（low 51.5，收盘从不低于昨低 → 不触发止损）
    closes = list(df.close.values); opens = list(df.open.values)
    highs = list(df.high.values); lows = list(df.low.values)
    px = closes[-1]
    for _ in range(25):                       # 温和回落 56 → 51.8
        px *= 0.997
        closes.append(px); opens.append(px * 1.001)
        highs.append(max(closes[-1], opens[-1]) * 1.002)
        lows.append(min(closes[-1], opens[-1]) * 0.99)
    for _ in range(40):                       # 横盘 51.8
        closes.append(51.8); opens.append(51.8)
        highs.append(52.0); lows.append(51.5)
    closes.append(51.0); opens.append(51.8)   # 末日：收盘砸穿昨低 51.5
    highs.append(51.8); lows.append(50.9)
    st = ignition_position_state(frame(closes, opens, highs, lows))
    assert st["closed"] and st["exit_reason"] == "stop_loss"


def test_rolling_stop_ignores_todays_low():
    """当根的长下影不参与止损线：今低砸到 50.0 但收盘收回 51.8 → 不出场。"""
    df = gentle_ignition()
    closes = list(df.close.values); opens = list(df.open.values)
    highs = list(df.high.values); lows = list(df.low.values)
    px = closes[-1]
    for _ in range(25):
        px *= 0.997
        closes.append(px); opens.append(px * 1.001)
        highs.append(max(closes[-1], opens[-1]) * 1.002)
        lows.append(min(closes[-1], opens[-1]) * 0.99)
    for _ in range(40):
        closes.append(51.8); opens.append(51.8)
        highs.append(52.0); lows.append(51.5)
    closes.append(51.8); opens.append(51.8)   # 末日：长下影 50.0，收盘收回
    highs.append(51.8); lows.append(50.0)
    st = ignition_position_state(frame(closes, opens, highs, lows))
    assert st["closed"] is False


def test_bear_channel_no_longer_forces_exit():
    """旧引擎在熊市通道里撞上沿会强制清仓，因果版已删；清仓理由统一为 upper_pressure_exit。"""
    src = inspect.getsource(ig.ignition_position_state)
    assert "upper_pressure_bear_exit" not in src
    assert "upper_pressure_exit" in src
    assert 'causal=True' in src


def test_half_exit_requires_backstop(monkeypatch):
    """减半出场若无任何兜底清仓（止盈/止损全关），必须直接报错，不许退化成买入持有。"""
    monkeypatch.setattr(ig, "IGNITION_UPPER_EXIT", "half")
    monkeypatch.setattr(ig, "IGNITION_USE_TRAILING", False)
    monkeypatch.setattr(ig, "IGNITION_STOP_PCT", 0.0)
    monkeypatch.setattr(ig, "IGNITION_STOP_BARS", 0)
    with pytest.raises(ValueError):
        ig.validate_exit_config()
