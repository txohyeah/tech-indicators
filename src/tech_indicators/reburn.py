"""低位复燃（reburn）信号与交易计划。

信号定义 = RSI(N) 自下而上穿越触发线，等价通达信「临界起爆点」的
`CROSS(RSI6, 40)`。该定义是对 2026-09-04 反推出的原件公式的还原，
取代此前 12 个分支条件的拟合版本（拟合不彻底、且依赖人工调参）。

风险闸门（空头通道 / MA20 下行 / MA60 下行）与量能分级仓位
（强量 100% / 弱量 50% / 单闸门命中 20%）是自有策略层，予以保留。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REBURN_RSI_PERIOD = 6          # 通达信参数 N
REBURN_RSI_TRIGGER = 40.0      # 通达信参数 LL（上限锁 40：只在弱势区触发）
REBURN_MIN_BARS = 62           # 计算与下游计划所需的最小历史长度
REBURN_WEAK_TARGET_POSITION_PCT = 0.50
REBURN_STRONG_TARGET_POSITION_PCT = 1.0
REBURN_RISK_CAP_POSITION_PCT = 0.20
REBURN_MA_SLOPE_LOOKBACK_BARS = 3


def build_reburn_buy_trade_plan(
    history: pd.DataFrame,
    rating: dict[str, Any],
    *,
    current_position_pct: float,
    fallback_trade_plan: dict[str, Any],
    timeframe_label: str = "4h",
) -> dict[str, Any]:
    label = timeframe_label or "K-line"
    if len(history) < 62:
        return {
            **fallback_trade_plan,
            "reburn_signal": None,
            "reburn_risk": {"insufficient_history": True},
            "reason": list(fallback_trade_plan.get("reason", []))
            + [f"insufficient {label} history for Reburn point; legacy plan kept"],
        }
    raw_channel_regime = str(rating.get("channel_regime") or "unknown")
    risk = reburn_risk_context(history, raw_channel_regime)
    signal = reburn_signal(history)
    strong_volume = reburn_strong_volume(history)
    base_plan = {
        "action": "hold" if current_position_pct > 0 else "wait",
        "side": "hold",
        "signal_type": "no_trade",
        "target_position_pct": current_position_pct,
        "position_cap_pct": REBURN_STRONG_TARGET_POSITION_PCT,
        "stop_line_name": None,
        "stop_line_price": None,
        "channel_regime": raw_channel_regime,
        "raw_channel_regime": raw_channel_regime,
        "reburn_signal": signal,
        "reburn_risk": risk,
        "metrics": {
            **(fallback_trade_plan.get("metrics") if isinstance(fallback_trade_plan.get("metrics"), dict) else {}),
            "reburn": signal,
            "reburn_risk": risk,
            "volume_vs_prev_ratio": reburn_volume_vs_prev_ratio(history),
            "reburn_rsi": reburn_rsi_value(history),
            "reburn_rsi_period": REBURN_RSI_PERIOD,
            "reburn_rsi_trigger": REBURN_RSI_TRIGGER,
        },
        "candidates": list(fallback_trade_plan.get("candidates", [])),
    }
    if not signal:
        return {
            **base_plan,
            "reason": [f"no {label} Reburn point; legacy Golden Bull buy signal ignored"],
        }
    if risk["risk_count"] >= 2:
        return {
            **base_plan,
            "reason": [
                f"{label} Reburn point appeared, but two or more risk filters are active; buy ignored",
                *reburn_risk_reasons(risk),
            ],
        }

    target = REBURN_STRONG_TARGET_POSITION_PCT if strong_volume else REBURN_WEAK_TARGET_POSITION_PCT
    signal_type = f"reburn_{label}_strong_buy" if strong_volume else f"reburn_{label}_weak_buy"
    if risk["risk_count"] == 1:
        target = min(target, REBURN_RISK_CAP_POSITION_PCT)
    if current_position_pct >= target - 0.001:
        return {
            **base_plan,
            "target_position_pct": current_position_pct,
            "position_cap_pct": target,
            "signal_type": signal_type,
            "reason": [
                f"{label} Reburn point confirmed, but target position already reached",
                *reburn_risk_reasons(risk),
            ],
        }
    return {
        **base_plan,
        "action": "buy_or_hold_full" if target >= REBURN_STRONG_TARGET_POSITION_PCT else "buy_or_hold_half",
        "side": "buy",
        "signal_type": signal_type,
        "target_position_pct": target,
        "position_cap_pct": target,
        "stop_line_name": "entry_candle_low",
        "stop_line_price": None,
        "reason": [
            f"{label} Reburn point confirmed",
            (
                "strong volume: latest volume is 1.8~2.2x previous bar"
                if strong_volume
                else "weak volume: latest volume is outside 1.8~2.2x previous bar"
            ),
            *reburn_risk_reasons(risk),
        ],
    }


def reburn_risk_context(history: pd.DataFrame, raw_channel_regime: str) -> dict[str, Any]:
    ma20_down = _ma_down(history, 20, REBURN_MA_SLOPE_LOOKBACK_BARS)
    ma60_down = _ma_down(history, 60, REBURN_MA_SLOPE_LOOKBACK_BARS)
    bear = raw_channel_regime == "bear"
    return {
        "bear": bear,
        "ma20_down": ma20_down,
        "ma60_down": ma60_down,
        "risk_count": int(bear) + int(ma20_down) + int(ma60_down),
    }


def reburn_risk_reasons(risk: dict[str, Any]) -> list[str]:
    reasons = []
    if risk.get("bear"):
        reasons.append("risk filter active: Golden Bull raw channel is bear")
    if risk.get("ma20_down"):
        reasons.append("risk filter active: MA20 is below its value 3 bars ago")
    if risk.get("ma60_down"):
        reasons.append("risk filter active: MA60 is below its value 3 bars ago")
    if not reasons:
        reasons.append("no bear/MA20-down/MA60-down risk filter active")
    return reasons


def reburn_strong_volume(history: pd.DataFrame) -> bool:
    ratio = reburn_volume_vs_prev_ratio(history)
    return ratio is not None and 1.8 <= ratio <= 2.2


def reburn_volume_vs_prev_ratio(history: pd.DataFrame) -> float | None:
    if len(history) < 2 or "vol" not in history:
        return None
    latest = _optional_float(history.iloc[-1].get("vol"))
    previous = _optional_float(history.iloc[-2].get("vol"))
    if latest is None or previous in (None, 0):
        return None
    return latest / previous


def reburn_rsi(history: pd.DataFrame, period: int | None = None) -> pd.Series:
    """Wilder RSI，等价通达信 `SMA(MAX(C-REF(C,1),0),N,1)/SMA(ABS(C-REF(C,1)),N,1)*100`。

    必须用**收盘价价差**口径。实测改成涨跌幅口径，与「临界起爆点」的一致率
    会从 J=0.958 掉到 J=0.837。
    """
    n = REBURN_RSI_PERIOD if period is None else max(int(period), 1)
    delta = history["close"].astype(float).diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    denom = gain + loss
    return (100.0 * gain / denom).where(denom != 0)


def reburn_rsi_value(history: pd.DataFrame) -> float | None:
    """最后一根 K 线的 RSI 值，供计划/图表展示；无法计算时返回 None。"""
    if len(history) < 2:
        return None
    values = reburn_rsi(history)
    if values.empty:
        return None
    latest = values.iloc[-1]
    return None if pd.isna(latest) else float(latest)


def reburn_signal(history: pd.DataFrame) -> bool:
    """低位复燃：RSI(N) 上穿触发线（`CROSS(RSI6, 40)`）。

    CROSS 自带"一波只报一次"的性质——必须先把 RSI 打回触发线下方，
    才可能再次上穿，因此不再需要额外的去重、追高、假低位等过滤条件。
    """
    if len(history) < REBURN_MIN_BARS:
        return False
    values = reburn_rsi(history).dropna()
    if len(values) < 2:
        return False
    current = float(values.iloc[-1])
    previous = float(values.iloc[-2])
    return bool(current > REBURN_RSI_TRIGGER and previous <= REBURN_RSI_TRIGGER)


def _ma_down(history: pd.DataFrame, period: int, lookback: int) -> bool:
    if len(history) < period + lookback:
        return False
    ma = history["close"].astype(float).rolling(period, min_periods=period).mean()
    latest = ma.iloc[-1]
    previous = ma.iloc[-1 - lookback]
    if pd.isna(latest) or pd.isna(previous):
        return False
    return bool(latest < previous)





def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
