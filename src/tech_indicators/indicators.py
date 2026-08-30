from __future__ import annotations

from typing import Any

import pandas as pd

from .kline_decision import build_unified_kline_trade_plan


def _last_float(series: pd.Series) -> float | None:
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def compute_indicators(df: pd.DataFrame, position_context: dict[str, object] | None = None) -> dict[str, Any]:
    if df.empty:
        return {"available": False, "warnings": ["无 K 线数据"]}

    data = df.sort_values("trade_date").copy()
    for column in ["open", "high", "low", "close", "vol", "pct_chg", "circ_mv"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    close = data["close"]
    vol = data["vol"] if "vol" in data else pd.Series(dtype="float64")

    for period in [5, 10, 20, 60]:
        data[f"ma{period}"] = close.rolling(period).mean()
    data["vol_ma5_prev"] = vol.shift(1).rolling(5).mean()

    exp_fast = close.ewm(span=12, adjust=False).mean()
    exp_slow = close.ewm(span=26, adjust=False).mean()
    data["macd_dif"] = exp_fast - exp_slow
    data["macd_dea"] = data["macd_dif"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = (data["macd_dif"] - data["macd_dea"]) * 2

    latest = data.iloc[-1]
    ma20 = _last_float(data["ma20"])
    close_latest = float(latest["close"]) if pd.notna(latest["close"]) else None
    vol_ma5 = _last_float(data["vol_ma5_prev"])
    vol_latest = float(latest["vol"]) if "vol" in latest and pd.notna(latest["vol"]) else None
    volume_ratio = (
        vol_latest / vol_ma5
        if vol_latest is not None and vol_ma5 not in (None, 0)
        else None
    )
    bias_ma20 = (
        (close_latest - ma20) / ma20 * 100
        if close_latest is not None and ma20 not in (None, 0)
        else None
    )

    body = None
    upper_shadow_ratio = None
    amplitude = None
    if all(pd.notna(latest.get(col)) for col in ["open", "close", "high", "low"]):
        open_ = float(latest["open"])
        high = float(latest["high"])
        low = float(latest["low"])
        close_ = float(latest["close"])
        body = abs(close_ - open_)
        upper_shadow = max(0.0, high - max(open_, close_))
        amplitude = max(0.0, high - low)
        upper_shadow_ratio = upper_shadow / amplitude if amplitude > 0 else None

    indicators: dict[str, Any] = {
        "available": True,
        "history_days": int(len(data)),
        "trade_date": str(latest["trade_date"]),
        "close": _round(close_latest),
        "pct_chg": _round(float(latest["pct_chg"]) if "pct_chg" in latest and pd.notna(latest["pct_chg"]) else None),
        "vol": _round(vol_latest),
        "circ_mv": _round(float(latest["circ_mv"]) if "circ_mv" in latest and pd.notna(latest["circ_mv"]) else None),
        "ma5": _round(_last_float(data["ma5"])),
        "ma10": _round(_last_float(data["ma10"])),
        "ma20": _round(ma20),
        "ma60": _round(_last_float(data["ma60"])),
        "vol_ma5": _round(vol_ma5),
        "volume_ratio_calc": _round(volume_ratio),
        "bias_ma20": _round(bias_ma20),
        "macd_dif": _round(_last_float(data["macd_dif"])),
        "macd_dea": _round(_last_float(data["macd_dea"])),
        "macd_hist": _round(_last_float(data["macd_hist"])),
        "is_yang": bool(latest["close"] > latest["open"]) if pd.notna(latest.get("close")) and pd.notna(latest.get("open")) else None,
        "upper_shadow_ratio": _round(upper_shadow_ratio),
        "body": _round(body),
        "amplitude": _round(amplitude),
        "ma_bullish": _ma_bullish(data),
        "near_ma5": _near_ma(close_latest, _last_float(data["ma5"]), 1.0),
        "near_ma10": _near_ma(close_latest, _last_float(data["ma10"]), 2.0),
        "recent_high_20": _round(float(data.tail(20)["high"].max()) if len(data) >= 20 else None),
        "recent_low_20": _round(float(data.tail(20)["low"].min()) if len(data) >= 20 else None),
        "structure_rising": check_structure_rising(data, 40),
        "macd_bottom_divergence": check_macd_bottom_divergence(data, 40),
        "breakout_ma": check_breakout_ma(data, ["ma20", "ma60"]),
        "consolidation_box": check_consolidation_box(data),
        "close_breakout_box": check_close_breakout_box(data),
        "breakout_volume": check_volume_ratio_between(data, 1.3, 3.5),
        "ideal_volume_ratio": check_volume_ratio_between(data, 1.6, 2.8),
        "trend_not_down_or_macd_divergence": check_trend_not_down_or_macd_divergence(data),
        "close_above_ma20": check_close_above_ma(data, "ma20"),
        "pre_box_drawdown": check_pre_box_drawdown(data),
        "box_low_not_breaking": check_box_low_not_breaking(data),
        "ma20_not_falling": check_ma_slope_not_falling(data, "ma20"),
        "box_volume_shrink": check_box_volume_shrink(data),
        "ma_convergence": check_ma_convergence(data),
        "breakout_candle_quality": check_breakout_candle_quality(data),
        "relative_low_position": check_relative_low_position(data),
        "upside_space": check_upside_space(data),
        "survival_pattern": check_survival_pattern(data),
        "suspected_controlled_range": check_suspected_controlled_range(data),
        "recent_limit_up": check_recent_limit_up(data),
        "baogongtou_pattern": check_baogongtou_pattern(data),
        "duanxian_auxiliary": check_duanxian_auxiliary_signals(data),
        "golden_bull_channel": check_golden_bull_channel(data),
        "golden_bull_profile": check_golden_bull_profile(data),
        "golden_bull_position_rating": check_golden_bull_position_rating(data, position_context=position_context),
        "_series": data,
    }
    indicators["warnings"] = _indicator_warnings(indicators)
    return indicators


def _ma_bullish(data: pd.DataFrame) -> bool | None:
    values = [_last_float(data[f"ma{period}"]) for period in [5, 10, 20]]
    if any(value is None for value in values):
        return None
    return bool(values[0] > values[1] > values[2])


def _near_ma(close: float | None, ma: float | None, threshold_pct: float) -> bool | None:
    if close is None or ma in (None, 0):
        return None
    return abs(close - ma) / ma * 100 <= threshold_pct


def check_structure_rising(data: pd.DataFrame, lookback: int = 40) -> dict[str, Any]:
    if len(data) < lookback:
        return {"passed": None, "status": "unavailable", "reason": f"历史不足 {lookback} 日"}
    window = data.tail(lookback)
    half = lookback // 2
    early_low = float(window.head(half)["low"].min())
    recent_low = float(window.tail(half)["low"].min())
    return {
        "passed": recent_low > early_low,
        "early_low": _round(early_low),
        "recent_low": _round(recent_low),
    }


def check_macd_bottom_divergence(
    data: pd.DataFrame,
    lookback: int = 40,
    price_epsilon: float = 1e-6,
) -> dict[str, Any]:
    if len(data) < lookback or "macd_dif" not in data:
        return {"passed": None, "status": "unavailable", "reason": f"历史不足 {lookback} 日"}
    window = data.tail(lookback).reset_index(drop=True)
    half = lookback // 2
    early_idx = int(window.head(half)["low"].idxmin())
    recent_idx = int(window.tail(half)["low"].idxmin())
    early_low = float(window.loc[early_idx, "low"])
    recent_low = float(window.loc[recent_idx, "low"])
    early_dif = float(window.loc[early_idx, "macd_dif"])
    recent_dif = float(window.loc[recent_idx, "macd_dif"])
    price_not_higher = recent_low <= early_low + price_epsilon
    macd_improved = recent_dif > early_dif
    passed = price_not_higher and macd_improved
    return {
        "passed": bool(passed),
        "early_low": _round(early_low),
        "recent_low": _round(recent_low),
        "early_dif": _round(early_dif),
        "recent_dif": _round(recent_dif),
        "price_not_higher": bool(price_not_higher),
        "macd_improved": bool(macd_improved),
    }


def check_breakout_ma(data: pd.DataFrame, targets: list[str]) -> dict[str, Any]:
    if len(data) < 2:
        return {"passed": None, "status": "unavailable", "reason": "历史不足 2 日"}
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    hits: list[str] = []
    for target in targets:
        if target not in data or pd.isna(latest[target]) or pd.isna(previous[target]):
            continue
        if previous["close"] <= previous[target] and latest["close"] > latest[target]:
            hits.append(target)
    return {"passed": bool(hits), "targets": hits}


def check_consolidation_box(
    data: pd.DataFrame,
    min_len: int = 5,
    max_len: int = 20,
    max_range_pct: float = 18.0,
) -> dict[str, Any]:
    box = _find_consolidation_box(data, min_len, max_len, max_range_pct)
    if box is None:
        if len(data) < min_len + 1:
            return {"passed": None, "status": "unavailable", "reason": f"history less than {min_len + 1} days"}
        return {"passed": False, "min_len": min_len, "max_len": max_len, "max_range_pct": max_range_pct}
    return {"passed": True, **box}


def check_close_breakout_box(
    data: pd.DataFrame,
    min_len: int = 5,
    max_len: int = 20,
    max_range_pct: float = 18.0,
    min_breakout_pct: float = 0.5,
) -> dict[str, Any]:
    box = _find_consolidation_box(data, min_len, max_len, max_range_pct)
    if box is None:
        if len(data) < min_len + 1:
            return {"passed": None, "status": "unavailable", "reason": f"history less than {min_len + 1} days"}
        return {"passed": False, "reason": "no valid consolidation box"}
    latest_close = _latest_numeric(data, "close")
    if latest_close is None:
        return {"passed": None, "status": "unavailable", "reason": "latest close unavailable"}
    threshold = box["box_high_close"] * (1 + min_breakout_pct / 100)
    passed = latest_close > threshold
    return {
        "passed": bool(passed),
        **box,
        "latest_close": _round(latest_close),
        "breakout_threshold": _round(threshold),
        "breakout_pct": _round((latest_close - box["box_high_close"]) / box["box_high_close"] * 100),
        "min_breakout_pct": min_breakout_pct,
    }


def check_volume_ratio_between(
    data: pd.DataFrame,
    min_ratio: float = 1.3,
    max_ratio: float = 3.5,
    min_len: int = 5,
    max_len: int = 20,
    max_range_pct: float = 18.0,
) -> dict[str, Any]:
    box = _find_consolidation_box(data, min_len, max_len, max_range_pct)
    if box is None:
        if len(data) < min_len + 1:
            return {"passed": None, "status": "unavailable", "reason": f"history less than {min_len + 1} days"}
        return {"passed": False, "reason": "no valid consolidation box"}
    latest_vol = _latest_numeric(data, "vol")
    box_avg_vol = box.get("box_avg_vol")
    if latest_vol is None or box_avg_vol in (None, 0):
        return {"passed": None, "status": "unavailable", "reason": "volume unavailable"}
    ratio = latest_vol / box_avg_vol
    return {
        "passed": bool(min_ratio <= ratio <= max_ratio),
        **box,
        "latest_vol": _round(latest_vol),
        "volume_ratio": _round(ratio),
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
    }


def check_trend_not_down_or_macd_divergence(
    data: pd.DataFrame,
    trend_lookback: int = 60,
    divergence_lookback: int = 60,
) -> dict[str, Any]:
    if len(data) < trend_lookback:
        return {"passed": None, "status": "unavailable", "reason": f"history less than {trend_lookback} days"}
    window = data.tail(trend_lookback).copy()
    half = trend_lookback // 2
    early_low = float(window.head(half)["low"].min())
    recent_low = float(window.tail(half)["low"].min())
    latest_close = _latest_numeric(data, "close")
    latest_ma20 = _last_float(data["ma20"]) if "ma20" in data else None
    previous_ma20 = _last_float(data["ma20"].iloc[:-10]) if "ma20" in data and len(data) > 10 else None
    close_above_ma20 = latest_close is not None and latest_ma20 not in (None, 0) and latest_close > latest_ma20
    low_not_breaking = recent_low >= early_low * 0.98
    ma20_not_falling_hard = latest_ma20 is not None and previous_ma20 not in (None, 0) and latest_ma20 >= previous_ma20 * 0.98
    not_down = bool(close_above_ma20 and (low_not_breaking or ma20_not_falling_hard))
    divergence = check_macd_bottom_divergence(data, divergence_lookback)
    return {
        "passed": bool(not_down or divergence.get("passed")),
        "not_down": not_down,
        "macd_bottom_divergence": divergence,
        "early_low": _round(early_low),
        "recent_low": _round(recent_low),
        "close_above_ma20": bool(close_above_ma20),
        "ma20_not_falling_hard": bool(ma20_not_falling_hard),
    }


def check_close_above_ma(data: pd.DataFrame, target: str = "ma20") -> dict[str, Any]:
    latest_close = _latest_numeric(data, "close")
    ma = _last_float(data[target]) if target in data else None
    if latest_close is None or ma in (None, 0):
        return {"passed": None, "status": "unavailable", "reason": f"{target} unavailable"}
    return {"passed": bool(latest_close > ma), "close": _round(latest_close), target: _round(ma)}


def check_pre_box_drawdown(
    data: pd.DataFrame,
    lookback: int = 30,
    max_drawdown_pct: float = 22.0,
    min_len: int = 5,
    max_len: int = 20,
    max_range_pct: float = 18.0,
) -> dict[str, Any]:
    box = _find_consolidation_box(data, min_len, max_len, max_range_pct)
    if box is None:
        return {"passed": False, "reason": "no valid consolidation box"} if len(data) >= min_len + 1 else {"passed": None, "status": "unavailable", "reason": f"history less than {min_len + 1} days"}
    box_start_index = int(box["box_start_index"])
    pre_start = max(0, box_start_index - lookback)
    pre_window = data.iloc[pre_start:box_start_index]
    if pre_window.empty or pre_window["high"].isna().all():
        return {"passed": None, "status": "unavailable", "reason": "pre-box window unavailable"}
    pre_high = float(pre_window["high"].max())
    box_low = float(box["box_low"])
    if pre_high <= 0:
        return {"passed": None, "status": "unavailable", "reason": "pre-box high unavailable"}
    drawdown_pct = (pre_high - box_low) / pre_high * 100
    return {
        "passed": bool(drawdown_pct <= max_drawdown_pct),
        **box,
        "lookback": lookback,
        "pre_high": _round(pre_high),
        "drawdown_pct": _round(drawdown_pct),
        "max_drawdown_pct": max_drawdown_pct,
    }


def check_box_low_not_breaking(
    data: pd.DataFrame,
    tolerance_pct: float = 2.0,
    min_len: int = 5,
    max_len: int = 20,
    max_range_pct: float = 18.0,
) -> dict[str, Any]:
    box = _find_consolidation_box(data, min_len, max_len, max_range_pct)
    if box is None:
        return {"passed": False, "reason": "no valid consolidation box"} if len(data) >= min_len + 1 else {"passed": None, "status": "unavailable", "reason": f"history less than {min_len + 1} days"}
    window = data.iloc[int(box["box_start_index"]) : int(box["box_end_index"]) + 1]
    half = len(window) // 2
    early_low = float(window.head(half)["low"].min())
    recent_low = float(window.tail(len(window) - half)["low"].min())
    floor = early_low * (1 - tolerance_pct / 100)
    return {
        "passed": bool(recent_low >= floor),
        **box,
        "early_low": _round(early_low),
        "recent_low": _round(recent_low),
        "tolerance_pct": tolerance_pct,
    }


def check_ma_slope_not_falling(
    data: pd.DataFrame,
    target: str = "ma20",
    lookback: int = 10,
    max_down_slope_pct: float = 2.0,
) -> dict[str, Any]:
    if target not in data or len(data) <= lookback:
        return {"passed": None, "status": "unavailable", "reason": f"{target} history unavailable"}
    latest = _last_float(data[target])
    previous = _last_float(data[target].iloc[:-lookback])
    if latest is None or previous in (None, 0):
        return {"passed": None, "status": "unavailable", "reason": f"{target} unavailable"}
    slope_pct = (latest - previous) / previous * 100
    return {
        "passed": bool(slope_pct >= -max_down_slope_pct),
        "target": target,
        "lookback": lookback,
        "latest": _round(latest),
        "previous": _round(previous),
        "slope_pct": _round(slope_pct),
        "max_down_slope_pct": max_down_slope_pct,
    }


def check_box_volume_shrink(
    data: pd.DataFrame,
    min_len: int = 5,
    max_len: int = 20,
    max_range_pct: float = 18.0,
) -> dict[str, Any]:
    box = _find_consolidation_box(data, min_len, max_len, max_range_pct)
    if box is None:
        return {"passed": False, "reason": "no valid consolidation box"} if len(data) >= min_len + 1 else {"passed": None, "status": "unavailable", "reason": f"history less than {min_len + 1} days"}
    window = data.iloc[-1 - box["box_len"] : -1]
    if "vol" not in window or window["vol"].isna().any():
        return {"passed": None, "status": "unavailable", "reason": "box volume unavailable"}
    half = len(window) // 2
    early_avg = float(window.head(half)["vol"].mean())
    recent_avg = float(window.tail(len(window) - half)["vol"].mean())
    return {
        "passed": bool(recent_avg < early_avg),
        **box,
        "early_avg_vol": _round(early_avg),
        "recent_avg_vol": _round(recent_avg),
    }


def check_ma_convergence(
    data: pd.DataFrame,
    targets: tuple[str, ...] = ("ma5", "ma10", "ma20"),
    max_spread_pct: float = 8.0,
) -> dict[str, Any]:
    values = {target: _last_float(data[target]) if target in data else None for target in targets}
    numeric_values = [value for value in values.values() if value is not None]
    if len(numeric_values) != len(targets):
        return {"passed": None, "status": "unavailable", "reason": "moving average unavailable", "values": values}
    anchor = _latest_numeric(data, "close") or numeric_values[-1]
    if anchor == 0:
        return {"passed": None, "status": "unavailable", "reason": "anchor is zero"}
    spread_pct = (max(numeric_values) - min(numeric_values)) / anchor * 100
    return {"passed": bool(spread_pct <= max_spread_pct), "spread_pct": _round(spread_pct), "max_spread_pct": max_spread_pct, "values": {k: _round(v) for k, v in values.items()}}


def check_breakout_candle_quality(data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        return {"passed": None, "status": "unavailable", "reason": "empty data"}
    latest = data.iloc[-1]
    required = ["open", "high", "low", "close"]
    if any(pd.isna(latest.get(column)) for column in required):
        return {"passed": None, "status": "unavailable", "reason": "latest OHLC unavailable"}
    open_ = float(latest["open"])
    high = float(latest["high"])
    low = float(latest["low"])
    close = float(latest["close"])
    amplitude = high - low
    body = abs(close - open_)
    if amplitude <= 0 or body <= 0:
        return {"passed": False, "reason": "no effective candle body"}
    close_location = (close - low) / amplitude
    upper_shadow_to_body = max(0.0, high - max(open_, close)) / body
    pct_chg = float(latest["pct_chg"]) if "pct_chg" in latest and pd.notna(latest["pct_chg"]) else None
    pct_ok = True if pct_chg is None else 1.0 <= pct_chg <= 9.8
    passed = close > open_ and close_location >= 0.65 and upper_shadow_to_body <= 1.2 and pct_ok
    return {
        "passed": bool(passed),
        "close_location": _round(close_location),
        "upper_shadow_to_body": _round(upper_shadow_to_body),
        "pct_chg": _round(pct_chg),
    }


def check_relative_low_position(
    data: pd.DataFrame,
    lookback: int = 60,
    max_gain_from_low_pct: float = 35.0,
) -> dict[str, Any]:
    if len(data) < lookback:
        return {"passed": None, "status": "unavailable", "reason": f"history less than {lookback} days"}
    latest_close = _latest_numeric(data, "close")
    recent_low = float(data.tail(lookback)["low"].min())
    if latest_close is None or recent_low <= 0:
        return {"passed": None, "status": "unavailable", "reason": "price unavailable"}
    gain_pct = (latest_close - recent_low) / recent_low * 100
    return {"passed": bool(gain_pct <= max_gain_from_low_pct), "gain_from_low_pct": _round(gain_pct), "recent_low": _round(recent_low), "max_gain_from_low_pct": max_gain_from_low_pct}


def check_upside_space(
    data: pd.DataFrame,
    lookback: int = 120,
    min_space_pct: float = 8.0,
) -> dict[str, Any]:
    if len(data) < lookback:
        return {"passed": None, "status": "unavailable", "reason": f"history less than {lookback} days"}
    latest_close = _latest_numeric(data, "close")
    recent_high = float(data.tail(lookback)["high"].max())
    if latest_close in (None, 0):
        return {"passed": None, "status": "unavailable", "reason": "latest close unavailable"}
    space_pct = (recent_high - latest_close) / latest_close * 100
    return {"passed": bool(space_pct >= min_space_pct), "space_pct": _round(space_pct), "recent_high": _round(recent_high), "min_space_pct": min_space_pct}


def _latest_numeric(data: pd.DataFrame, column: str) -> float | None:
    if column not in data or data.empty or pd.isna(data.iloc[-1].get(column)):
        return None
    return float(data.iloc[-1][column])


def _find_consolidation_box(
    data: pd.DataFrame,
    min_len: int,
    max_len: int,
    max_range_pct: float,
) -> dict[str, Any] | None:
    required = ["high", "low", "close"]
    if any(column not in data for column in required) or len(data) < min_len + 1:
        return None
    best: dict[str, Any] | None = None
    for length in range(min(max_len, len(data) - 1), min_len - 1, -1):
        start_pos = len(data) - 1 - length
        end_pos = len(data) - 2
        window = data.iloc[start_pos : end_pos + 1]
        if window[required].isna().any().any():
            continue
        box_low = float(window["low"].min())
        box_high = float(window["high"].max())
        if box_low <= 0:
            continue
        range_pct = (box_high - box_low) / box_low * 100
        if range_pct > max_range_pct:
            continue
        box_high_close = float(window["close"].max())
        box: dict[str, Any] = {
            "box_len": length,
            "box_start_index": int(start_pos),
            "box_end_index": int(end_pos),
            "box_start_trade_date": str(window.iloc[0].get("trade_date")) if "trade_date" in window else None,
            "box_end_trade_date": str(window.iloc[-1].get("trade_date")) if "trade_date" in window else None,
            "box_low": _round(box_low),
            "box_high": _round(box_high),
            "box_high_close": _round(box_high_close),
            "box_range_pct": _round(range_pct),
            "max_range_pct": max_range_pct,
        }
        if "vol" in window and not window["vol"].isna().all():
            box["box_avg_vol"] = _round(float(window["vol"].mean()))
        best = box
        break
    return best


def check_survival_pattern(
    data: pd.DataFrame,
    min_len: int = 3,
    max_len: int = 7,
    threshold: float = 3.0,
) -> dict[str, Any]:
    required = ["trade_date", "open", "close", "pct_chg"]
    missing = [column for column in required if column not in data]
    if missing:
        return {
            "passed": None,
            "status": "unavailable",
            "reason": f"missing columns: {', '.join(missing)}",
        }
    if len(data) < min_len:
        return {"passed": None, "status": "unavailable", "reason": f"history less than {min_len} days"}

    max_window = min(max_len, len(data))
    for length in range(max_window, min_len - 1, -1):
        window = data.tail(length).reset_index(drop=True)
        if window[["open", "close", "pct_chg"]].isna().any().any():
            continue

        first = window.iloc[0]
        latest = window.iloc[-1]
        middle = window.iloc[1:-1]
        first_pct_chg = float(first["pct_chg"])
        latest_pct_chg = float(latest["pct_chg"])
        first_open = float(first["open"])
        latest_close = float(latest["close"])
        middle_ok = bool(middle.empty or middle["pct_chg"].abs().le(threshold).all())

        if (
            first_pct_chg <= -threshold
            and latest_pct_chg >= threshold
            and middle_ok
            and latest_close >= first_open
        ):
            return {
                "passed": True,
                "length": length,
                "start_trade_date": str(first["trade_date"]),
                "end_trade_date": str(latest["trade_date"]),
                "first_pct_chg": _round(first_pct_chg),
                "latest_pct_chg": _round(latest_pct_chg),
                "first_open": _round(first_open),
                "latest_close": _round(latest_close),
            }

    return {"passed": False, "min_len": min_len, "max_len": max_window, "threshold": threshold}


def check_suspected_controlled_range(
    data: pd.DataFrame,
    lookback: int = 120,
    range_threshold_pct: float = 15.0,
    shadow_day_ratio: float = 0.45,
    max_body_ratio: float = 0.45,
    min_shadow_ratio: float = 0.55,
) -> dict[str, Any]:
    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in data]
    if missing:
        return {"passed": None, "status": "unavailable", "reason": f"missing columns: {', '.join(missing)}"}
    if len(data) < lookback:
        return {"passed": None, "status": "unavailable", "reason": f"history less than {lookback} days"}

    window = data.tail(lookback).copy()
    if window[required].isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "window contains missing OHLC values"}

    min_close = float(window["close"].min())
    max_close = float(window["close"].max())
    if min_close <= 0:
        return {"passed": None, "status": "unavailable", "reason": "min close is not positive"}
    range_pct = (max_close - min_close) / min_close * 100

    amplitude = window["high"] - window["low"]
    valid = amplitude > 0
    body_ratio = (window["close"] - window["open"]).abs() / amplitude.where(valid)
    upper_shadow = window["high"] - window[["open", "close"]].max(axis=1)
    lower_shadow = window[["open", "close"]].min(axis=1) - window["low"]
    shadow_ratio = (upper_shadow.clip(lower=0) + lower_shadow.clip(lower=0)) / amplitude.where(valid)
    shadow_heavy = valid & body_ratio.le(max_body_ratio) & shadow_ratio.ge(min_shadow_ratio)
    valid_days = int(valid.sum())
    heavy_days = int(shadow_heavy.sum())
    heavy_ratio = heavy_days / valid_days if valid_days else 0.0

    return {
        "passed": bool(range_pct <= range_threshold_pct and heavy_ratio >= shadow_day_ratio),
        "lookback": lookback,
        "range_pct": _round(range_pct),
        "shadow_heavy_days": heavy_days,
        "valid_days": valid_days,
        "shadow_heavy_ratio": _round(heavy_ratio),
        "range_threshold_pct": range_threshold_pct,
        "shadow_day_ratio": shadow_day_ratio,
    }


def check_recent_limit_up(
    data: pd.DataFrame,
    lookback: int = 10,
    limit_up_pct: float = 9.8,
) -> dict[str, Any]:
    if "pct_chg" not in data:
        return {"passed": None, "status": "unavailable", "reason": "missing columns: pct_chg"}
    if len(data) < lookback:
        return {"passed": None, "status": "unavailable", "reason": f"history less than {lookback} days"}

    window = data.tail(lookback).copy()
    pct_chg = pd.to_numeric(window["pct_chg"], errors="coerce")
    if pct_chg.isna().all():
        return {"passed": None, "status": "unavailable", "reason": "window contains no pct_chg values"}
    thresholds = window.apply(lambda row: _limit_up_threshold(row, limit_up_pct), axis=1)
    hits = window[pct_chg.ge(thresholds)]
    return {
        "passed": bool(len(hits) >= 1),
        "lookback": lookback,
        "limit_up_pct": limit_up_pct,
        "limit_up_count": int(len(hits)),
        "limit_up_dates": [str(value) for value in hits.get("trade_date", pd.Series(dtype="object")).tolist()],
    }


def check_baogongtou_pattern(
    data: pd.DataFrame,
    lookback: int = 10,
    min_limit_up_count: int = 2,
    limit_up_pct: float = 9.8,
    max_window_gain_pct: float = 15.0,
) -> dict[str, Any]:
    required = ["open", "close", "pct_chg"]
    missing = [column for column in required if column not in data]
    if missing:
        return {"passed": None, "status": "unavailable", "reason": f"missing columns: {', '.join(missing)}"}
    if len(data) < lookback:
        return {"passed": None, "status": "unavailable", "reason": f"history less than {lookback} days"}

    window = data.tail(lookback).copy()
    if window[required].isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "window contains missing open/close/pct_chg values"}

    pct_chg = pd.to_numeric(window["pct_chg"], errors="coerce")
    thresholds = window.apply(lambda row: _limit_up_threshold(row, limit_up_pct), axis=1)
    limit_up_hits = window[pct_chg.ge(thresholds)]
    first_open = float(window.iloc[0]["open"])
    latest_close = float(window.iloc[-1]["close"])
    if first_open <= 0:
        return {"passed": None, "status": "unavailable", "reason": "first open is not positive"}
    window_gain_pct = (latest_close - first_open) / first_open * 100
    return {
        "passed": bool(len(limit_up_hits) >= min_limit_up_count and window_gain_pct <= max_window_gain_pct),
        "lookback": lookback,
        "limit_up_pct": limit_up_pct,
        "limit_up_count": int(len(limit_up_hits)),
        "limit_up_dates": [str(value) for value in limit_up_hits.get("trade_date", pd.Series(dtype="object")).tolist()],
        "window_gain_pct": _round(window_gain_pct),
        "max_window_gain_pct": max_window_gain_pct,
    }


def _limit_up_threshold(row: pd.Series, default_pct: float = 9.8) -> float:
    code = _row_stock_code(row)
    if code is None:
        return default_pct
    if code.endswith(".BJ") or code.startswith(("4", "8")):
        return 29.8
    if code.startswith(("300", "301", "688", "689")):
        return 19.8
    return default_pct


def _row_stock_code(row: pd.Series) -> str | None:
    for column in ("ts_code", "code", "symbol"):
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column]).strip().upper()
        if value:
            return value
    return None


def check_duanxian_auxiliary_signals(data: pd.DataFrame) -> dict[str, Any]:
    required = ["open", "high", "low", "close", "vol"]
    missing = [column for column in required if column not in data]
    if missing:
        return {"passed": None, "status": "unavailable", "reason": f"missing columns: {', '.join(missing)}"}
    if len(data) < 25:
        return {"passed": None, "status": "unavailable", "reason": "history less than 25 days"}

    frame = data.sort_values("trade_date").copy() if "trade_date" in data else data.copy()
    for column in required + ["pct_chg"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required].tail(5).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "recent OHLCV unavailable"}

    for period in [5, 10, 20]:
        frame[f"ma{period}"] = frame["close"].rolling(period, min_periods=period).mean()
        frame[f"vol_ma{period}"] = frame["vol"].rolling(period, min_periods=period).mean()
    exp_fast = frame["close"].ewm(span=12, adjust=False).mean()
    exp_slow = frame["close"].ewm(span=26, adjust=False).mean()
    frame["macd_dif"] = exp_fast - exp_slow
    frame["macd_dea"] = frame["macd_dif"].ewm(span=9, adjust=False).mean()
    frame["obv"] = _obv(frame)
    frame["obv_ma100"] = frame["obv"].rolling(100, min_periods=20).mean()

    signals = {
        "price_support_triangle": _duanxian_price_support_triangle(frame),
        "volume_support_triangle": _duanxian_volume_support_triangle(frame),
        "bullish_sandwich": _duanxian_bullish_sandwich(frame),
        "bearish_sandwich": _duanxian_bearish_sandwich(frame),
        "triple_dead_cross_top": _duanxian_triple_dead_cross_top(frame),
        "obv_golden_cross": _duanxian_obv_cross(frame, "golden"),
        "obv_dead_cross": _duanxian_obv_cross(frame, "dead"),
        "ant_channel_hold": _duanxian_ant_channel_hold(frame),
        "three_line_stop": _duanxian_three_line_stop(frame),
        "sesame_volume": _duanxian_sesame_volume(frame),
    }
    bullish = [
        key
        for key in [
            "price_support_triangle",
            "volume_support_triangle",
            "bullish_sandwich",
            "obv_golden_cross",
            "ant_channel_hold",
            "sesame_volume",
        ]
        if signals[key].get("passed") is True
    ]
    bearish = [
        key
        for key in [
            "bearish_sandwich",
            "triple_dead_cross_top",
            "obv_dead_cross",
            "three_line_stop",
        ]
        if signals[key].get("passed") is True
    ]
    return {
        "passed": bool(bullish or bearish),
        "bullish_signals": bullish,
        "bearish_signals": bearish,
        "signals": signals,
    }


def _duanxian_price_support_triangle(frame: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    if len(frame) < lookback + 20 or frame[["ma5", "ma10", "ma20"]].tail(1).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "MA5/MA10/MA20 unavailable"}
    latest = frame.iloc[-1]
    window = frame.tail(lookback + 1)
    had_bearish_order = bool((window["ma20"] > window["ma10"]).any() and (window["ma10"] > window["ma5"]).any())
    latest_bullish_order = bool(latest["ma5"] > latest["ma10"] > latest["ma20"])
    close_above_support = bool(latest["close"] > max(latest["ma5"], latest["ma10"], latest["ma20"]))
    spread_pct = (max(latest["ma5"], latest["ma10"], latest["ma20"]) - min(latest["ma5"], latest["ma10"], latest["ma20"])) / latest["close"] * 100
    ma5_cross_ma10 = bool(_cross(window["ma5"], window["ma10"]).tail(lookback).any())
    ma10_cross_ma20 = bool(_cross(window["ma10"], window["ma20"]).tail(lookback).any())
    passed = had_bearish_order and latest_bullish_order and close_above_support and ma5_cross_ma10 and ma10_cross_ma20
    return {
        "passed": bool(passed),
        "lookback": lookback,
        "had_bearish_order": had_bearish_order,
        "latest_bullish_order": latest_bullish_order,
        "close_above_support": close_above_support,
        "ma_spread_pct": _round(spread_pct),
        "ma5_cross_ma10": ma5_cross_ma10,
        "ma10_cross_ma20": ma10_cross_ma20,
    }


def _duanxian_volume_support_triangle(frame: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    if len(frame) < lookback + 20 or frame[["vol_ma5", "vol_ma10", "vol_ma20"]].tail(1).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "VOL MA5/10/20 unavailable"}
    latest = frame.iloc[-1]
    window = frame.tail(lookback + 1)
    had_bearish_order = bool((window["vol_ma20"] > window["vol_ma10"]).any() and (window["vol_ma10"] > window["vol_ma5"]).any())
    latest_bullish_order = bool(latest["vol_ma5"] > latest["vol_ma10"] > latest["vol_ma20"])
    ma5_cross_ma10 = bool(_cross(window["vol_ma5"], window["vol_ma10"]).tail(lookback).any())
    ma5_cross_ma20 = bool(_cross(window["vol_ma5"], window["vol_ma20"]).tail(lookback).any())
    ma10_cross_ma20 = bool(_cross(window["vol_ma10"], window["vol_ma20"]).tail(lookback).any())
    passed = had_bearish_order and latest_bullish_order and ma5_cross_ma10 and ma5_cross_ma20 and ma10_cross_ma20
    return {
        "passed": bool(passed),
        "lookback": lookback,
        "had_bearish_order": had_bearish_order,
        "latest_bullish_order": latest_bullish_order,
        "ma5_cross_ma10": ma5_cross_ma10,
        "ma5_cross_ma20": ma5_cross_ma20,
        "ma10_cross_ma20": ma10_cross_ma20,
    }


def _duanxian_bullish_sandwich(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 6:
        return {"passed": None, "status": "unavailable", "reason": "history less than 6 days"}
    a, b, c = frame.tail(3).itertuples(index=False)
    avg_vol = float(frame["vol"].iloc[-8:-3].mean()) if len(frame) >= 8 else float(frame["vol"].iloc[:-3].mean())
    first_yang = float(a.close) > float(a.open)
    middle_rest = float(b.close) <= float(b.open) or abs(float(b.close) - float(b.open)) <= abs(float(a.close) - float(a.open)) * 0.5
    third_yang = float(c.close) > float(c.open)
    reclaim_first = float(c.close) >= max(float(a.open), float(a.close))
    volume_confirm = pd.notna(avg_vol) and avg_vol > 0 and float(c.vol) >= avg_vol * 1.1
    return {
        "passed": bool(first_yang and middle_rest and third_yang and reclaim_first and volume_confirm),
        "first_yang": first_yang,
        "middle_rest": middle_rest,
        "third_yang": third_yang,
        "reclaim_first": reclaim_first,
        "volume_confirm": volume_confirm,
        "latest_volume_ratio": _round(float(c.vol) / avg_vol if avg_vol else None),
    }


def _duanxian_bearish_sandwich(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 6:
        return {"passed": None, "status": "unavailable", "reason": "history less than 6 days"}
    a, b, c = frame.tail(3).itertuples(index=False)
    avg_vol = float(frame["vol"].iloc[-8:-3].mean()) if len(frame) >= 8 else float(frame["vol"].iloc[:-3].mean())
    first_yin = float(a.close) < float(a.open)
    middle_rebound = float(b.close) >= float(b.open)
    third_yin = float(c.close) < float(c.open)
    engulf = float(c.close) <= min(float(a.open), float(a.close))
    volume_confirm = pd.notna(avg_vol) and avg_vol > 0 and max(float(a.vol), float(c.vol)) >= avg_vol * 1.2
    return {
        "passed": bool(first_yin and middle_rebound and third_yin and engulf and volume_confirm),
        "first_yin": first_yin,
        "middle_rebound": middle_rebound,
        "third_yin": third_yin,
        "engulf": engulf,
        "volume_confirm": volume_confirm,
        "latest_volume_ratio": _round(float(c.vol) / avg_vol if avg_vol else None),
    }


def _duanxian_triple_dead_cross_top(frame: pd.DataFrame, lookback: int = 5) -> dict[str, Any]:
    if len(frame) < 30:
        return {"passed": None, "status": "unavailable", "reason": "history less than 30 days"}
    window = frame.tail(lookback + 1)
    price_dead = bool(_cross(window["ma10"], window["ma5"]).tail(lookback).any())
    volume_dead = bool(_cross(window["vol_ma10"], window["vol_ma5"]).tail(lookback).any())
    macd_dead = bool(_cross(window["macd_dea"], window["macd_dif"]).tail(lookback).any())
    high_position = _relative_position(frame["close"], 60)
    return {
        "passed": bool(price_dead and volume_dead and macd_dead and (high_position is None or high_position >= 0.55)),
        "price_dead_cross": price_dead,
        "volume_dead_cross": volume_dead,
        "macd_dead_cross": macd_dead,
        "relative_position_60": _round(high_position),
    }


def _duanxian_obv_cross(frame: pd.DataFrame, direction: str) -> dict[str, Any]:
    if len(frame) < 30 or frame[["obv", "obv_ma100"]].tail(2).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "OBV history unavailable"}
    if direction == "golden":
        passed = bool(_cross(frame["obv"].tail(2), frame["obv_ma100"].tail(2)).iloc[-1])
    else:
        passed = bool(_cross(frame["obv_ma100"].tail(2), frame["obv"].tail(2)).iloc[-1])
    return {
        "passed": passed,
        "direction": direction,
        "obv": _round(_last_float(frame["obv"])),
        "obv_ma100": _round(_last_float(frame["obv_ma100"])),
    }


def _duanxian_ant_channel_hold(frame: pd.DataFrame, lookback: int = 5) -> dict[str, Any]:
    if len(frame) < 20 or frame[["ma5", "ma10"]].tail(lookback).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "MA5/MA10 unavailable"}
    window = frame.tail(lookback)
    above_ma = bool((window["close"] > window[["ma5", "ma10"]].max(axis=1)).all())
    channel_open = bool((window["ma5"] > window["ma10"]).all())
    pct_changes = frame["pct_chg"] if "pct_chg" in frame else frame["close"].pct_change() * 100
    recent_pct = pct_changes.tail(lookback)
    gentle_steps = bool(recent_pct.abs().max() <= 3.5 and recent_pct.clip(lower=0).mean() <= 1.8)
    no_surge = bool(recent_pct.max() <= 4.5 and (recent_pct >= 3.0).sum() <= 1)
    latest_gap_pct = (float(window.iloc[-1]["ma5"]) - float(window.iloc[-1]["ma10"])) / float(window.iloc[-1]["close"]) * 100
    latest_ma5_distance_pct = (
        (float(window.iloc[-1]["close"]) - float(window.iloc[-1]["ma5"])) / float(window.iloc[-1]["close"]) * 100
    )
    return {
        "passed": bool(above_ma and channel_open and latest_gap_pct >= 0.2 and gentle_steps and no_surge and latest_ma5_distance_pct <= 5),
        "lookback": lookback,
        "above_ma": above_ma,
        "channel_open": channel_open,
        "gentle_steps": gentle_steps,
        "no_surge": no_surge,
        "latest_gap_pct": _round(latest_gap_pct),
        "latest_ma5_distance_pct": _round(latest_ma5_distance_pct),
        "max_recent_pct_chg": _round(float(recent_pct.max())),
    }


def _duanxian_three_line_stop(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 25 or frame[["ma5", "ma10", "ma20"]].tail(6).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "MA5/MA10/MA20 unavailable"}
    latest = frame.iloc[-1]
    close = float(latest["close"])
    broken = [target for target in ["ma5", "ma10", "ma20"] if close < float(latest[target])]
    previous = frame.tail(6).iloc[:-1]
    lines = previous[["ma5", "ma10", "ma20"]]
    above_line_counts = (previous["close"].to_numpy()[:, None] >= lines.to_numpy()).sum(axis=1)
    had_hold_context = bool((above_line_counts >= 2).sum() >= 3)
    newly_broke_ma20 = bool(close < float(latest["ma20"]) and float(frame.iloc[-2]["close"]) >= float(frame.iloc[-2]["ma20"]))
    stop_triggered = bool(had_hold_context and (len(broken) >= 2 or newly_broke_ma20))
    return {
        "passed": stop_triggered,
        "broken_lines": broken,
        "line_break_count": len(broken),
        "had_hold_context": had_hold_context,
        "newly_broke_ma20": newly_broke_ma20,
        "suggested_reduce_fraction": _round(len(broken) / 3),
    }


def _duanxian_sesame_volume(frame: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    if len(frame) < lookback + 20 or frame[["ma20", "vol_ma20"]].tail(1).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "MA20/VOL MA20 unavailable"}
    latest = frame.iloc[-1]
    close = float(latest["close"])
    ma20 = float(latest["ma20"])
    vol = float(latest["vol"])
    vol_ma20 = float(latest["vol_ma20"])
    recent_volume = frame["vol"].tail(lookback)
    low_quantile = float(recent_volume.quantile(0.2))
    volume_shrunk = bool(vol <= vol_ma20 * 0.55 and vol <= low_quantile)
    not_broken = bool(close >= ma20 * 0.98)
    pct_chg = float(latest["pct_chg"]) if "pct_chg" in frame and pd.notna(latest["pct_chg"]) else None
    calm_price = bool(pct_chg is None or abs(pct_chg) <= 3.0)
    return {
        "passed": bool(volume_shrunk and not_broken and calm_price),
        "lookback": lookback,
        "volume_shrunk": volume_shrunk,
        "not_broken": not_broken,
        "calm_price": calm_price,
        "volume_ratio_ma20": _round(vol / vol_ma20 if vol_ma20 else None),
        "volume_quantile20": _round(low_quantile),
        "close_ma20_distance_pct": _round((close - ma20) / ma20 * 100 if ma20 else None),
    }


def _obv(frame: pd.DataFrame) -> pd.Series:
    direction = frame["close"].diff().apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
    direction.iloc[0] = 0
    return (direction * frame["vol"]).cumsum()


def _relative_position(series: pd.Series, lookback: int) -> float | None:
    if len(series) < lookback:
        return None
    window = pd.to_numeric(series.tail(lookback), errors="coerce")
    low = float(window.min())
    high = float(window.max())
    latest = float(window.iloc[-1])
    if high <= low:
        return None
    return (latest - low) / (high - low)


def _indicator_warnings(indicators: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key in ["ma20", "ma60", "volume_ratio_calc", "bias_ma20", "macd_dif", "circ_mv"]:
        if indicators.get(key) is None:
            warnings.append(f"{key} 无法计算")
    for key in [
        "structure_rising",
        "macd_bottom_divergence",
        "consolidation_box",
        "close_breakout_box",
        "breakout_volume",
        "trend_not_down_or_macd_divergence",
        "close_above_ma20",
        "pre_box_drawdown",
        "box_low_not_breaking",
        "ma20_not_falling",
        "box_volume_shrink",
        "ma_convergence",
        "breakout_candle_quality",
        "relative_low_position",
        "upside_space",
        "survival_pattern",
        "suspected_controlled_range",
        "recent_limit_up",
        "baogongtou_pattern",
        "duanxian_auxiliary",
        "golden_bull_channel",
        "golden_bull_profile",
        "golden_bull_position_rating",
    ]:
        value = indicators.get(key) or {}
        if value.get("passed") is None:
            warnings.append(f"{key} 无法验证")
    return warnings


def check_golden_bull_channel(data: pd.DataFrame) -> dict[str, Any]:
    required = ["open", "high", "low", "close", "vol"]
    missing = [column for column in required if column not in data]
    if missing:
        return {"passed": None, "status": "unavailable", "reason": f"missing columns: {', '.join(missing)}"}
    if len(data) < 80:
        return {"passed": None, "status": "unavailable", "reason": "history less than 80 days"}

    frame = data.sort_values("trade_date").copy() if "trade_date" in data else data.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required].tail(2).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "latest OHLCV unavailable"}

    channel_lines = compute_golden_bull_lines(frame)
    golden_bull = channel_lines["golden_bull"]
    golden_bull_trend = channel_lines["golden_bull_trend"]
    golden_bull_2 = channel_lines["golden_bull_2"]

    close_delta = frame["close"] - frame["close"].shift(1)
    denominator = _tdx_xma(_tdx_xma(close_delta.abs(), 6), 6)
    var23 = 100 * _tdx_xma(_tdx_xma(close_delta, 6), 6) / denominator.replace(0, pd.NA)
    pullback_buy = (
        _llv(var23, 2).eq(_llv(var23, 7))
        & _count(var23.lt(0), 2).gt(0)
        & _cross(var23, var23.rolling(2, min_periods=2).mean())
    )

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    idx = frame.index[-1]
    trend = golden_bull_trend.loc[idx]
    bull = golden_bull.loc[idx]

    buy_signal = bool(
        pd.notna(trend)
        and trend > float(latest["high"])
        and bool(pullback_buy.loc[idx])
        and float(latest["low"]) <= trend
    )

    ddx = _golden_bull_ddx(frame)
    v2_input = ddx.where(frame["close"].ge(frame["close"].shift(1)), -ddx / 100)
    v2 = _tdx_sma(v2_input, 2, 1)
    dy = bool(float(latest["close"]) < float(previous["close"]))
    dy2 = v2.shift(1).loc[idx] - (1 if dy else 0)
    ma5 = frame["close"].rolling(5, min_periods=5).mean().loc[idx]
    ma60 = frame["close"].rolling(60, min_periods=60).mean().loc[idx]
    rise_signal = bool(
        pd.notna(trend)
        and pd.notna(bull)
        and pd.notna(dy2)
        and pd.notna(ma5)
        and pd.notna(ma60)
        and float(latest["close"]) > float(latest["open"])
        and dy2 < 0.02
        and ma5 > ma60
        and float(previous["close"]) > 0
        and float(latest["close"]) / float(previous["close"]) >= 1.02
        and float(latest["high"]) < bull
        and float(latest["low"]) < trend
    )

    return {
        "passed": buy_signal or rise_signal,
        "buy_signal": buy_signal,
        "rise_signal": rise_signal,
        "trade_date": str(latest.get("trade_date")) if "trade_date" in latest else None,
        "golden_bull": _round(float(bull) if pd.notna(bull) else None),
        "golden_bull_trend": _round(float(trend) if pd.notna(trend) else None),
        "golden_bull_2": _round(float(golden_bull_2.loc[idx]) if pd.notna(golden_bull_2.loc[idx]) else None),
        "var23": _round(float(var23.loc[idx]) if pd.notna(var23.loc[idx]) else None),
        "dy2": _round(float(dy2) if pd.notna(dy2) else None),
        "xma_note": "TDX XMA is a future function; recent missing future bars are padded with the known-window mean.",
    }


def compute_golden_bull_lines(data: pd.DataFrame) -> pd.DataFrame:
    required = ["high", "low"]
    missing = [column for column in required if column not in data]
    if missing:
        raise KeyError(f"missing columns: {', '.join(missing)}")

    frame = data.sort_values("trade_date").copy() if "trade_date" in data else data.copy()
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    high_xma = _tdx_xma(_tdx_xma(high, 25), 25)
    low_xma = _tdx_xma(_tdx_xma(low, 25), 25)
    golden_bull = (high_xma - low_xma) + high_xma
    golden_bull_trend = low_xma - (high_xma - low_xma)
    golden_bull_2 = golden_bull_trend.ewm(span=25, adjust=False).mean()
    lines = pd.DataFrame(
        {
            "golden_bull": golden_bull,
            "golden_bull_trend": golden_bull_trend,
            "golden_bull_2": golden_bull_2,
        },
        index=frame.index,
    )
    lines["channel_upper"] = lines.max(axis=1)
    lines["channel_lower"] = lines.min(axis=1)
    return lines


def check_golden_bull_profile(data: pd.DataFrame) -> dict[str, Any]:
    required = ["open", "high", "low", "close", "vol"]
    missing = [column for column in required if column not in data]
    if missing:
        return {"passed": None, "status": "unavailable", "reason": f"missing columns: {', '.join(missing)}"}
    if len(data) < 80:
        return {"passed": None, "status": "unavailable", "reason": "history less than 80 days"}

    frame = data.sort_values("trade_date").copy() if "trade_date" in data else data.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required].tail(2).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "latest OHLCV unavailable"}

    high_xma = _tdx_xma(_tdx_xma(frame["high"], 25), 25)
    low_xma = _tdx_xma(_tdx_xma(frame["low"], 25), 25)
    upper = (high_xma - low_xma) + high_xma
    lower = low_xma - (high_xma - low_xma)
    life = lower.ewm(span=25, adjust=False).mean()

    idx = frame.index[-1]
    latest = frame.iloc[-1]
    close = float(latest["close"])
    low = float(latest["low"])
    upper_latest = float(upper.loc[idx]) if pd.notna(upper.loc[idx]) else None
    lower_latest = float(lower.loc[idx]) if pd.notna(lower.loc[idx]) else None
    life_latest = float(life.loc[idx]) if pd.notna(life.loc[idx]) else None
    if upper_latest is None or lower_latest is None or life_latest is None:
        return {"passed": None, "status": "unavailable", "reason": "golden bull channel unavailable"}

    life_slope_pct = _line_slope_pct(life, 20)
    lower_slope_pct = _line_slope_pct(lower, 10)
    upper_slope_pct = _line_slope_pct(upper, 10)
    above_life_ratio_40 = _above_line_ratio(frame["close"], life, 40)
    support = max(value for value in [lower_latest, life_latest] if value is not None)
    distance_to_support_pct = (close - support) / support * 100 if support else None
    distance_to_life_pct = (close - life_latest) / life_latest * 100 if life_latest else None
    distance_to_lower_pct = (close - lower_latest) / lower_latest * 100 if lower_latest else None
    distance_to_upper_pct = (upper_latest - close) / close * 100 if close else None
    low_to_life_pct = (low - life_latest) / life_latest * 100 if life_latest else None
    close_below_life_days = _consecutive_close_below(frame["close"], life)

    signal = check_golden_bull_channel(frame)
    bull_context = (
        life_slope_pct is not None
        and lower_slope_pct is not None
        and above_life_ratio_40 is not None
        and life_slope_pct >= -1.0
        and lower_slope_pct >= -2.0
        and (above_life_ratio_40 >= 0.55 or close >= life_latest)
    )
    healthy_channel = (
        lower_slope_pct is not None
        and upper_slope_pct is not None
        and lower_slope_pct >= -1.5
        and upper_slope_pct >= -1.5
        and upper_latest > lower_latest
    )
    near_support = distance_to_support_pct is not None and -3.0 <= distance_to_support_pct <= 6.0
    deep_pullback_opportunity = (
        bull_context
        and bool(signal.get("passed"))
        and (
            near_support
            or (low_to_life_pct is not None and -4.0 <= low_to_life_pct <= 2.0)
            or (distance_to_lower_pct is not None and -3.0 <= distance_to_lower_pct <= 4.0)
        )
    )
    overextended = (
        (distance_to_support_pct is not None and distance_to_support_pct > 12.0)
        or (distance_to_upper_pct is not None and distance_to_upper_pct < 2.0)
    )
    trend_break_risk = (
        close_below_life_days >= 3
        and close < lower_latest
        and life_slope_pct is not None
        and life_slope_pct < -1.0
    )
    bear_context = (
        life_slope_pct is not None
        and lower_slope_pct is not None
        and above_life_ratio_40 is not None
        and life_slope_pct < -2.0
        and lower_slope_pct < -2.0
        and above_life_ratio_40 < 0.35
    )
    volume_momentum = _golden_bull_volume_momentum(frame)

    return {
        "passed": bool(bull_context or deep_pullback_opportunity),
        "bull_context": {
            "passed": bool(bull_context),
            "life_slope_pct": _round(life_slope_pct),
            "lower_slope_pct": _round(lower_slope_pct),
            "above_life_ratio_40": _round(above_life_ratio_40),
        },
        "healthy_channel": {
            "passed": bool(healthy_channel),
            "upper_slope_pct": _round(upper_slope_pct),
            "lower_slope_pct": _round(lower_slope_pct),
        },
        "deep_pullback_opportunity": {
            "passed": bool(deep_pullback_opportunity),
            "low_to_life_pct": _round(low_to_life_pct),
            "distance_to_lower_pct": _round(distance_to_lower_pct),
            "signal": bool(signal.get("passed")),
        },
        "near_support": {
            "passed": bool(near_support),
            "distance_to_support_pct": _round(distance_to_support_pct),
        },
        "volume_momentum": volume_momentum,
        "overextended": {
            "passed": bool(overextended),
            "distance_to_support_pct": _round(distance_to_support_pct),
            "distance_to_upper_pct": _round(distance_to_upper_pct),
        },
        "trend_break_risk": {
            "passed": bool(trend_break_risk),
            "close_below_life_days": close_below_life_days,
            "life_slope_pct": _round(life_slope_pct),
        },
        "bear_context": {
            "passed": bool(bear_context),
            "life_slope_pct": _round(life_slope_pct),
            "lower_slope_pct": _round(lower_slope_pct),
            "above_life_ratio_40": _round(above_life_ratio_40),
        },
        "signal": signal,
        "golden_bull": _round(upper_latest),
        "golden_bull_trend": _round(lower_latest),
        "life_line": _round(life_latest),
        "distance_to_life_pct": _round(distance_to_life_pct),
        "distance_to_support_pct": _round(distance_to_support_pct),
        "distance_to_upper_pct": _round(distance_to_upper_pct),
        "close_below_life_days": close_below_life_days,
        "xma_note": "TDX XMA is a future function; refine scoring treats the signal as a candidate source and ranks with reproducible trend, support, and momentum checks.",
    }


def check_golden_bull_position_rating(
    data: pd.DataFrame,
    *,
    position_context: dict[str, object] | None = None,
) -> dict[str, Any]:
    required = ["open", "high", "low", "close", "vol"]
    missing = [column for column in required if column not in data]
    if missing:
        return {"passed": None, "status": "unavailable", "reason": f"missing columns: {', '.join(missing)}"}
    if len(data) < 120:
        return {"passed": None, "status": "unavailable", "reason": "history less than 120 days"}

    frame = data.sort_values("trade_date").copy() if "trade_date" in data else data.copy()
    for column in required + ["pct_chg"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[required].tail(20).isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "recent OHLCV unavailable"}

    channel_lines = compute_golden_bull_lines(frame)
    upper = channel_lines["golden_bull"]
    lower = channel_lines["golden_bull_trend"]
    golden2 = channel_lines["golden_bull_2"]
    channel_upper = channel_lines["channel_upper"]
    channel_lower = channel_lines["channel_lower"]

    idx = frame.index[-1]
    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    close = float(latest["close"])
    open_ = float(latest["open"])
    high = float(latest["high"])
    low = float(latest["low"])
    latest_volume = float(latest["vol"])
    previous_volume = float(previous["vol"])
    prev_close = float(previous["close"])
    ma20_series = frame["close"].rolling(20, min_periods=20).mean()
    ma60_series = frame["close"].rolling(60, min_periods=60).mean()
    ma20_latest = float(ma20_series.iloc[-1])
    ma20_previous = float(ma20_series.iloc[-2]) if pd.notna(ma20_series.iloc[-2]) else None
    ma20_previous_2 = float(ma20_series.iloc[-3]) if len(ma20_series) >= 3 and pd.notna(ma20_series.iloc[-3]) else None
    ma60_latest = float(ma60_series.iloc[-1])
    ma60_previous = float(ma60_series.iloc[-2]) if pd.notna(ma60_series.iloc[-2]) else None
    ma60_previous_2 = float(ma60_series.iloc[-3]) if len(ma60_series) >= 3 and pd.notna(ma60_series.iloc[-3]) else None
    ma20_slope_pct = (
        (ma20_latest - ma20_previous) / ma20_previous * 100
        if ma20_previous not in (None, 0)
        else None
    )
    ma60_slope_pct = (
        (ma60_latest - ma60_previous) / ma60_previous * 100
        if ma60_previous not in (None, 0)
        else None
    )
    ma_spread_pct = (ma20_latest - ma60_latest) / close * 100 if close else None
    previous_ma_spread_pct = (
        (float(ma20_series.iloc[-2]) - float(ma60_series.iloc[-2])) / prev_close * 100
        if prev_close and pd.notna(ma20_series.iloc[-2]) and pd.notna(ma60_series.iloc[-2])
        else None
    )
    ma20_ma60_spread_widening = (
        ma_spread_pct is not None
        and previous_ma_spread_pct is not None
        and ma20_latest > ma60_latest
        and ma_spread_pct > previous_ma_spread_pct
    )
    upper_latest = float(upper.loc[idx]) if pd.notna(upper.loc[idx]) else None
    lower_latest = float(lower.loc[idx]) if pd.notna(lower.loc[idx]) else None
    golden2_latest = float(golden2.loc[idx]) if pd.notna(golden2.loc[idx]) else None
    channel_upper_latest = float(channel_upper.loc[idx]) if pd.notna(channel_upper.loc[idx]) else None
    channel_lower_latest = float(channel_lower.loc[idx]) if pd.notna(channel_lower.loc[idx]) else None
    if (
        upper_latest is None
        or lower_latest is None
        or golden2_latest is None
        or channel_upper_latest is None
        or channel_lower_latest is None
    ):
        return {"passed": None, "status": "unavailable", "reason": "golden bull channel unavailable"}

    support = max(lower_latest, golden2_latest)
    upper_slope_pct = _line_slope_pct(channel_upper, 10)
    lower_slope_pct = _line_slope_pct(lower, 10)
    golden2_slope_pct = _line_slope_pct(golden2, 20)
    above_support_ratio_40 = _above_line_ratio(frame["close"], golden2, 40)
    channel_width_pct = (channel_upper_latest - channel_lower_latest) / close * 100 if close else None
    prev_width_pct = _channel_width_pct(channel_upper, channel_lower, frame["close"], 20)
    width_change_pct = (
        (channel_width_pct - prev_width_pct) / prev_width_pct * 100
        if channel_width_pct is not None and prev_width_pct not in (None, 0)
        else None
    )
    volume_ratio = _latest_volume_ratio(frame)
    volume_vs_prev_ratio = latest_volume / previous_volume if previous_volume else None
    recent = frame.tail(10)
    recent_gain_pct = (close - float(recent.iloc[0]["close"])) / float(recent.iloc[0]["close"]) * 100
    daily_return_pct = (close - prev_close) / prev_close * 100 if prev_close else None
    avg_abs_return_10_pct = (
        float((frame["close"].pct_change() * 100).abs().iloc[-11:-1].mean())
        if len(frame) >= 12
        else None
    )
    upper_distance_pct = (channel_upper_latest - close) / close * 100 if close else None
    support_distance_pct = (close - support) / support * 100 if support else None
    lower_distance_pct = (close - channel_lower_latest) / channel_lower_latest * 100 if channel_lower_latest else None
    close_below_support_days = _consecutive_close_below(frame["close"], golden2)
    close_above_upper_days = _consecutive_close_above(frame["close"], channel_upper)
    close_below_upper_days = _consecutive_close_below(frame["close"], channel_upper)

    channel_regime = _golden_channel_regime(lower_latest, golden2_latest)
    channel_strength, channel_gap_pct = _golden_channel_strength(lower_latest, golden2_latest, close)
    line_values = {
        "trend_upper": upper_latest,
        "life_line": lower_latest,
        "trend_confirmation_line": golden2_latest,
    }
    upper_source = max(line_values, key=line_values.get)
    lower_source = min(line_values, key=line_values.get)
    channel_regime_evidence = {
        "rule": "trend_confirmation_line_below_life_line_is_bull; life_line_below_trend_confirmation_line_is_bear",
        "life_line": _round(lower_latest),
        "trend_confirmation_line": _round(golden2_latest),
        "gap_pct_of_price": _round(channel_gap_pct),
        "channel_strength": channel_strength,
        "lower_source": lower_source,
        "golden_bull_trend_slope_pct": _round(lower_slope_pct),
        "golden_bull_2_slope_pct": _round(golden2_slope_pct),
        "above_support_ratio_40": _round(above_support_ratio_40),
        "close_above_support": close >= support,
        "bull_context": channel_regime == "bull",
        "bear_context": channel_regime == "bear",
    }

    previous_upper = float(channel_upper.iloc[-2])
    previous_support = max(float(lower.iloc[-2]), float(golden2.iloc[-2]))
    previous_above_upper = bool(frame["close"].tail(20).iloc[:-1].gt(channel_upper.tail(20).iloc[:-1]).any())
    recent_upper_touch = _recent_upper_touch(frame["high"], frame["close"], channel_upper, lookback=6, threshold_pct=2.5)
    close_position = (close - low) / (high - low) if high != low else 1.0
    upper_shadow_state = _golden_upper_shadow_state(
        open_=open_,
        high=high,
        low=low,
        close=close,
        channel_upper=channel_upper_latest,
        volume_ratio=volume_ratio,
        close_position=close_position,
    )
    unconfirmed_sharp_bearish = (
        daily_return_pct is not None
        and daily_return_pct <= -5.0
        and close_position <= 0.35
    )
    fresh_upper_breakout = prev_close <= previous_upper and close > channel_upper_latest
    if fresh_upper_breakout:
        upper_shadow_state = None
    upper_support = previous_above_upper and low <= channel_upper_latest * 1.02 and close >= channel_upper_latest * 0.99
    upper_lost = _golden_upper_lost(
        previous_above_upper=previous_above_upper,
        recent_upper_touch=recent_upper_touch,
        close=close,
        high=high,
        upper_latest=channel_upper_latest,
        close_below_support_days=close_below_support_days,
        upper_distance_pct=upper_distance_pct,
    )
    upper_exhaustion = (
        close >= channel_upper_latest * 0.98
        and recent_gain_pct >= 8.0
        and close < open_
        and (volume_ratio is None or volume_ratio < 1.0)
    )
    lower_support = low <= support * 1.03 and close >= support and close >= open_ * 0.995
    bull_channel_pullback = (
        channel_regime == "bull"
        and previous_above_upper
        and close > support
        and support_distance_pct is not None
        and support_distance_pct <= 12.0
        and upper_distance_pct is not None
        and upper_distance_pct > 6.0
        and recent_gain_pct <= -5.0
        and close_below_support_days == 0
        and (volume_ratio is None or volume_ratio <= 1.2)
    )
    lower_panic = low < channel_lower_latest * 0.94 and close > low * 1.03
    near_upper = upper_distance_pct is not None and -2.0 <= upper_distance_pct <= 3.0
    is_bearish_candle = close < open_
    upper_pressure_bearish = (
        is_bearish_candle
        and (near_upper or high >= channel_upper_latest * 0.98)
    )
    bull_pressure = channel_regime == "bull" and near_upper
    bear_pressure = (
        channel_regime == "bear"
        and not is_bearish_candle
        and (near_upper or high >= channel_upper_latest * 0.98)
    )
    upper_below_pressure = (
        channel_regime == "bull"
        and not upper_support
        and not upper_lost
        and not upper_exhaustion
        and upper_distance_pct is not None
        and 3.0 < upper_distance_pct <= 12.0
        and support_distance_pct is not None
        and support_distance_pct >= 8.0
        and close_below_support_days == 0
    )
    channel_squeeze = (
        channel_width_pct is not None
        and channel_width_pct <= 14.0
        and (width_change_pct is None or width_change_pct <= -12.0)
    )
    upper_walk = _near_upper_ratio(frame["close"], channel_upper, 10, 4.0) >= 0.6 and channel_regime != "bear"
    golden2_lost = close < golden2_latest and close >= lower_latest and not lower_support
    volume_break = _golden_volume_break(
        prev_close=prev_close,
        previous_support=previous_support,
        close=close,
        support=support,
        open_=open_,
        close_position=close_position,
        volume_ratio=volume_ratio,
    )
    below_support_volume_recovery = _golden_below_support_volume_recovery(
        close_below_support_days=close_below_support_days,
        close=close,
        open_=open_,
        prev_close=prev_close,
        high=high,
        support=support,
        volume_ratio=volume_ratio,
    )
    reclaim_channel = _recent_below_then_reclaim(frame["close"], golden2, lookback=12)

    scene_flags = {
        "fresh_upper_breakout": fresh_upper_breakout,
        "upper_support": upper_support,
        "upper_lost": upper_lost,
        "upper_exhaustion": upper_exhaustion,
        "lower_support": lower_support,
        "bull_channel_pullback": bull_channel_pullback,
        "lower_panic": lower_panic,
        "bull_pressure": bull_pressure,
        "bear_pressure": bear_pressure,
        "upper_pressure_bearish": upper_pressure_bearish,
        "upper_below_pressure": upper_below_pressure,
        "channel_squeeze": channel_squeeze,
        "upper_walk": upper_walk,
        "golden2_lost": golden2_lost,
        "volume_break": volume_break,
        "below_support_volume_recovery": below_support_volume_recovery,
        "reclaim_channel": reclaim_channel,
    }
    scenes = _resolve_golden_position_scenes(scene_flags)
    base_ratings = _golden_position_ratings(scenes, channel_regime, breakout_close_position=close_position)
    selling_exhaustion_reversal = _golden_selling_exhaustion_reversal(frame)
    auxiliary_signals = (
        []
        if scenes == ["fresh_upper_breakout"]
        else _golden_auxiliary_signals(
            frame,
            channel_regime,
            channel_strength,
            selling_exhaustion_reversal=selling_exhaustion_reversal,
        )
    )
    ratings = _apply_golden_rating_modifiers(base_ratings, auxiliary_signals)
    ratings = _suppress_unconfirmed_bullish_actions(ratings, unconfirmed_sharp_bearish)
    ratings = _apply_upper_shadow_constraint(ratings, upper_shadow_state)
    ratings = _suppress_upper_pressure_new_positions(ratings, bear_pressure or upper_pressure_bearish)
    clearance_state = _golden_clearance_state(
        is_bearish_candle=is_bearish_candle,
        close=close,
        high=high,
        upper_latest=channel_upper_latest,
        close_below_upper_days=close_below_upper_days,
        recent_upper_touch=recent_upper_touch,
        volume_break=volume_break,
    )
    ratings = _apply_clearance_gate(ratings, clearance_state)
    cost_price = _context_float(position_context, "cost_price")
    profit_pct = (close - cost_price) / cost_price * 100 if cost_price not in (None, 0) else None
    risk_flags = _golden_position_risk_flags(
        scenes,
        channel_regime,
        profit_pct,
        unconfirmed_sharp_bearish,
        upper_shadow_state,
    )

    result = {
        "passed": bool(ratings),
        "trade_date": str(latest.get("trade_date")) if "trade_date" in latest else None,
        "channel_regime": channel_regime,
        "channel_strength": channel_strength,
        "channel_regime_evidence": channel_regime_evidence,
        "auxiliary_signals": auxiliary_signals,
        "position_context": "with_cost" if cost_price is not None else "technical_only",
        "cost_price": _round(cost_price),
        "profit_pct": _round(profit_pct),
        "position_size": (position_context or {}).get("position_size"),
        "buy_date": (position_context or {}).get("buy_date"),
        "notes": (position_context or {}).get("notes"),
        "scenes": scenes,
        "primary_scene": scenes[0] if scenes else "no_clear_signal",
        "ratings": ratings,
        "risk_flags": risk_flags,
        "invalidations": _golden_position_invalidations(scenes),
        "metrics": {
            "golden_bull": _round(upper_latest),
            "golden_bull_trend": _round(lower_latest),
            "golden_bull_2": _round(golden2_latest),
            "ma20": _round(ma20_latest),
            "ma60": _round(ma60_latest),
            "prev_ma20": _round(ma20_previous),
            "prev2_ma20": _round(ma20_previous_2),
            "prev_ma60": _round(ma60_previous),
            "prev2_ma60": _round(ma60_previous_2),
            "ma20_slope_pct": _round(ma20_slope_pct),
            "ma60_slope_pct": _round(ma60_slope_pct),
            "ma20_ma60_spread_pct": _round(ma_spread_pct),
            "prev_ma20_ma60_spread_pct": _round(previous_ma_spread_pct),
            "ma20_ma60_spread_widening": ma20_ma60_spread_widening,
            "upper_line": _round(channel_upper_latest),
            "bull_bear_boundary": _round(lower_latest),
            "lower_line": _round(channel_lower_latest),
            "support": _round(support),
            "distance_to_upper_pct": _round(upper_distance_pct),
            "distance_to_support_pct": _round(support_distance_pct),
            "distance_to_lower_pct": _round(lower_distance_pct),
            "channel_width_pct": _round(channel_width_pct),
            "width_change_pct": _round(width_change_pct),
            "volume": _round(latest_volume),
            "prev_volume": _round(previous_volume),
            "volume_ratio": _round(volume_ratio),
            "volume_vs_prev_ratio": _round(volume_vs_prev_ratio),
            "volume_vs_prev_pct": _round(volume_vs_prev_ratio * 100 if volume_vs_prev_ratio is not None else None),
            "selling_exhaustion_reversal": selling_exhaustion_reversal,
            "recent_gain_pct": _round(recent_gain_pct),
            "daily_return_pct": _round(daily_return_pct),
            "avg_abs_return_10_pct": _round(avg_abs_return_10_pct),
            "close_position": _round(close_position),
            "unconfirmed_sharp_bearish": unconfirmed_sharp_bearish,
            "upper_shadow_state": upper_shadow_state,
            "close_below_support_days": close_below_support_days,
            "close_above_upper_days": close_above_upper_days,
            "close_below_upper_days": close_below_upper_days,
            "is_bearish_candle": is_bearish_candle,
            "clearance_state": clearance_state,
            "upper_slope_pct": _round(upper_slope_pct),
            "lower_slope_pct": _round(lower_slope_pct),
            "golden2_slope_pct": _round(golden2_slope_pct),
            "above_support_ratio_40": _round(above_support_ratio_40),
            "prev_close": _round(prev_close),
            "open": _round(open_),
            "high": _round(high),
            "low": _round(low),
            "close": _round(close),
        },
        "raw_lines": {
            "golden_bull": _round(upper_latest),
            "golden_bull_trend": _round(lower_latest),
            "golden_bull_2": _round(golden2_latest),
        },
        "indicator_lines": {
            "trend_upper": _round(upper_latest),
            "life_line": _round(lower_latest),
            "trend_confirmation_line": _round(golden2_latest),
        },
        "channel_lines": {
            "upper": _round(channel_upper_latest),
            "upper_source": upper_source,
            "bull_bear_boundary": _round(lower_latest),
            "lower": _round(channel_lower_latest),
            "lower_source": lower_source,
            "life_line": _round(lower_latest),
            "trend_confirmation_line": _round(golden2_latest),
        },
        "xma_note": "TDX XMA is a future function; recent missing future bars are padded with the known-window mean for a reproducible right-edge approximation.",
    }
    result["trade_plan"] = build_unified_kline_trade_plan(result, frame, timeframe_label="1d")
    return result


def _tdx_xma(series: pd.Series, period: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    half = period // 2
    raw = values.tolist()
    result: list[float | None] = []
    for index in range(len(raw)):
        start = max(0, index - half)
        end = index + half + 1
        known = [float(value) for value in raw[start : min(end, len(raw))] if pd.notna(value)]
        if not known:
            result.append(None)
            continue
        missing = max(0, end - len(raw))
        if missing:
            pad_value = sum(known) / len(known)
            known = known + [pad_value] * missing
        result.append(sum(known) / len(known))
    return pd.Series(result, index=series.index, dtype="float64")


def _tdx_sma(series: pd.Series, period: int, weight: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result: list[float | None] = []
    previous: float | None = None
    for value in values:
        if pd.isna(value):
            result.append(previous)
            continue
        current = float(value) if previous is None else (weight * float(value) + (period - weight) * previous) / period
        previous = current
        result.append(current)
    return pd.Series(result, index=series.index, dtype="float64")


def _cross(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.gt(right) & left.shift(1).le(right.shift(1))


def _count(condition: pd.Series, period: int) -> pd.Series:
    return condition.fillna(False).astype(int).rolling(period, min_periods=period).sum()


def _llv(series: pd.Series, period: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(period, min_periods=period).min()


def _golden_bull_ddx(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    open_ = frame["open"]
    close = frame["close"]
    jj = (high + low + close) / 3
    qj0 = frame["vol"] / (high - low).where(high.ne(low), 4)
    qj1 = qj0 * (jj - pd.concat([close, open_], axis=1).min(axis=1))
    qj2 = qj0 * (pd.concat([open_, close], axis=1).min(axis=1) - low)
    qj3 = qj0 * (high - pd.concat([open_, close], axis=1).max(axis=1))
    qj4 = qj0 * (pd.concat([close, open_], axis=1).max(axis=1) - jj)
    return ((qj1 + qj2) - (qj3 + qj4)) / 10000


def _line_slope_pct(series: pd.Series, lookback: int) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= lookback:
        return None
    latest = float(values.iloc[-1])
    previous = float(values.iloc[-lookback - 1])
    if previous == 0:
        return None
    return (latest - previous) / previous * 100


def _above_line_ratio(close: pd.Series, line: pd.Series, lookback: int) -> float | None:
    frame = pd.concat([pd.to_numeric(close, errors="coerce"), pd.to_numeric(line, errors="coerce")], axis=1).dropna()
    if len(frame) < lookback:
        return None
    window = frame.tail(lookback)
    return float((window.iloc[:, 0] >= window.iloc[:, 1]).mean())


def _consecutive_close_below(close: pd.Series, line: pd.Series) -> int:
    frame = pd.concat([pd.to_numeric(close, errors="coerce"), pd.to_numeric(line, errors="coerce")], axis=1).dropna()
    count = 0
    for _, row in frame.iloc[::-1].iterrows():
        if float(row.iloc[0]) < float(row.iloc[1]):
            count += 1
        else:
            break
    return count


def _consecutive_close_above(close: pd.Series, line: pd.Series) -> int:
    frame = pd.concat([pd.to_numeric(close, errors="coerce"), pd.to_numeric(line, errors="coerce")], axis=1).dropna()
    count = 0
    for _, row in frame.iloc[::-1].iterrows():
        if float(row.iloc[0]) > float(row.iloc[1]):
            count += 1
        else:
            break
    return count


def _latest_volume_ratio(frame: pd.DataFrame) -> float | None:
    if len(frame) < 6 or "vol" not in frame:
        return None
    vol = pd.to_numeric(frame["vol"], errors="coerce")
    latest_vol = vol.iloc[-1]
    prev_avg = vol.shift(1).rolling(5).mean().iloc[-1]
    if pd.isna(latest_vol) or pd.isna(prev_avg) or prev_avg == 0:
        return None
    return float(latest_vol / prev_avg)


def _golden_selling_exhaustion_reversal(
    frame: pd.DataFrame,
    lookback: int = 5,
    min_down_streak: int = 3,
    min_volume_shrink_pct: float = 50.0,
    min_lower_shadow_ratio: float = 0.4,
) -> dict[str, Any]:
    required = ["open", "high", "low", "close", "vol"]
    if len(frame) < lookback or any(column not in frame for column in required):
        return {"passed": None, "status": "unavailable", "reason": "selling exhaustion history unavailable"}

    window = frame.tail(lookback).copy()
    for column in required:
        window[column] = pd.to_numeric(window[column], errors="coerce")
    if window[required].isna().any().any():
        return {"passed": None, "status": "unavailable", "reason": "selling exhaustion OHLCV unavailable"}

    close = window["close"].reset_index(drop=True)
    volume = window["vol"].reset_index(drop=True)
    down_streak = 0
    volume_declining_in_streak = True
    for idx in range(len(window) - 1, 0, -1):
        if close.iloc[idx] < close.iloc[idx - 1]:
            down_streak += 1
            if volume.iloc[idx] > volume.iloc[idx - 1]:
                volume_declining_in_streak = False
            continue
        break

    top_volume = float(volume.max())
    latest_volume = float(volume.iloc[-1])
    volume_shrink_pct = (top_volume - latest_volume) / top_volume * 100 if top_volume else None

    latest = window.iloc[-1]
    candle_range = float(latest["high"] - latest["low"])
    lower_shadow = max(0.0, float(min(latest["open"], latest["close"]) - latest["low"]))
    lower_shadow_ratio = lower_shadow / candle_range if candle_range > 0 else None
    close_position = (
        (float(latest["close"]) - float(latest["low"])) / candle_range
        if candle_range > 0
        else None
    )
    latest_pct_chg = (
        (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        if float(close.iloc[-2]) != 0
        else None
    )

    passed = (
        down_streak >= min_down_streak
        and volume_declining_in_streak
        and volume_shrink_pct is not None
        and volume_shrink_pct >= min_volume_shrink_pct
        and lower_shadow_ratio is not None
        and lower_shadow_ratio >= min_lower_shadow_ratio
        and latest_pct_chg is not None
        and latest_pct_chg <= 0
    )
    return {
        "passed": bool(passed),
        "lookback": lookback,
        "down_streak": down_streak,
        "volume_declining_in_streak": volume_declining_in_streak,
        "top_volume": _round(top_volume),
        "latest_volume": _round(latest_volume),
        "volume_shrink_pct": _round(volume_shrink_pct),
        "lower_shadow_ratio": _round(lower_shadow_ratio),
        "close_position": _round(close_position),
        "latest_pct_chg": _round(latest_pct_chg),
        "min_down_streak": min_down_streak,
        "min_volume_shrink_pct": min_volume_shrink_pct,
        "min_lower_shadow_ratio": min_lower_shadow_ratio,
    }


def _channel_width_pct(upper: pd.Series, lower: pd.Series, close: pd.Series, offset: int) -> float | None:
    if len(close) <= offset:
        return None
    upper_value = upper.iloc[-offset - 1]
    lower_value = lower.iloc[-offset - 1]
    close_value = close.iloc[-offset - 1]
    if pd.isna(upper_value) or pd.isna(lower_value) or pd.isna(close_value) or float(close_value) == 0:
        return None
    return (float(upper_value) - float(lower_value)) / float(close_value) * 100


def _near_upper_ratio(close: pd.Series, upper: pd.Series, lookback: int, threshold_pct: float) -> float:
    frame = pd.concat([pd.to_numeric(close, errors="coerce"), pd.to_numeric(upper, errors="coerce")], axis=1).dropna()
    if len(frame) < lookback:
        return 0.0
    window = frame.tail(lookback)
    distance = (window.iloc[:, 1] - window.iloc[:, 0]).abs() / window.iloc[:, 0].replace(0, pd.NA) * 100
    return float(distance.le(threshold_pct).mean())


def _recent_upper_touch(high: pd.Series, close: pd.Series, upper: pd.Series, lookback: int, threshold_pct: float) -> bool:
    frame = pd.concat(
        [
            pd.to_numeric(high, errors="coerce"),
            pd.to_numeric(close, errors="coerce"),
            pd.to_numeric(upper, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if len(frame) < lookback:
        return False
    window = frame.tail(lookback)
    tolerance = 1 - threshold_pct / 100
    return bool((window.iloc[:, 0].ge(window.iloc[:, 2] * tolerance) | window.iloc[:, 1].ge(window.iloc[:, 2] * tolerance)).any())


def _golden_upper_lost(
    *,
    previous_above_upper: bool,
    recent_upper_touch: bool,
    close: float,
    high: float,
    upper_latest: float,
    close_below_support_days: int,
    upper_distance_pct: float | None,
) -> bool:
    return bool(
        previous_above_upper
        and recent_upper_touch
        and close < upper_latest * 0.99
        and high < upper_latest * 1.01
        and close_below_support_days == 0
        and upper_distance_pct is not None
        and upper_distance_pct <= 8.0
    )


def _golden_volume_break(
    *,
    prev_close: float,
    previous_support: float,
    close: float,
    support: float,
    open_: float,
    close_position: float,
    volume_ratio: float | None,
) -> bool:
    return bool(
        prev_close >= previous_support * 0.99
        and close < support * 0.99
        and close < open_
        and close_position < 0.45
        and volume_ratio is not None
        and volume_ratio >= 1.6
    )


def _golden_below_support_volume_recovery(
    *,
    close_below_support_days: int,
    close: float,
    open_: float,
    prev_close: float,
    high: float,
    support: float,
    volume_ratio: float | None,
) -> bool:
    return bool(
        close_below_support_days >= 3
        and close > open_
        and close > prev_close
        and volume_ratio is not None
        and volume_ratio >= 1.6
        and (high >= support * 0.99 or close >= support * 0.97)
    )


def _recent_below_then_reclaim(close: pd.Series, line: pd.Series, lookback: int) -> bool:
    frame = pd.concat([pd.to_numeric(close, errors="coerce"), pd.to_numeric(line, errors="coerce")], axis=1).dropna()
    if len(frame) < lookback:
        return False
    window = frame.tail(lookback)
    latest_close = float(window.iloc[-1, 0])
    latest_line = float(window.iloc[-1, 1])
    previous = window.iloc[:-1]
    return bool(previous.iloc[:, 0].lt(previous.iloc[:, 1]).any() and latest_close >= latest_line)


def _golden_channel_regime(life_line: float, trend_confirmation_line: float) -> str:
    if abs(life_line - trend_confirmation_line) <= 1e-8:
        return "transition"
    return "bull" if trend_confirmation_line < life_line else "bear"


def _golden_channel_strength(life_line: float, trend_confirmation_line: float, close: float) -> tuple[int, float]:
    gap_pct = abs(life_line - trend_confirmation_line) / close * 100 if close else 0.0
    if gap_pct < 0.5:
        return 1, gap_pct
    if gap_pct < 1.0:
        return 2, gap_pct
    if gap_pct < 2.0:
        return 3, gap_pct
    if gap_pct < 4.0:
        return 4, gap_pct
    return 5, gap_pct


_SCENE_RATINGS: dict[str, dict[str, list[tuple[str, int, str]]]] = {
    "fresh_upper_breakout": {
        "bull": [("持有", 5, "今日首次有效突破动态上轨，持仓趋势进入强化阶段。")],
        "bear": [("持有", 5, "今日首次有效突破动态上轨，突破当日优先持有并观察能否站稳。")],
        "neutral": [("持有", 5, "今日首次有效突破动态上轨，突破当日不追涨加仓也不提前减仓。")],
    },
    "upper_support": {
        "bull": [("加仓", 5, "上轨支撑确认，趋势延续强。"), ("持有", 4, "突破后的回踩未破，不宜因接近上轨直接卖出。")],
        "bear": [("持有", 3, "熊市里突破可信度下降，先确认承接。"), ("轻仓加仓", 2, "只适合小仓跟随。")],
        "neutral": [("持有", 3, "上轨回踩暂时有效，但通道环境未完全转强。")],
    },
    "upper_lost": {
        "bull": [("减仓", 3, "上轨支撑失效，强势状态降级。"), ("清仓", 1, "牛市里先给修复时间。")],
        "bear": [("减仓", 4, "熊市里上轨失守更容易变成反弹结束。"), ("清仓", 5, "反抽失败时退出优先级最高。")],
        "neutral": [("减仓", 3, "强势支撑失效，先降低仓位。")],
    },
    "upper_exhaustion": {
        "bull": [("减仓", 5, "无量连涨后的阴线，减仓信号很强。"), ("清仓", 4, "若继续转弱或跌回上轨，清仓候选增强。")],
        "bear": [("减仓", 5, "熊市缩量冲高更像诱多。"), ("清仓", 5, "站不回上轨或放量下跌时退出更坚决。")],
        "neutral": [("减仓", 4, "缩量上涨后的转阴降低上涨质量。"), ("清仓", 2, "等待是否继续破位确认。")],
    },
    "lower_support": {
        "bull": [("持有", 5, "金牛趋势线支撑有效，持股逻辑完整。"), ("加仓", 4, "若缩量回踩后放量转强，可考虑加仓。")],
        "bear": [("持有", 3, "熊市支撑有效也要降低信任度。"), ("试仓", 2, "没有放量转强前只适合小仓。"), ("减仓", 2, "反弹无力时准备降仓。")],
        "neutral": [("持有", 4, "趋势线被守住，继续观察。"), ("加仓", 2, "等待更明确的转强确认。")],
    },
    "bull_channel_pullback": {
        "bull": [("持有", 4, "牛市通道内回踩仍守在趋势支撑上方，优先按趋势持有。"), ("加仓", 2, "缩量回踩后若重新放量上攻，可小幅加仓。")],
        "bear": [("观察", 3, "熊市环境下通道内回踩不宜直接放大仓位。")],
        "neutral": [("持有", 3, "回踩未破趋势支撑，但通道强度仍需确认。")],
    },
    "lower_panic": {
        "bull": [("建仓", 4, "牛市深破下轨后的修复概率更高。"), ("加仓", 3, "已有底仓时可小幅提高仓位。")],
        "bear": [("试仓", 2, "熊市超跌只能小仓试错。"), ("观望", 4, "没有止跌证据前观望更强。")],
        "neutral": [("试仓", 3, "超跌修复可观察，但不宜重仓。")],
    },
    "bull_pressure": {
        "bull": [("持有", 4, "牛市里趋势权重大于压力权重。"), ("减仓", 2, "只在压力位明显滞涨时轻微降仓。")],
        "bear": [("不适用", 0, "该场景定义为牛市通道。")],
        "neutral": [("持有", 3, "接近压力但未出现明显风险确认。")],
    },
    "bear_pressure": {
        "bull": [("不适用", 0, "该场景定义为熊市通道。")],
        "bear": [("观察", 4, "熊市通道接近上轨压力，但当日尚未收阴确认转弱。"), ("减仓", 2, "压力环境下可轻微降低仓位，但不宜提前退出。")],
        "neutral": [("减仓", 3, "压力位失败时先保护仓位。")],
    },
    "upper_pressure_bearish": {
        "bull": [("减仓", 3, "上轨附近出现阴线，短线承压但趋势结构尚未确认破坏。"), ("观察", 3, "等待上轨支撑是否继续有效。")],
        "bear": [("减仓", 4, "熊市通道上轨附近出现阴线，压力开始得到确认。"), ("观察", 3, "尚未有效跌破上轨，暂不升级为退出信号。")],
        "neutral": [("减仓", 3, "上轨附近出现阴线，可先降低部分仓位。"), ("观察", 3, "等待是否有效跌破上轨。")],
    },
    "upper_below_pressure": {
        "bull": [("持有", 3, "牛市通道中仍在上轨下方，趋势未坏但不应按突破加仓。"), ("观察", 3, "等待放量站上上轨或回踩支撑后的确认。")],
        "bear": [("观察", 4, "熊市里上轨下方遇压优先观察。"), ("减仓", 2, "反抽无力时可降低仓位。")],
        "neutral": [("观察", 3, "接近上轨压力但尚未突破。")],
    },
    "channel_squeeze": {
        "bull": [("观望", 4, "通道收窄方向未定，等放量突破。"), ("持有", 3, "已有仓位可持有但不宜盲目加仓。")],
        "bear": [("观望", 5, "熊市收窄后向下选择代价更高。"), ("减仓", 2, "仓位过重可先降风险。")],
        "neutral": [("观望", 4, "通道收窄是变盘预警。")],
    },
    "upper_walk": {
        "bull": [("持有", 5, "沿上轨稳步运行，是强趋势持有信号。"), ("加仓", 2, "只在健康回踩后小幅加仓。")],
        "bear": [("持有", 3, "熊市里贴上轨也要降低持续性信任。"), ("减仓", 2, "量价背离或上影增多时先降仓。")],
        "neutral": [("持有", 4, "贴上轨运行说明趋势较强。")],
    },
    "golden2_lost": {
        "bull": [("持有", 3, "趋势线未破，可以给修复时间。"), ("减仓", 2, "短线仓或重仓可先降一点。")],
        "bear": [("减仓", 3, "熊市跌破金牛2更容易继续试探趋势线。"), ("观望", 4, "未收回金牛2前不轻易加仓。")],
        "neutral": [("观望", 3, "跌破金牛2属于预警层。"), ("减仓", 2, "仓位较重时先降风险。")],
    },
    "volume_break": {
        "bull": [("减仓", 5, "即使在牛市，放量破线也要先控风险。"), ("清仓", 3, "不能快速收回趋势线时清仓评级上升。")],
        "bear": [("减仓", 5, "熊市放量破线通常不值得硬扛。"), ("清仓", 5, "连续破位或反抽失败时清仓最强。")],
        "neutral": [("减仓", 5, "放量破线确认支撑失效。"), ("清仓", 3, "等待反抽是否失败。")],
    },
    "below_support_volume_recovery": {
        "bull": [("观察", 4, "支撑下方放量上涨并尝试收复，等待收盘重新站稳有效支撑。"), ("持有", 3, "修复动作已经出现，暂不按破位减仓。")],
        "bear": [("观察", 4, "熊市通道中出现放量修复，但站稳支撑前仍需谨慎。"), ("持有", 2, "已有仓位可观察修复持续性，不宜追涨。")],
        "neutral": [("观察", 4, "支撑下方放量上涨并尝试收复，等待进一步确认。"), ("持有", 3, "当前属于修复尝试，不按放量破位处理。")],
    },
    "reclaim_channel": {
        "bull": [("建仓", 3, "牛市里重回通道有修复价值。"), ("加仓", 3, "已有底仓时等回踩不破后再加。")],
        "bear": [("试仓", 2, "熊市修复容易失败，只适合小仓。"), ("观望", 3, "没有连续站稳前观望仍有强度。")],
        "neutral": [("试仓", 3, "重回通道是修复信号。"), ("观望", 3, "仍需站稳确认。")],
    },
}


def _resolve_golden_position_scenes(scene_flags: dict[str, bool]) -> list[str]:
    if scene_flags.get("volume_break"):
        return ["volume_break"]
    if scene_flags.get("fresh_upper_breakout"):
        return ["fresh_upper_breakout"]
    if scene_flags.get("below_support_volume_recovery"):
        return ["below_support_volume_recovery"]
    return [name for name, passed in scene_flags.items() if passed]


def _golden_position_ratings(
    scenes: list[str],
    channel_regime: str,
    *,
    breakout_close_position: float | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        options = _SCENE_RATINGS.get(scene, {})
        for action, strength, reason in options.get(channel_regime) or options.get("neutral") or []:
            if strength <= 0:
                continue
            if scene == "fresh_upper_breakout" and breakout_close_position is not None and breakout_close_position < 0.6:
                strength = 4
                reason = "今日首次站上动态上轨，但收盘位置偏低，按较弱突破继续持有观察。"
            current = merged.get(action)
            if current is None or strength > current["strength"]:
                merged[action] = {
                    "action": action,
                    "strength": int(strength),
                    "reason": reason,
                    "scene": scene,
                }
    if not merged:
        merged["持有"] = {
            "action": "持有",
            "strength": 2,
            "reason": "暂无明确加减仓信号。",
            "scene": "no_clear_signal",
        }
    return sorted(merged.values(), key=lambda item: (-int(item["strength"]), str(item["action"])))


def _golden_auxiliary_signals(
    frame: pd.DataFrame,
    channel_regime: str,
    channel_strength: int,
    *,
    selling_exhaustion_reversal: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    survival = check_survival_pattern(frame)
    if survival.get("passed") is True:
        signals.append(
            {
                "name": "绝地求生",
                "bullish_adjustment": 2,
                "bearish_adjustment": -1,
                "reason": "绝地求生修复形态与做多动作共振",
            }
        )
    divergence = check_macd_bottom_divergence(frame, 40)
    if divergence.get("passed") is True:
        signals.append(
            {
                "name": "MACD底背离",
                "bullish_adjustment": 1,
                "bearish_adjustment": -1,
                "reason": "MACD底背离提高修复可信度",
            }
        )
    if selling_exhaustion_reversal is None:
        selling_exhaustion_reversal = _golden_selling_exhaustion_reversal(frame)
    if selling_exhaustion_reversal.get("passed") is True:
        signals.append(
            {
                "name": "卖压衰竭长下影",
                "bullish_adjustment": 1,
                "bearish_adjustment": -2,
                "reason": "连续缩量下跌后出现长下影，卖盘衰竭提高反转观察价值",
            }
        )
    duanxian = check_duanxian_auxiliary_signals(frame)
    duanxian_signals = duanxian.get("signals") if isinstance(duanxian, dict) else {}
    if isinstance(duanxian_signals, dict):
        signals.extend(_duanxian_golden_auxiliary_signals(duanxian_signals))
    baogongtou = check_baogongtou_pattern(frame)
    if baogongtou.get("passed") is True:
        signals.append(
            {
                "name": "包公头",
                "bullish_adjustment": 2,
                "bearish_adjustment": -1,
                "reason": "10日内多次涨停但区间涨幅不大，说明资金反复试盘或蓄势",
            }
        )
    else:
        recent_limit_up = check_recent_limit_up(frame)
        if recent_limit_up.get("passed") is True:
            signals.append(
                {
                    "name": "10日内涨停",
                    "bullish_adjustment": 1,
                    "bearish_adjustment": 0,
                    "reason": "近期出现涨停，短线资金活跃度提高",
                }
            )
    if channel_strength >= 4 and channel_regime in {"bull", "bear"}:
        bullish_adjustment = 1 if channel_regime == "bull" else -1
        bearish_adjustment = 1 if channel_regime == "bear" else -1
        regime_name = "牛市" if channel_regime == "bull" else "熊市"
        signals.append(
            {
                "name": f"{regime_name}通道{channel_strength}分",
                "bullish_adjustment": bullish_adjustment,
                "bearish_adjustment": bearish_adjustment,
                "reason": f"{regime_name}通道结构清晰，方向一致信号增强",
            }
        )
    return signals


def _duanxian_golden_auxiliary_signals(duanxian_signals: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = {
        "price_support_triangle": ("短线是银价托", 1, -1, "5/10/20 日均线由空头收敛转为价托，增强修复或持有可信度"),
        "volume_support_triangle": ("短线是银量托", 1, -1, "5/10/20 均量线形成量托，说明底部承接和资金预警改善"),
        "bullish_sandwich": ("短线是银多方炮", 1, -1, "两阳夹一阴并放量收复，增强短线进攻确认"),
        "obv_golden_cross": ("短线是银OBV金叉", 1, -1, "OBV 上穿均线，能量潮改善"),
        "ant_channel_hold": ("短线是银蚂蚁功", 1, 0, "股价小步贴着 5/10 日均线通道上行，持股趋势未闭合"),
        "sesame_volume": ("短线是银芝麻量", 1, 0, "成交量收缩至 20 日低位且价格未破 MA20，短线抛压阶段性减轻"),
        "bearish_sandwich": ("短线是银空方炮", -1, 2, "两阴夹一阳并放量下压，增加派发或转弱风险"),
        "triple_dead_cross_top": ("短线是银三死叉", -2, 2, "价均线、量均线、MACD 同步死叉，顶部风险增强"),
        "obv_dead_cross": ("短线是银OBV死叉", -1, 1, "OBV 跌破均线，能量潮转弱"),
        "three_line_stop": ("短线是银三线止损", -2, 2, "前期处于均线上方持股后收盘跌破分级止损线，降低持有强度"),
    }
    signals: list[dict[str, Any]] = []
    for key, (name, bullish_adjustment, bearish_adjustment, reason) in mapping.items():
        value = duanxian_signals.get(key)
        if isinstance(value, dict) and value.get("passed") is True:
            signals.append(
                {
                    "name": name,
                    "bullish_adjustment": bullish_adjustment,
                    "bearish_adjustment": bearish_adjustment,
                    "reason": reason,
                    "source": key,
                }
            )
    return signals


def _apply_golden_rating_modifiers(
    ratings: list[dict[str, Any]],
    auxiliary_signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bullish_actions = {"建仓", "加仓", "轻仓加仓", "试仓", "持有"}
    bearish_actions = {"减仓", "清仓"}
    display_only: list[dict[str, Any]] = []
    for rating in ratings:
        action = str(rating.get("action"))
        base_strength = int(rating.get("strength") or 1)
        modifiers: list[dict[str, Any]] = []
        for signal in auxiliary_signals:
            if action in bullish_actions:
                adjustment = int(signal.get("bullish_adjustment") or 0)
            elif action in bearish_actions:
                adjustment = int(signal.get("bearish_adjustment") or 0)
            else:
                adjustment = 0
            if adjustment:
                modifiers.append(
                    {
                        "indicator": signal.get("name"),
                        "adjustment": adjustment,
                        "reason": signal.get("reason"),
                    }
                )
        display_only.append(
            {
                **rating,
                "base_strength": base_strength,
                "strength_adjustment": 0,
                "strength": base_strength,
                "modifiers": modifiers,
                "reason": str(rating.get("reason") or ""),
            }
        )
    return sorted(display_only, key=lambda item: (-int(item["strength"]), str(item["action"])))


def _suppress_unconfirmed_bullish_actions(
    ratings: list[dict[str, Any]],
    unconfirmed_sharp_bearish: bool,
) -> list[dict[str, Any]]:
    if not unconfirmed_sharp_bearish:
        return ratings
    blocked_actions = {"建仓", "加仓", "轻仓加仓", "试仓"}
    filtered = [rating for rating in ratings if rating.get("action") not in blocked_actions]
    existing_observation = next((rating for rating in filtered if rating.get("action") == "观察"), None)
    if existing_observation is None or int(existing_observation.get("strength") or 0) <= 4:
        filtered = [rating for rating in filtered if rating.get("action") != "观察"]
        filtered.append(
            {
                "action": "观察",
                "strength": 4,
                "reason": "当日大跌且收盘接近日内低位，尚无止跌确认，暂不新增仓位。",
                "scene": "unconfirmed_sharp_bearish",
                "base_strength": 4,
                "strength_adjustment": 0,
                "modifiers": [],
            }
        )
    return sorted(filtered, key=lambda item: (-int(item["strength"]), str(item["action"])))


def _golden_upper_shadow_state(
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    channel_upper: float,
    volume_ratio: float | None,
    close_position: float,
) -> str | None:
    candle_range = high - low
    if candle_range <= 0:
        return None
    upper_shadow = high - max(open_, close)
    body = abs(close - open_)
    rejected = (
        high >= channel_upper
        and volume_ratio is not None
        and volume_ratio >= 1.2
        and upper_shadow >= body
        and upper_shadow / candle_range >= 0.35
    )
    if not rejected:
        return None
    if close < channel_upper or close_position <= 0.35:
        return "failed"
    return "rejected"


def _apply_upper_shadow_constraint(
    ratings: list[dict[str, Any]],
    upper_shadow_state: str | None,
) -> list[dict[str, Any]]:
    if upper_shadow_state not in {"rejected", "failed"}:
        return ratings
    blocked_actions = {"建仓", "加仓", "轻仓加仓", "试仓"}
    filtered = [rating for rating in ratings if rating.get("action") not in blocked_actions]
    if upper_shadow_state == "failed":
        action = "减仓"
        strength = 4
        reason = "放量长上影后收盘转弱，突破失败，先降低部分仓位。"
    else:
        action = "观察"
        strength = 4
        reason = "触及或突破上轨后出现放量长上影，抛压较强，暂不新增仓位。"
    current = next((rating for rating in filtered if rating.get("action") == action), None)
    if current is None or int(current.get("strength") or 0) < strength:
        filtered = [rating for rating in filtered if rating.get("action") != action]
        filtered.append(
            {
                "action": action,
                "strength": strength,
                "reason": reason,
                "scene": f"upper_shadow_{upper_shadow_state}",
                "base_strength": strength,
                "strength_adjustment": 0,
                "modifiers": [],
            }
        )
    return sorted(filtered, key=lambda item: (-int(item["strength"]), str(item["action"])))


def _golden_clearance_state(
    *,
    is_bearish_candle: bool,
    close: float,
    high: float,
    upper_latest: float,
    close_below_upper_days: int,
    recent_upper_touch: bool,
    volume_break: bool,
) -> str:
    if volume_break:
        return "confirmed"
    effective_upper_break = close < upper_latest * 0.99
    failed_upper_retest = (
        is_bearish_candle
        and effective_upper_break
        and close_below_upper_days >= 2
        and recent_upper_touch
        and high < upper_latest * 1.01
    )
    if failed_upper_retest:
        return "confirmed"
    if is_bearish_candle and effective_upper_break:
        return "candidate"
    return "blocked"


def _suppress_upper_pressure_new_positions(
    ratings: list[dict[str, Any]],
    upper_pressure_active: bool,
) -> list[dict[str, Any]]:
    if not upper_pressure_active:
        return ratings
    blocked_actions = {"建仓", "加仓", "轻仓加仓", "试仓"}
    return [rating for rating in ratings if rating.get("action") not in blocked_actions]


def _apply_clearance_gate(
    ratings: list[dict[str, Any]],
    clearance_state: str,
) -> list[dict[str, Any]]:
    if clearance_state == "confirmed":
        return ratings
    if clearance_state == "blocked":
        return [rating for rating in ratings if rating.get("action") != "清仓"]

    adjusted: list[dict[str, Any]] = []
    for rating in ratings:
        if rating.get("action") != "清仓":
            adjusted.append(rating)
            continue
        base_strength = min(2, int(rating.get("base_strength") or rating.get("strength") or 1))
        strength_adjustment = min(0, int(rating.get("strength_adjustment") or 0))
        adjusted.append(
            {
                **rating,
                "base_strength": base_strength,
                "strength_adjustment": strength_adjustment,
                "strength": max(1, min(2, base_strength + strength_adjustment)),
                "reason": f"{rating.get('reason', '')}；当前仅为阴线跌破上轨的清仓候选，等待连续失守或反抽失败确认。",
            }
        )
    return sorted(adjusted, key=lambda item: (-int(item["strength"]), str(item["action"])))


def _golden_position_risk_flags(
    scenes: list[str],
    channel_regime: str,
    profit_pct: float | None,
    unconfirmed_sharp_bearish: bool = False,
    upper_shadow_state: str | None = None,
) -> list[str]:
    flags: list[str] = []
    if "volume_break" in scenes:
        flags.append("放量跌破金牛趋势线/下轨")
    if "below_support_volume_recovery" in scenes:
        flags.append("仍未完全站稳有效支撑")
    if "upper_exhaustion" in scenes:
        flags.append("缩量连涨后转阴")
    if "fresh_upper_breakout" in scenes:
        flags.append("突破当日不追涨加仓，关注后续能否站稳上轨")
    if unconfirmed_sharp_bearish:
        flags.append("当日大跌且收盘接近日内低位，尚未止跌")
    if upper_shadow_state == "rejected":
        flags.append("上轨附近放量长上影，突破承接不足")
    if upper_shadow_state == "failed":
        flags.append("上轨附近放量长上影且收盘转弱，突破失败")
    if channel_regime == "bear":
        flags.append("熊市通道")
    if profit_pct is not None and profit_pct <= -8:
        flags.append("成本价下方亏损超过 8%")
    if profit_pct is not None and profit_pct >= 20 and any(scene in scenes for scene in ["upper_exhaustion", "upper_lost"]):
        flags.append("较大浮盈叠加高位转弱")
    return flags


def _golden_position_invalidations(scenes: list[str]) -> list[str]:
    invalidations: list[str] = []
    if "fresh_upper_breakout" in scenes:
        invalidations.append("收盘跌回动态上轨下方且无法快速收复")
    if any(scene in scenes for scene in ["upper_support", "upper_walk"]):
        invalidations.append("跌回上轨下方且反抽失败")
    if any(scene in scenes for scene in ["lower_support", "bull_channel_pullback", "lower_panic", "reclaim_channel"]):
        invalidations.append("放量跌破金牛趋势线/下轨")
    if "channel_squeeze" in scenes:
        invalidations.append("放量向下破位")
    if "golden2_lost" in scenes:
        invalidations.append("继续跌破金牛趋势线")
    if "below_support_volume_recovery" in scenes:
        invalidations.append("放量修复后再次转弱并创近期新低")
    return invalidations or ["次日走势与当前场景相反时重新评级"]


def _context_float(context: dict[str, object] | None, key: str) -> float | None:
    if not context or context.get(key) in (None, ""):
        return None
    try:
        return float(context[key])
    except (TypeError, ValueError):
        return None


def _golden_bull_volume_momentum(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) < 6 or "vol" not in frame:
        return {"passed": None, "status": "unavailable", "reason": "volume history less than 6 days"}
    vol = pd.to_numeric(frame["vol"], errors="coerce")
    vol_ma5_prev = vol.shift(1).rolling(5).mean()
    latest_vol = vol.iloc[-1]
    prev_avg = vol_ma5_prev.iloc[-1]
    volume_ratio = float(latest_vol / prev_avg) if pd.notna(latest_vol) and pd.notna(prev_avg) and prev_avg != 0 else None
    macd_improving = None
    if "macd_hist" in frame and len(frame) >= 3:
        hist = pd.to_numeric(frame["macd_hist"], errors="coerce")
        if hist.tail(3).notna().all():
            macd_improving = bool(hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3])
    passed = (
        volume_ratio is not None
        and 1.1 <= volume_ratio <= 3.0
        and (macd_improving is True or macd_improving is None)
    )
    return {
        "passed": bool(passed),
        "volume_ratio": _round(volume_ratio),
        "macd_improving": macd_improving,
    }
