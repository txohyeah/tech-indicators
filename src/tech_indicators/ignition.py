"""起爆点（ignition）择时模块 —— 通达信「临界起爆点」还原 + 自有买卖规则（定稿 v2）。

信号：RSI(N) 自下而上穿越 LL（等价 `CROSS(RSI6,40)`）。
买入过滤：跌得够深（距 60 日高点 < -40%）且已离开底部（距 60 日低点 > 5%）、
          起爆 K 线非假阴线、上影不超过全幅 3/4。
卖出：硬止损 max(买价×(1-10%), 起爆点及其前 4 根最低价) 收盘触发；未进入利润奔跑前
      遇新起爆点则重锚止损；浮盈≥20% 后只按「最高点 − 总涨幅×25%」移动止盈；
      盘中摸到金牛上沿但收盘收回下方（阴线或长上影阳线）→ 减半，若当日为熊市通道 → 清仓。

回测依据（2018-2026 九年、逐年重建 60 只票池、10 槽 × 10% 权益、含费、含退市股）：
    裸 CROSS(RSI6,40)     累计 +1036%  年化 31.0%  最差单年  -9.2%  年内最大回撤 -18.9%  2473 笔
    裸 + 形态过滤         累计 +1141%  年化 32.3%  最差单年  -5.6%  年内最大回撤 -18.0%  2334 笔
    裸 + 位置过滤         累计   +85%  年化  7.0%  最差单年  +0.2%  年内最大回撤  -6.8%   118 笔
    定稿 v2（位置+形态）  累计   +88%  年化  7.2%  最差单年  +0.2%  年内最大回撤  -6.0%   115 笔
形态过滤是纯赚（收益/夏普/最差年/回撤全面更好）；位置过滤是一笔很贵的保险——用约 12 倍
收益换来回撤压掉 3 倍、胜率 52%->71%，是否值得尚未定论。本模块把三道条件分开暴露
（ignition_cross_signal / ignition_position_filter / ignition_candle_filter），
默认组成仅作当前定稿，不代表唯一合理搭配；把位置条件降级为「仓位调节」而非一票否决的方案
正在验证。三道条件之外的东西不进信号。
注：更早一版九年回测引擎把未复权价算出的金牛上沿与复权价比较，其结论「加买入过滤把
9 年 -9% 变成 +56%」已作废。双 MA 趋势闸门（MA20 与 MA60 同时向下则忽略起爆点）已用
24379 笔信号的前向收益独立复验（事件研究不经过金牛线，故不受上述 bug 影响）：被该闸门
拦截的一组 20 日均值 +1.24%、胜率 50.1%，放行的一组只有 +0.17%、胜率 45.0% —— 它是
反向过滤器，且会把「超跌起爆」信号量砍掉 82%，故永久不设；同向的「熊市通道」状态只用于
卖出端。ARC 深套门控的否定结论仍出自该引擎，待用事件研究法复验。
纯计算、零 IO。过程记录见 memory/2026-09-04/ignition-bear-market.md。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indicators import compute_golden_bull_lines

# ---- 参数（对应通达信公式的参数表）----
IGNITION_RSI_PERIOD = 6          # 通达信参数 N
IGNITION_TRIGGER = 40.0          # 通达信参数 LL，只允许在弱势区触发
IGNITION_LOOKBACK = 60           # 位置回看窗口
IGNITION_DEEP_DRAWDOWN_PCT = -40.0
IGNITION_OFF_BOTTOM_PCT = 5.0
IGNITION_UPPER_SHADOW_MAX = 0.75
IGNITION_DELAY_GAIN_PCT = 5.0    # 起爆日涨幅超过该值则次日再买

# ---- 卖出规则（自有策略层）----
IGNITION_STOP_PCT = 0.10         # 硬止损：买价下方 10%（与结构位取高者）
IGNITION_STOP_BARS = 30          # 结构止损：滚动窗口，截至昨日的最近 30 根最低价
IGNITION_RUN_GAIN_PCT = 0.20     # 利润奔跑启动门槛
IGNITION_TRAIL_FRACTION = 0.35   # 回撤总涨幅的该比例即离场（0.35 实测优于 0.25）
IGNITION_UPPER_TOUCH = 0.98      # 盘中触及上沿的容差
IGNITION_MIN_BARS = 80
# ---- 2026-09-05 因果版 + 15 槽组合层重测定稿（见 memory/2026-09-05/exit-rules-causal-final.md）----
IGNITION_UPPER_EXIT = "full"         # 撞上沿的出场方式：full=全清（已验证）/ half=减半（会占槽位，组合层更差）
IGNITION_USE_TRAILING = False        # 移动止盈默认关闭：关闭的 C2 在无前视池上年化与回撤都优于开启的 C3
IGNITION_REANCHOR = False            # 「新起爆点重锚止损」默认关闭：滚动止损已包含该效果，且旧结论出自坏引擎
SUPPORTED_UPPER_EXITS = ("full", "half")


def validate_exit_config() -> None:
    """挡住会退化成「买入持有」的配置组合。

    减半若不配任何最终清仓规则（移动止盈或结构/硬止损），剩下那半仓会永久持有 ——
    实测这种配置在自选池上跑出 +61.9%，看着像策略，其实是买入持有换了个马甲。
    """
    if IGNITION_UPPER_EXIT not in SUPPORTED_UPPER_EXITS:
        raise ValueError(f"IGNITION_UPPER_EXIT 只能是 {SUPPORTED_UPPER_EXITS}，收到 {IGNITION_UPPER_EXIT!r}")
    if IGNITION_UPPER_EXIT == "half" and not (IGNITION_USE_TRAILING or IGNITION_STOP_BARS or IGNITION_STOP_PCT):
        raise ValueError("减半出场必须有兜底清仓规则（移动止盈或止损），否则退化为买入持有")


def _returns(closes: pd.Series) -> pd.Series:
    return closes.diff()


def _pct_change(closes: pd.Series, frame: pd.DataFrame) -> pd.Series:
    if "pct_chg" in frame and frame["pct_chg"].notna().any():
        return pd.to_numeric(frame["pct_chg"], errors="coerce")
    return (closes / closes.shift(1) - 1.0) * 100.0


def ignition_rsi(history: pd.DataFrame, period: int | None = None) -> pd.Series:
    """Wilder RSI，等价通达信 `SMA(MAX(C-REF(C,1),0),N,1)/SMA(ABS(C-REF(C,1)),N,1)*100`。

    必须用**收盘价价差**口径；改用涨跌幅口径与「临界起爆点」的一致率会从 0.958 掉到 0.837。
    """
    n = IGNITION_RSI_PERIOD if period is None else max(int(period), 1)
    delta = _returns(pd.to_numeric(history["close"], errors="coerce"))
    gain = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    denom = gain + loss
    return (100.0 * gain / denom).where(denom != 0)


def ignition_cross_signal(history: pd.DataFrame) -> pd.Series:
    """裸信号：RSI 自下而上穿越 LL（不含任何买入过滤）。"""
    rsi = ignition_rsi(history)
    above = rsi > IGNITION_TRIGGER
    return (above & ~above.shift(1).fillna(False).astype(bool)).fillna(False)


def ignition_position_filter(history: pd.DataFrame) -> pd.Series:
    """位置过滤：跌得够深（距 60 日高点 < -40%）且已离底（距 60 日低点 > 5%）。

    注意代价：9 年回测里加上它，累计从 +1036% 掉到 +85%（年化 31.0% -> 7.0%），
    换来最差单年 -9.2% -> +0.2%、年内最大回撤 -18.9% -> -6.8%、胜率 52% -> 71%。
    是保险而非免费改进，是否作为一票否决尚未定论。
    """
    closes = pd.to_numeric(history["close"], errors="coerce")
    highs = pd.to_numeric(history["high"], errors="coerce")
    lows = pd.to_numeric(history["low"], errors="coerce")
    drawdown = (closes / highs.rolling(IGNITION_LOOKBACK).max() - 1.0) * 100.0
    off_bottom = (closes / lows.rolling(IGNITION_LOOKBACK).min() - 1.0) * 100.0
    return (drawdown < IGNITION_DEEP_DRAWDOWN_PCT) & (off_bottom > IGNITION_OFF_BOTTOM_PCT)


def ignition_candle_filter(history: pd.DataFrame) -> pd.Series:
    """起爆 K 线形态过滤：排除假阴线（收涨但收阴）与上影超过全幅 3/4 的长上影。"""
    closes = pd.to_numeric(history["close"], errors="coerce")
    opens = pd.to_numeric(history["open"], errors="coerce")
    highs = pd.to_numeric(history["high"], errors="coerce")
    lows = pd.to_numeric(history["low"], errors="coerce")
    span = (highs - lows).replace(0.0, np.nan)
    upper_shadow = (highs - pd.concat([opens, closes], axis=1).max(axis=1)) / span
    fake_bear = (closes < opens) & (_pct_change(closes, history) > 0)
    long_shadow = upper_shadow > IGNITION_UPPER_SHADOW_MAX
    return (~fake_bear.fillna(False)) & (~long_shadow.fillna(False))


def ignition_signal_series(history: pd.DataFrame) -> pd.Series:
    """定稿 v2 的完整买入信号序列（信号 + 位置 + 形态）。"""
    if len(history) < IGNITION_MIN_BARS:
        return pd.Series(False, index=history.index)
    return (
        ignition_cross_signal(history)
        & ignition_position_filter(history).fillna(False)
        & ignition_candle_filter(history).fillna(False)
    ).fillna(False)


def ignition_signal(history: pd.DataFrame) -> bool:
    """最后一根 K 线是否为定稿 v2 的起爆买点。"""
    series = ignition_signal_series(history)
    return bool(series.iloc[-1]) if len(series) else False


def ignition_signal_breakdown(history: pd.DataFrame) -> dict[str, Any]:
    """逐项拆解：为什么买 / 为什么没买（给 CLI 和复盘用）。"""
    closes = pd.to_numeric(history["close"], errors="coerce")
    highs = pd.to_numeric(history["high"], errors="coerce")
    lows = pd.to_numeric(history["low"], errors="coerce")
    rsi = ignition_rsi(history)
    drawdown = (closes / highs.rolling(IGNITION_LOOKBACK).max() - 1.0) * 100.0
    off_bottom = (closes / lows.rolling(IGNITION_LOOKBACK).min() - 1.0) * 100.0
    opens = pd.to_numeric(history["open"], errors="coerce")
    span = (highs - lows).replace(0.0, np.nan)
    shadow = ((highs - pd.concat([opens, closes], axis=1).max(axis=1)) / span).iloc[-1]
    return {
        "rsi": None if pd.isna(rsi.iloc[-1]) else round(float(rsi.iloc[-1]), 2),
        "rsi_prev": None if len(rsi) < 2 or pd.isna(rsi.iloc[-2]) else round(float(rsi.iloc[-2]), 2),
        "trigger": IGNITION_TRIGGER,
        "cross": bool(ignition_cross_signal(history).iloc[-1]),
        "drawdown_pct": None if pd.isna(drawdown.iloc[-1]) else round(float(drawdown.iloc[-1]), 2),
        "off_bottom_pct": None if pd.isna(off_bottom.iloc[-1]) else round(float(off_bottom.iloc[-1]), 2),
        "position_ok": bool(ignition_position_filter(history).fillna(False).iloc[-1]),
        "upper_shadow_ratio": None if pd.isna(shadow) else round(float(shadow), 3),
        "candle_ok": bool(ignition_candle_filter(history).fillna(False).iloc[-1]),
        "signal": ignition_signal(history),
    }


def _stop_line(entry_px: float, lows: pd.Series, i: int) -> float:
    """止损线 = max(买价×(1-10%), 截至昨日的最近 30 根最低价)。

    窗口刻意**不含当根**：含当根时 `close < min(...)` 恒不成立，止损位是永远打不到的纸面价。
    """
    line = entry_px * (1.0 - IGNITION_STOP_PCT) if IGNITION_STOP_PCT else -float("inf")
    start = max(0, i - IGNITION_STOP_BARS)
    if i > start and IGNITION_STOP_BARS:
        ref = float(lows.iloc[start:i].min())
        if np.isfinite(ref):
            line = max(line, ref)
    return line


def golden_channel_state(history: pd.DataFrame, causal: bool = False) -> pd.DataFrame:
    """金牛通道状态：upper=通道上沿，bear=趋势确认线压在生命线上方（熊市通道）。

    卖出规则的「冲不过上沿」与「熊市通道清仓」都依赖这两项，回测与实盘共用本函数。

    ⚠️ ``causal`` 必须显式选对：默认 ``False`` 与通达信图一致，但**含未来函数**
    （第 i 根用了其后 12 根数据，且图上最近 12 根会随行情漂移）。
    **做回测或"当时是否触发"的判断时一律传 ``causal=True``**；只有画图和与通达信对表时用默认。
    """
    lines = compute_golden_bull_lines(history, causal=causal)
    out = pd.DataFrame(index=lines.index)
    out["upper"] = lines["channel_upper"]
    out["lower"] = lines["golden_bull_trend"]
    out["confirm"] = lines["golden_bull_2"]
    out["bear"] = out["confirm"] > out["lower"]        # 熊市通道：趋势确认线压在生命线上方
    return out


def ignition_position_state(history: pd.DataFrame) -> dict[str, Any] | None:
    """从最近一个起爆点起重放卖出状态机，得到当前持仓应有的状态。

    返回 None 表示历史上还没有过买点。状态是确定性的：同样的历史得到同样的状态。
    """
    if len(history) < IGNITION_MIN_BARS:
        return None
    frame = history.reset_index(drop=True)
    closes = pd.to_numeric(frame["close"], errors="coerce")
    opens = pd.to_numeric(frame["open"], errors="coerce")
    highs = pd.to_numeric(frame["high"], errors="coerce")
    lows = pd.to_numeric(frame["low"], errors="coerce")
    pct = _pct_change(closes, frame)
    sig = ignition_signal_series(frame)
    marks = np.where(sig.values)[0]
    if not len(marks):
        return None
    s = int(marks[-1])
    delay = float(pct.iloc[s]) > IGNITION_DELAY_GAIN_PCT
    if delay and s + 1 >= len(frame):
        # 起爆日涨幅超阈值、但"次日"还不存在：买点待定，不能当成今天已建仓
        return {
            "signal_date": str(frame["trade_date"].iloc[s]) if "trade_date" in frame else None,
            "entry_date": None,
            "entry_price": None,
            "pending_entry": True,
            "bars_held": 0,
            "last_close": round(float(closes.iloc[-1]), 4),
            "highest_since_entry": None,
            "unrealized_pct": None,
            "stop_line": None,
            "running": False,
            "half_reduced": False,
            "exit_reason": None,
            "exit_date": None,
            "bear_channel": bool(golden_channel_state(frame, causal=True)["bear"].iloc[-1]),
            "closed": False,
        }
    entry = s + 1 if delay else s
    entry_px = float(closes.iloc[entry])
    validate_exit_config()
    stop = _stop_line(entry_px, lows, entry)
    gold = golden_channel_state(frame, causal=True)
    hi = float(highs.iloc[entry])
    running = False
    half = False
    exit_reason: str | None = None
    exit_index: int | None = None
    for i in range(entry + 1, len(frame)):
        close = float(closes.iloc[i])
        hi = max(hi, float(highs.iloc[i]))
        stop = _stop_line(entry_px, lows, i)          # 滚动：截至昨日的最近 30 根
        if IGNITION_USE_TRAILING and not running and hi / entry_px - 1.0 >= IGNITION_RUN_GAIN_PCT:
            running = True
        if IGNITION_USE_TRAILING and running and (hi - close) >= IGNITION_TRAIL_FRACTION * (hi - entry_px):
            exit_reason, exit_index = "trailing_take_profit", i
            break
        if close < stop:
            exit_reason, exit_index = "stop_loss", i
            break
        if (not half) and float(highs.iloc[i]) >= float(gold["upper"].iloc[i]) * IGNITION_UPPER_TOUCH \
                and close < float(gold["upper"].iloc[i]):
            body = abs(close - float(opens.iloc[i]))
            shadow = float(highs.iloc[i]) - max(close, float(opens.iloc[i]))
            bearish = close < float(opens.iloc[i])
            long_shadow = close >= float(opens.iloc[i]) and body > 0 and shadow >= 2.0 * body \
                and shadow >= 0.03 * close
            if bearish or long_shadow:
                if IGNITION_UPPER_EXIT == "full":
                    exit_reason, exit_index = "upper_pressure_exit", i
                    break
                half = True
    now = len(frame) - 1
    closed = exit_reason is not None
    last_index = exit_index if closed and exit_index is not None else now
    exit_px = float(closes.iloc[last_index])
    return {
        "signal_date": str(frame["trade_date"].iloc[s]) if "trade_date" in frame else None,
        "entry_date": str(frame["trade_date"].iloc[entry]) if "trade_date" in frame else None,
        "entry_price": round(entry_px, 4),
        "pending_entry": False,
        "bars_held": last_index - entry,
        "last_close": round(float(closes.iloc[now]), 4),
        "highest_since_entry": round(hi, 4),
        "exit_price": round(exit_px, 4) if closed else None,
        "unrealized_pct": round((exit_px / entry_px - 1.0) * 100.0, 2) if closed
        else round((float(closes.iloc[now]) / entry_px - 1.0) * 100.0, 2),
        "stop_line": round(stop, 4),
        "running": running,
        "half_reduced": half,
        "exit_reason": exit_reason,
        "exit_date": str(frame["trade_date"].iloc[exit_index]) if exit_index is not None and "trade_date" in frame else None,
        "bear_channel": bool(gold["bear"].iloc[now]),
        "closed": closed,
    }


def build_ignition_trade_plan(
    history: pd.DataFrame,
    *,
    current_position_pct: float = 0.0,
    max_positions: int = 5,
    position_pct: float = 0.20,
    timeframe_label: str = "1d",
) -> dict[str, Any]:
    """定稿 v2 的当日交易计划。

    无持仓且有信号 → buy（起爆日涨幅超阈值时提示次日买）；
    有持仓 → 依据重放出的状态给 hold / reduce_half / sell_all。
    """
    label = timeframe_label or "K-line"
    state = ignition_position_state(history)
    breakdown = ignition_signal_breakdown(history)
    plan: dict[str, Any] = {
        "action": "wait",
        "side": "hold",
        "signal_type": "no_trade",
        "target_position_pct": current_position_pct,
        "position_cap_pct": position_pct,
        "max_positions": max_positions,
        "stop_line_name": None,
        "stop_line_price": None,
        "ignition_signal": breakdown["signal"],
        "ignition_state": state,
        "ignition_breakdown": breakdown,
        "reason": [],
    }
    if state is None:
        plan["reason"] = [f"no {label} ignition signal in history; nothing to manage"]
        if breakdown["signal"]:
            plan["reason"] = [f"{label} ignition point; entry pending"]
        return plan

    plan["stop_line_name"] = "max(entry-10pct, ignition 5-bar low)"
    plan["stop_line_price"] = state["stop_line"]

    if state.get("pending_entry"):
        plan["action"] = "hold" if current_position_pct > 0 else "buy_next_bar"
        plan["side"] = "hold" if current_position_pct > 0 else "buy"
        plan["signal_type"] = "ignition_entry_pending"
        plan["target_position_pct"] = current_position_pct if current_position_pct > 0 else position_pct
        plan["stop_line_name"] = None
        plan["stop_line_price"] = None
        plan["reason"] = [
            f"ignition bar gained more than {IGNITION_DELAY_GAIN_PCT}%; entry is scheduled for the next bar close"
        ]
        return plan

    if state["closed"]:
        reason_map = {
            "stop_loss": f"position already stopped out at {state['exit_date']} (close below {state['stop_line']})",
            "trailing_take_profit": f"position already taken profit at {state['exit_date']} (gave back 25% of the gain)",
            "upper_pressure_bear_exit": f"position already exited at {state['exit_date']} (failed at Golden Bull upper line in bear channel)",
        }
        plan["action"] = "hold" if current_position_pct <= 0 else "sell_all"
        plan["side"] = "sell" if current_position_pct > 0 else "hold"
        plan["signal_type"] = f"ignition_{state['exit_reason']}"
        plan["target_position_pct"] = 0.0 if current_position_pct > 0 else 0.0
        plan["reason"] = [reason_map.get(state["exit_reason"], "position closed")]
        return plan

    if current_position_pct <= 0:
        if breakdown["signal"]:
            delay = float(_pct_change(pd.to_numeric(history["close"], errors="coerce"), history).iloc[-1]) > IGNITION_DELAY_GAIN_PCT
            plan["action"] = "buy_next_bar" if delay else "buy"
            plan["side"] = "buy"
            plan["signal_type"] = "ignition_entry"
            plan["target_position_pct"] = position_pct
            plan["reason"] = [
                "ignition point confirmed: RSI6 crossed up through 40 after a deep pullback, off the 60-day low, candle clean"
            ]
            if delay:
                plan["reason"].append("ignition bar gained >5%; enter next bar close instead of chasing")
        else:
            plan["reason"] = ["holding nothing and today is not an ignition point"]
        return plan

    plan["action"] = "hold"
    plan["side"] = "hold"
    plan["signal_type"] = "ignition_running" if state["running"] else "ignition_holding"
    plan["target_position_pct"] = current_position_pct
    notes = [
        f"bars held {state['bars_held']}, unrealized {state['unrealized_pct']:+.2f}%",
        f"stop line {state['stop_line']}",
    ]
    if state["running"]:
        trail = state["highest_since_entry"] - IGNITION_TRAIL_FRACTION * (
            state["highest_since_entry"] - state["entry_price"])
        notes.append(f"profit mode on: sell if close drops below {round(trail, 4)}")
        plan["stop_line_name"] = f"trailing {IGNITION_TRAIL_FRACTION:.0%} of total gain"
        plan["stop_line_price"] = round(trail, 4)
    else:
        notes.append("not in profit mode yet: fixed stop still armed, a new ignition point would re-anchor it")
    if state["half_reduced"]:
        notes.append("half position already reduced at the Golden Bull upper line")
    if state["bear_channel"]:
        notes.append("bear channel is informational only - it is NOT an exit trigger for deep-dip entries")
    plan["reason"] = notes
    return plan
