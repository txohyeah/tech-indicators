"""tech-indicators：A 股 / Crypto 通用技术指标、交易计划与策略评分库。

纯计算、零 IO：输入 OHLCV DataFrame，输出指标 / 信号 / 评分。
调用方（stock-analytics 的 A 股分析、crypto-research 等）负责取数与持久化。

公共入口：
- compute_indicators / compute_golden_bull_lines / check_*  指标计算（indicators）
- build_golden_bull_trade_plan / build_reburn_buy_trade_plan / build_unified_kline_trade_plan（交易计划）
- get_strategy / load_strategies / RuleEvaluator（策略定义与评分）
- run_chart（HTML K 线渲染）
"""

from __future__ import annotations

from . import (
    chart,
    contract,
    errors,
    golden_bull_trading,
    indicators,
    kline_decision,
    models,
    reburn,
)
from .strategies import RuleEvaluator, Strategy, get_strategy, load_strategies

__version__ = "0.1.0"

__all__ = [
    "chart",
    "contract",
    "errors",
    "golden_bull_trading",
    "indicators",
    "kline_decision",
    "models",
    "reburn",
    "RuleEvaluator",
    "Strategy",
    "get_strategy",
    "load_strategies",
    "__version__",
]