"""策略评分模型（从 stock-research models.py 迁移，仅保留策略评估相关部分）。

StockCode / PositionInput / MissingDataContract 等 A 股领域模型留在调用方（stock-analytics），
不进入纯技术指标包。RuleResult / StrategyEvaluation 是策略评估的纯数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuleResult:
    name: str
    passed: bool
    reason: str
    weight: int = 0
    value: Any = None
    status: str = "passed"


@dataclass
class StrategyEvaluation:
    code: str
    ts_code: str
    name: str
    score: int
    grade: str
    bucket: str
    close: float | None
    pct_chg: float | None
    indicators: dict[str, Any]
    rule_results: list[RuleResult] = field(default_factory=list)
    core_rule_results: list[RuleResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exclude_reasons: list[str] = field(default_factory=list)

    @property
    def hit_reasons(self) -> list[str]:
        return [r.reason for r in self.rule_results if r.passed and r.weight > 0]

    @property
    def penalty_reasons(self) -> list[str]:
        return [r.reason for r in self.rule_results if r.passed and r.weight < 0]