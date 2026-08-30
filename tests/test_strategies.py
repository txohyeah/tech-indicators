"""策略定义与评分引擎测试。"""

from __future__ import annotations

import pytest

from tech_indicators.errors import UserInputError
from tech_indicators.indicators import compute_indicators
from tech_indicators.strategies import RuleEvaluator, get_strategy, load_strategies

EXPECTED_STRATEGIES = {
    "breakout_point": "突破点策略",
    "golden_bull_channel": "金牛通道",
    "golden_bull_position_rating": "金牛通道持仓评级",
    "golden_bull_refine": "金牛通道二次评分",
    "turning_point": "拐点策略",
}


def test_load_strategies_has_all_five():
    strategies = load_strategies()
    assert {k: v.display_name for k, v in strategies.items()} == EXPECTED_STRATEGIES


def test_get_strategy_known_name():
    strategy = get_strategy("turning_point")
    assert strategy.name == "turning_point"
    assert strategy.executable is True
    assert strategy.category == "reversal"
    assert strategy.core_rules == [1, 2]


def test_get_strategy_unknown_raises():
    with pytest.raises(UserInputError):
        get_strategy("no_such_strategy")


def test_evaluate_turning_point(daily_600519):
    ind = compute_indicators(daily_600519)
    evaluator = RuleEvaluator()
    evaluation = evaluator.evaluate(
        get_strategy("turning_point"),
        {"code": "600519", "ts_code": "600519.SH", "name": "贵州茅台"},
        ind,
    )
    assert evaluation.name == "贵州茅台"  # StrategyEvaluation.name 是证券名称
    assert evaluation.ts_code == "600519.SH"
    assert evaluation.bucket in ("selected", "watch", "excluded")
    assert evaluation.grade in ("S", "A", "B", "C", "D")
    assert len(evaluation.rule_results) >= 5


def test_evaluate_all_strategies_smoke(daily_600519):
    ind = compute_indicators(daily_600519)
    evaluator = RuleEvaluator()
    for name in EXPECTED_STRATEGIES:
        evaluation = evaluator.evaluate(get_strategy(name), {"ts_code": "600519.SH"}, ind)
        assert evaluation.bucket in ("selected", "watch", "excluded"), name
        assert evaluation.score >= 0, name


def test_evaluation_support_fields(daily_600519):
    ind = compute_indicators(daily_600519)
    evaluation = RuleEvaluator().evaluate(get_strategy("golden_bull_channel"), {"ts_code": "600519.SH"}, ind)
    assert isinstance(evaluation.warnings, list)
    assert isinstance(evaluation.exclude_reasons, list)
    assert isinstance(evaluation.hit_reasons, list)
    assert isinstance(evaluation.penalty_reasons, list)