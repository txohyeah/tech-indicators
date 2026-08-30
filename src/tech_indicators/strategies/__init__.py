from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in minimal bundled runtimes
    yaml = None

from ..errors import UserInputError
from ..indicators import (
    check_breakout_candle_quality,
    check_box_volume_shrink,
    check_close_above_ma,
    check_close_breakout_box,
    check_consolidation_box,
    check_box_low_not_breaking,
    check_golden_bull_channel,
    check_golden_bull_position_rating,
    check_golden_bull_profile,
    check_ma_convergence,
    check_ma_slope_not_falling,
    check_pre_box_drawdown,
    check_relative_low_position,
    check_trend_not_down_or_macd_divergence,
    check_upside_space,
    check_volume_ratio_between,
)
from ..models import RuleResult, StrategyEvaluation
STRATEGIES_DIR = Path(__file__).resolve().parent / "data"


TURNING_POINT_FALLBACK: dict[str, Any] = {
    "name": "turning_point",
    "display_name": "拐点策略",
    "description": "捕捉趋势由弱转强的临界点。",
    "category": "reversal",
    "core_rules": [1, 2],
    "executable_rules": {
        "min_history_days": 80,
        "reversal_exemptions": [
            {"core_rule": 2, "reason": "拐点策略允许在趋势反转早期不满足完整多头排列，但必须有右侧确认信号"}
        ],
        "hard_filters": [
            {"op": "upper_shadow_ratio_lte", "value": 0.3, "fail_reason": "上影线过长，冲高回落风险较高"},
        ],
        "scoring": [
            {"op": "structure_rising", "lookback": 40, "weight": 25, "reason": "低点结构抬升"},
            {"op": "macd_bottom_divergence", "lookback": 40, "weight": 25, "reason": "MACD 底背离"},
            {"op": "is_yang", "weight": 15, "reason": "阳线确认"},
            {"op": "volume_ratio_gte", "value": 2.0, "weight": 20, "reason": "倍量确认"},
            {
                "op": "volume_ratio_gte",
                "value": 1.5,
                "weight": 10,
                "reason": "温和放量",
                "exclusive_with": [{"volume_ratio_gte": 2.0}],
            },
            {"op": "bias_abs_lte", "indicator": "bias_ma20", "value": 5, "weight": 2, "reason": "乖离率安全"},
            {"op": "bias_abs_gt", "indicator": "bias_ma20", "value": 10, "weight": -20, "reason": "乖离率超过 10%，等待横盘消化后再判断"},
            {"op": "circ_mv_lt", "value": 600000, "weight": -10, "reason": "流通市值低于 60 亿，流动性和控盘风险需折价"},
            {"op": "suspected_controlled_range", "weight": -30, "reason": "半年窄幅震荡且长影线频繁，疑似高度控盘"},
            {"op": "recent_limit_up", "weight": 5, "reason": "10 日内出现涨停，短线资金活跃"},
            {"op": "baogongtou_pattern", "weight": 10, "reason": "包公头形态确认"},
            {"op": "survival_pattern", "min_len": 3, "max_len": 7, "threshold": 3.0, "weight": 10, "reason": "绝地求生形态确认"},
            {"op": "breakout_ma", "targets": ["ma20", "ma60"], "weight": 5, "reason": "突破关键均线"},
        ],
        "grading": {"selected_score": 75, "watch_score": 55},
    },
}


@dataclass(frozen=True)
class Strategy:
    name: str
    display_name: str
    description: str
    category: str
    core_rules: list[int]
    executable_rules: dict[str, Any] | None
    path: Path
    raw: dict[str, Any]

    @property
    def executable(self) -> bool:
        return bool(self.executable_rules)


def load_strategies(directory: Path = STRATEGIES_DIR) -> dict[str, Strategy]:
    strategies: dict[str, Strategy] = {}
    if not directory.exists():
        return {"turning_point": _strategy_from_data(TURNING_POINT_FALLBACK, directory / "turning_point.yaml")}
    for path in sorted(directory.glob("*.yaml")):
        data = _load_yaml(path)
        if not data:
            continue
        strategy = _strategy_from_data(data, path)
        strategies[strategy.name] = strategy
    if "turning_point" not in strategies:
        strategies["turning_point"] = _strategy_from_data(TURNING_POINT_FALLBACK, directory / "turning_point.yaml")
    return strategies


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        if path.name == "turning_point.yaml":
            return TURNING_POINT_FALLBACK
        raise UserInputError("PyYAML is required to load strategy YAML files")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _strategy_from_data(data: dict[str, Any], path: Path) -> Strategy:
    name = data.get("name") or data.get("id")
    if not name:
        raise UserInputError(f"Strategy file missing name: {path}")
    return Strategy(
        name=str(name),
        display_name=str(data.get("display_name") or data.get("name") or name),
        description=str(data.get("description") or ""),
        category=str(data.get("category") or data.get("type") or ""),
        core_rules=[int(item) for item in data.get("core_rules", [])],
        executable_rules=data.get("executable_rules"),
        path=path,
        raw=data,
    )


def get_strategy(name: str) -> Strategy:
    strategies = load_strategies()
    if name not in strategies:
        raise UserInputError(f"Strategy not found: {name}")
    strategy = strategies[name]
    if not strategy.executable:
        raise UserInputError(f"Strategy is not executable: {name}")
    return strategy


class RuleEvaluator:
    def evaluate(self, strategy: Strategy, stock: dict[str, Any], indicators: dict[str, Any]) -> StrategyEvaluation:
        rules = strategy.executable_rules or {}
        warnings = list(indicators.get("warnings") or [])
        exclude_reasons: list[str] = []
        rule_results: list[RuleResult] = []

        min_history = int(rules.get("min_history_days") or 0)
        if indicators.get("history_days", 0) < min_history:
            warnings.append(f"历史 K 线不足 {min_history} 日")

        for rule in rules.get("hard_filters", []):
            result = self._evaluate_rule(rule, indicators)
            result.weight = 0
            rule_results.append(result)
            if result.status == "unavailable":
                warnings.append(f"{result.name} 无法验证")
            elif not result.passed:
                exclude_reasons.append(rule.get("fail_reason") or result.reason)

        score = int(rules.get("base_score") or 0)
        if score:
            rule_results.append(
                RuleResult(
                    "base_score",
                    True,
                    str(rules.get("base_score_reason") or "候选基础分"),
                    weight=score,
                )
            )
        applied_ops: set[str] = set()
        for rule in rules.get("scoring", []):
            if self._is_excluded_by_previous(rule, applied_ops):
                continue
            result = self._evaluate_rule(rule, indicators)
            result.weight = int(rule.get("weight") or 0)
            rule_results.append(result)
            if result.status == "unavailable":
                warnings.append(f"{result.name} 无法验证")
                continue
            if result.passed:
                score += result.weight
                applied_ops.add(self._rule_key(rule))

        if rules.get("score_floor") is not None:
            score = max(score, int(rules["score_floor"]))
        if rules.get("score_ceiling") is not None:
            score = min(score, int(rules["score_ceiling"]))

        core_results = self._core_rule_results(strategy, indicators, rules)
        for result in core_results:
            if result.status == "unavailable":
                warnings.append(f"核心规则 {result.name} 未验证")

        selected_score = int((rules.get("grading") or {}).get("selected_score", 75))
        watch_score = int((rules.get("grading") or {}).get("watch_score", 55))
        if exclude_reasons:
            bucket = "excluded"
        elif score >= selected_score:
            bucket = "selected"
        elif score >= watch_score:
            bucket = "watch"
        else:
            bucket = "excluded"
            exclude_reasons.append(f"分数低于观察阈值 {watch_score}")

        return StrategyEvaluation(
            code=str(stock.get("code") or stock.get("symbol") or ""),
            ts_code=str(stock.get("ts_code") or ""),
            name=str(stock.get("name") or stock.get("ts_code") or ""),
            score=score,
            grade=_grade(score),
            bucket=bucket,
            close=indicators.get("close"),
            pct_chg=indicators.get("pct_chg"),
            indicators={k: v for k, v in indicators.items() if k != "_series"},
            rule_results=rule_results,
            core_rule_results=core_results,
            warnings=warnings,
            exclude_reasons=exclude_reasons,
        )

    def _evaluate_rule(self, rule: dict[str, Any], indicators: dict[str, Any]) -> RuleResult:
        op = str(rule.get("op"))
        reason = str(rule.get("reason") or rule.get("fail_reason") or op)

        if op == "bias_abs_lte":
            value = indicators.get(str(rule.get("indicator", "bias_ma20")))
            return _compare_abs_lte(op, reason, value, float(rule.get("value")))
        if op == "bias_abs_gt":
            value = indicators.get(str(rule.get("indicator", "bias_ma20")))
            return _compare_abs_gt(op, reason, value, float(rule.get("value")))
        if op == "upper_shadow_ratio_lte":
            return _compare_lte(op, reason, indicators.get("upper_shadow_ratio"), float(rule.get("value")))
        if op == "structure_rising":
            return _nested_bool(op, reason, indicators.get("structure_rising"))
        if op == "macd_bottom_divergence":
            return _nested_bool(op, reason, indicators.get("macd_bottom_divergence"))
        if op == "is_yang":
            return _bool_value(op, reason, indicators.get("is_yang"))
        if op == "volume_ratio_gte":
            return _compare_gte(op, reason, indicators.get("volume_ratio_calc"), float(rule.get("value")))
        if op == "circ_mv_lt":
            return _compare_lt(op, reason, indicators.get("circ_mv"), float(rule.get("value")))
        if op == "breakout_ma":
            return _nested_bool(op, reason, indicators.get("breakout_ma"))
        if op == "consolidation_box":
            value = _with_series(
                indicators,
                check_consolidation_box,
                indicators.get("consolidation_box"),
                min_len=int(rule.get("min_len", 5)),
                max_len=int(rule.get("max_len", 20)),
                max_range_pct=float(rule.get("max_range_pct", 18.0)),
            )
            return _nested_bool(op, reason, value)
        if op == "close_breakout_box":
            value = _with_series(
                indicators,
                check_close_breakout_box,
                indicators.get("close_breakout_box"),
                min_len=int(rule.get("min_len", 5)),
                max_len=int(rule.get("max_len", 20)),
                max_range_pct=float(rule.get("max_range_pct", 18.0)),
                min_breakout_pct=float(rule.get("min_breakout_pct", 0.5)),
            )
            return _nested_bool(op, reason, value)
        if op == "volume_ratio_between":
            value = _with_series(
                indicators,
                check_volume_ratio_between,
                indicators.get("breakout_volume"),
                min_ratio=float(rule.get("min", 1.3)),
                max_ratio=float(rule.get("max", 3.5)),
            )
            return _nested_bool(op, reason, value)
        if op == "ideal_volume_ratio":
            value = _with_series(
                indicators,
                check_volume_ratio_between,
                indicators.get("ideal_volume_ratio"),
                min_ratio=float(rule.get("min", 1.6)),
                max_ratio=float(rule.get("max", 2.8)),
            )
            return _nested_bool(op, reason, value)
        if op == "trend_not_down_or_macd_divergence":
            value = _with_series(
                indicators,
                check_trend_not_down_or_macd_divergence,
                indicators.get("trend_not_down_or_macd_divergence"),
                trend_lookback=int(rule.get("trend_lookback", 60)),
                divergence_lookback=int(rule.get("divergence_lookback", 60)),
            )
            return _nested_bool(op, reason, value)
        if op == "close_above_ma":
            target = str(rule.get("target") or "ma20")
            value = _with_series(
                indicators,
                check_close_above_ma,
                indicators.get(f"close_above_{target}"),
                target=target,
            )
            return _nested_bool(op, reason, value)
        if op == "pre_box_drawdown_lte":
            value = _with_series(
                indicators,
                check_pre_box_drawdown,
                indicators.get("pre_box_drawdown"),
                lookback=int(rule.get("lookback", 30)),
                max_drawdown_pct=float(rule.get("max_drawdown_pct", 22.0)),
            )
            return _nested_bool(op, reason, value)
        if op == "box_low_not_breaking":
            value = _with_series(
                indicators,
                check_box_low_not_breaking,
                indicators.get("box_low_not_breaking"),
                tolerance_pct=float(rule.get("tolerance_pct", 2.0)),
            )
            return _nested_bool(op, reason, value)
        if op == "ma_slope_not_falling":
            target = str(rule.get("target") or "ma20")
            value = _with_series(
                indicators,
                check_ma_slope_not_falling,
                indicators.get(f"{target}_not_falling"),
                target=target,
                lookback=int(rule.get("lookback", 10)),
                max_down_slope_pct=float(rule.get("max_down_slope_pct", 2.0)),
            )
            return _nested_bool(op, reason, value)
        if op == "box_volume_shrink":
            value = _with_series(indicators, check_box_volume_shrink, indicators.get("box_volume_shrink"))
            return _nested_bool(op, reason, value)
        if op == "ma_convergence":
            targets = tuple(str(target) for target in rule.get("targets", ["ma5", "ma10", "ma20"]))
            value = _with_series(
                indicators,
                check_ma_convergence,
                indicators.get("ma_convergence"),
                targets=targets,
                max_spread_pct=float(rule.get("max_spread_pct", 8.0)),
            )
            return _nested_bool(op, reason, value)
        if op == "breakout_candle_quality":
            value = _with_series(
                indicators,
                check_breakout_candle_quality,
                indicators.get("breakout_candle_quality"),
            )
            return _nested_bool(op, reason, value)
        if op == "relative_low_position":
            value = _with_series(
                indicators,
                check_relative_low_position,
                indicators.get("relative_low_position"),
                lookback=int(rule.get("lookback", 60)),
                max_gain_from_low_pct=float(rule.get("max_gain_from_low_pct", 35.0)),
            )
            return _nested_bool(op, reason, value)
        if op == "upside_space":
            value = _with_series(
                indicators,
                check_upside_space,
                indicators.get("upside_space"),
                lookback=int(rule.get("lookback", 120)),
                min_space_pct=float(rule.get("min_space_pct", 8.0)),
            )
            return _nested_bool(op, reason, value)
        if op == "survival_pattern":
            return _nested_bool(op, reason, indicators.get("survival_pattern"))
        if op == "suspected_controlled_range":
            return _nested_bool(op, reason, indicators.get("suspected_controlled_range"))
        if op == "recent_limit_up":
            return _nested_bool(op, reason, indicators.get("recent_limit_up"))
        if op == "baogongtou_pattern":
            return _nested_bool(op, reason, indicators.get("baogongtou_pattern"))
        if op == "duanxian_signal":
            signal = str(rule.get("signal") or "")
            value = indicators.get("duanxian_auxiliary")
            if signal:
                value = ((value or {}).get("signals") or {}).get(signal) if isinstance(value, dict) else None
            return _nested_bool(op if not signal else f"{op}:{signal}", reason, value)
        if op == "golden_bull_signal":
            value = _with_series(
                indicators,
                check_golden_bull_channel,
                indicators.get("golden_bull_channel"),
            )
            return _nested_bool(op, reason, value)
        if op == "golden_bull_profile":
            criterion = str(rule.get("criterion") or "")
            value = _with_series(
                indicators,
                check_golden_bull_profile,
                indicators.get("golden_bull_profile"),
            )
            if criterion:
                value = value.get(criterion) if isinstance(value, dict) else value
            return _nested_bool(op if not criterion else f"{op}:{criterion}", reason, value)
        if op == "golden_bull_position_rating":
            value = _with_series(
                indicators,
                check_golden_bull_position_rating,
                indicators.get("golden_bull_position_rating"),
            )
            min_strength = int(rule.get("min_strength") or 1)
            if not isinstance(value, dict) or value.get("passed") is None:
                return RuleResult(op, False, reason, value=value, status="unavailable")
            ratings = value.get("ratings") or []
            passed = any(int(item.get("strength") or 0) >= min_strength for item in ratings if isinstance(item, dict))
            return RuleResult(op, passed, reason, value=value, status="passed" if passed else "failed")
        return RuleResult(op, False, f"未知规则算子: {op}", status="unavailable")

    def _core_rule_results(
        self,
        strategy: Strategy,
        indicators: dict[str, Any],
        rules: dict[str, Any],
    ) -> list[RuleResult]:
        results: list[RuleResult] = []
        exemptions = {int(item.get("core_rule")): item for item in rules.get("reversal_exemptions", []) if item.get("core_rule")}
        for rule_id in strategy.core_rules:
            if rule_id == 1:
                results.append(_compare_abs_lte("1 严进策略", "乖离率 < 5%", indicators.get("bias_ma20"), 5.0))
            elif rule_id == 2:
                if rule_id in exemptions:
                    results.append(RuleResult("2 趋势交易", True, str(exemptions[rule_id].get("reason")), status="exempted"))
                else:
                    results.append(_bool_value("2 趋势交易", "MA5 > MA10 > MA20", indicators.get("ma_bullish")))
            elif rule_id == 6:
                results.append(_nested_bool("6 量价配合", "突破伴随温和放量", indicators.get("breakout_volume")))
            elif rule_id in {5, 7}:
                results.append(RuleResult(str(rule_id), False, "v1 数据源暂不支持自动验证", status="unavailable"))
            else:
                results.append(RuleResult(str(rule_id), False, "该核心规则暂未接入当前策略检查", status="unavailable"))
        return results

    def _is_excluded_by_previous(self, rule: dict[str, Any], applied_ops: set[str]) -> bool:
        for exclusive in rule.get("exclusive_with", []) or []:
            if isinstance(exclusive, dict):
                for key, value in exclusive.items():
                    if f"{key}:{value}" in applied_ops:
                        return True
        return False

    def _rule_key(self, rule: dict[str, Any]) -> str:
        if "value" in rule:
            return f"{rule.get('op')}:{rule.get('value')}"
        return str(rule.get("op"))


def _compare_abs_lte(name: str, reason: str, value: Any, threshold: float) -> RuleResult:
    if value is None:
        return RuleResult(name, False, reason, value=value, status="unavailable")
    passed = abs(float(value)) <= threshold
    return RuleResult(name, passed, reason, value=value, status="passed" if passed else "failed")


def _with_series(indicators: dict[str, Any], func: Any, fallback: Any, **kwargs: Any) -> Any:
    series = indicators.get("_series")
    if series is None:
        return fallback
    return func(series, **kwargs)


def _compare_abs_gt(name: str, reason: str, value: Any, threshold: float) -> RuleResult:
    if value is None:
        return RuleResult(name, False, reason, value=value, status="unavailable")
    passed = abs(float(value)) > threshold
    return RuleResult(name, passed, reason, value=value, status="passed" if passed else "failed")


def _compare_lte(name: str, reason: str, value: Any, threshold: float) -> RuleResult:
    if value is None:
        return RuleResult(name, False, reason, value=value, status="unavailable")
    passed = float(value) <= threshold
    return RuleResult(name, passed, reason, value=value, status="passed" if passed else "failed")


def _compare_gte(name: str, reason: str, value: Any, threshold: float) -> RuleResult:
    if value is None:
        return RuleResult(name, False, reason, value=value, status="unavailable")
    passed = float(value) >= threshold
    return RuleResult(name, passed, reason, value=value, status="passed" if passed else "failed")


def _compare_lt(name: str, reason: str, value: Any, threshold: float) -> RuleResult:
    if value is None:
        return RuleResult(name, False, reason, value=value, status="unavailable")
    passed = float(value) < threshold
    return RuleResult(name, passed, reason, value=value, status="passed" if passed else "failed")


def _bool_value(name: str, reason: str, value: Any) -> RuleResult:
    if value is None:
        return RuleResult(name, False, reason, value=value, status="unavailable")
    return RuleResult(name, bool(value), reason, value=value, status="passed" if value else "failed")


def _nested_bool(name: str, reason: str, value: Any) -> RuleResult:
    if not isinstance(value, dict) or value.get("passed") is None:
        return RuleResult(name, False, reason, value=value, status="unavailable")
    passed = bool(value.get("passed"))
    return RuleResult(name, passed, reason, value=value, status="passed" if passed else "failed")


def _grade(score: int) -> str:
    if score >= 90:
        return "S"
    if score >= 75:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"
