# tech-indicators

A 股 / Crypto 通用的技术指标、交易计划与策略评分库。从 stock-research 拆分而来，
纯计算、零 IO：输入 OHLCV DataFrame，输出指标 / 信号 / 评分，由调用方负责取数与持久化。

## 模块

| 模块 | 职责 |
|------|------|
| `indicators` | 核心指标计算（金牛通道/评级、绝地求生、短线是银、MACD 背离等） |
| `golden_bull_trading` | 金牛交易计划生成 |
| `reburn` | 复燃点信号与交易计划：信号 = `CROSS(RSI6, 40)`（RSI 自下而上穿越弱势触发线），叠加空头/MA20↓/MA60↓ 风险闸门与量能分级仓位 |
| `ignition` | 起爆点择时：`CROSS(RSI6,40)` 信号 + 三道买入过滤（跌够深/已离底/起爆K线形态干净）+ 完整卖出状态机（结构止损、新起爆点重锚、浮盈 20% 后按回撤 25% 移动止盈、金牛上沿压制减半或熊市通道清仓）。三道过滤分开暴露，默认组成非唯一合理搭配——形态过滤已证实纯赚，位置过滤代价高昂（见模块 docstring），是否保留待拍板 |
| `kline_decision` | 统一 K 线交易计划（组合 golden_bull + reburn） |
| `chart` | HTML K 线渲染（零依赖，纯字符串拼接） |
| `strategies` | 策略定义（yaml）+ 规则评分引擎（Strategy / RuleEvaluator） |
| `contract` | OHLCV 输入契约校验与列名映射 |
| `errors` | 包内异常基类（TechIndicatorsError 等） |
| `models` | 策略评分数据结构（RuleResult / StrategyEvaluation） |

## 输入契约

所有计算入口接受 pandas DataFrame，必需列（tushare daily 风格）：

    open / high / low / close / vol

常用可选列：`amount` / `pct_chg` / `pre_close` / `trade_date`。

调用方负责把自有列名映射到标准列名；`contract.to_standard_ohlcv(df, sort_by="trade_date")`
提供轻量重命名 + 升序排序辅助，`contract.validate_ohlcv(df)` 做严格校验。

## 使用示例

```python
import pandas as pd
from tech_indicators import compute_indicators, get_strategy
from tech_indicators.contract import to_standard_ohlcv

df = to_standard_ohlcv(raw_df, sort_by="trade_date")      # 标准化 + 按日期升序
indicators = compute_indicators(df)                        # 指标字典
strategy = get_strategy("turning_point")                   # 加载策略定义
evaluation = RuleEvaluator().evaluate(strategy, {"ts_code": "600519.SH"}, indicators)
```

## 开发

```bash
python3 -m venv venv && venv/bin/pip install -e ".[dev]"
venv/bin/pytest
```