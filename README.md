# tech-indicators

A 股 / Crypto 通用的技术指标、交易计划与策略评分库。从 stock-research 拆分而来，
纯计算、零 IO：输入 OHLCV DataFrame，输出指标 / 信号 / 评分，由调用方负责取数与持久化。

## 模块

| 模块 | 职责 |
|------|------|
| `indicators` | 核心指标计算（金牛通道/评级、绝地求生、短线是银、MACD 背离等）。⚠️ 金牛通道默认与通达信画图一致，**含未来函数**（居中 XMA 会用后 12 根数据，图上最近 12 根还会随行情漂移）；做回测或"当时能否看到"的判断时必须传 `causal=True` |
| `golden_bull_trading` | 金牛交易计划生成 |
| `reburn` | 复燃点信号与交易计划：信号 = `CROSS(RSI6, 40)`（RSI 自下而上穿越弱势触发线），叠加空头/MA20↓/MA60↓ 风险闸门与量能分级仓位 |
| `ignition` | 起爆点择时：`CROSS(RSI6,40)` 信号 + 三道买入过滤（跌够深/已离底/起爆K线形态干净）+ 卖出状态机。**2026-09-05 因果版定稿**（15 槽组合层 8 面板实测，见模块 docstring）：卖出默认 = 撞金牛上沿·全清 + 滚动止损线 `max(买价×0.90, 截至昨日最近 30 根最低价)`（窗口不含当根）；移动止盈、新起爆点重锚、熊市通道清仓均已删除或默认关闭（实测更差/未复验）。三道买入条件分开暴露。`golden_channel_state` 默认仍是非因果口径；**ignition 模块内部已全部固定传 `causal=True`**，其他模块直接调用时回测仍需自己显式传。**卖出算法唯一实现 = `IgnitionPosition`**（2026-09-06）：单票状态机 `ignition_position_state` 与 stock-analytics 回测内核都推它，规则只改这一处；止损线公式独立暴露为 `ignition_stop_line` |
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