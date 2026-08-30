"""OHLCV 输入契约。

纯计算包的输入统一约定为 DataFrame，必须包含以下列（tushare daily 风格命名）：

    open / high / low / close / vol

常见可选列：

    amount / pct_chg / pre_close / trade_date

调用方（stock-analytics 的 A 股 daily、crypto-research 的 K 线等）负责把
自有列名映射到标准列名后传入；`to_standard_ohlcv` 提供轻量重命名辅助。
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .errors import DataInsufficientError

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "vol")
OPTIONAL_COLUMNS: tuple[str, ...] = ("amount", "pct_chg", "pre_close", "trade_date")

# 常见来源到标准列名的映射（调用方可按需覆盖）
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "volume": "vol",
    "quote_volume": "amount",
}


def validate_ohlcv(df: pd.DataFrame, *, min_rows: int = 1) -> None:
    """校验 DataFrame 是否满足 OHLCV 契约，不满足抛 DataInsufficientError。"""
    if not isinstance(df, pd.DataFrame):
        raise DataInsufficientError("Input must be a pandas DataFrame")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataInsufficientError(
            f"Missing required OHLCV columns: {missing}; got {list(df.columns)}"
        )
    if len(df) < min_rows:
        raise DataInsufficientError(
            f"Not enough rows: {len(df)} < {min_rows}",
            payload={"required": min_rows, "actual": len(df)},
        )
    if df["close"].isna().any():
        raise DataInsufficientError("close column contains NaN values")


def to_standard_ohlcv(
    df: pd.DataFrame,
    *,
    column_map: Mapping[str, str] | None = None,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """按列名映射重命名并规范化（可选按日期列升序排序）。

    - column_map：源列名 → 标准列名，缺省用 DEFAULT_COLUMN_MAP
    - sort_by：若有 trade_date/date 等日期列，建议传入以升序排序（指标计算依赖时间序）
    """
    mapping = dict(DEFAULT_COLUMN_MAP)
    if column_map:
        mapping.update(column_map)
    frame = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
    validate_ohlcv(frame)
    return frame