"""OHLCV 输入契约测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from tech_indicators.contract import REQUIRED_COLUMNS, to_standard_ohlcv, validate_ohlcv
from tech_indicators.errors import DataInsufficientError


def _basic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "vol": [100.0, 120.0],
        }
    )


def test_required_columns_defined():
    assert set(REQUIRED_COLUMNS) == {"open", "high", "low", "close", "vol"}


def test_validate_ok():
    validate_ohlcv(_basic_frame())


def test_validate_missing_column_raises():
    with pytest.raises(DataInsufficientError):
        validate_ohlcv(_basic_frame().drop(columns=["vol"]))


def test_validate_short_frame_raises():
    with pytest.raises(DataInsufficientError):
        validate_ohlcv(_basic_frame(), min_rows=3)


def test_validate_nan_close_raises():
    frame = _basic_frame()
    frame.loc[0, "close"] = None
    with pytest.raises(DataInsufficientError):
        validate_ohlcv(frame)


def test_to_standard_maps_volume_and_sorts():
    raw = pd.DataFrame(
        {
            "trade_date": ["20240103", "20240102"],
            "open": [2.0, 1.0],
            "high": [2.2, 1.2],
            "low": [1.9, 0.9],
            "close": [2.1, 1.1],
            "volume": [200.0, 100.0],  # 非标准列名 → vol
        }
    )
    df = to_standard_ohlcv(raw, sort_by="trade_date")
    assert list(df["trade_date"]) == ["20240102", "20240103"]
    assert "vol" in df.columns and "volume" not in df.columns
    assert list(df["vol"]) == [100.0, 200.0]


def test_to_standard_custom_mapping_wins():
    raw = _basic_frame().rename(columns={"vol": "qty"})
    df = to_standard_ohlcv(raw, column_map={"qty": "vol"})
    assert "vol" in df.columns and "qty" not in df.columns


def test_daily_600519_passes_contract(daily_600519):
    assert len(daily_600519) >= 100
    validate_ohlcv(daily_600519)
    assert list(daily_600519["trade_date"]) == sorted(daily_600519["trade_date"])