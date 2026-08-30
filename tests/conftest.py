"""pytest 共享 fixture：加载 600519 真实日线样本。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def daily_600519_raw() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / "600519_daily.csv")


@pytest.fixture(scope="session")
def daily_600519() -> pd.DataFrame:
    """标准 OHLCV（按 trade_date 升序，列名 = open/high/low/close/vol/amount/...）。"""
    from tech_indicators.contract import to_standard_ohlcv

    raw = pd.read_csv(FIXTURES_DIR / "600519_daily.csv")
    return to_standard_ohlcv(raw, sort_by="trade_date")