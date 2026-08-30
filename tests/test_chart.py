"""HTML K 线渲染测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from tech_indicators.chart import (
    ChartSeriesConfig,
    build_chart_frame,
    format_symbol,
    parse_chart_indicators,
    render_chart_html,
)


def test_format_symbol_a_share():
    assert format_symbol("600519.SH") == "600519"
    assert format_symbol("000001.SZ") == "000001"
    assert format_symbol("920001.BJ") == "920001"


def test_format_symbol_crypto_passthrough():
    assert format_symbol("BTCUSDT") == "BTCUSDT"
    assert format_symbol("ETHUSDT") == "ETHUSDT"


def test_parse_chart_indicators_defaults():
    config = parse_chart_indicators(None)
    assert isinstance(config, ChartSeriesConfig)
    assert config.golden_bull is True


def test_parse_chart_indicators_ma_only():
    config = parse_chart_indicators("ma")
    assert config.golden_bull is False
    assert config.requested == ("ma",)


def test_build_chart_frame_adds_columns(daily_600519):
    config = ChartSeriesConfig(ma_periods=(5, 10, 20), volume_ma=True, golden_bull=True, requested=("ma", "golden_bull"))
    frame = build_chart_frame(daily_600519, config)
    assert len(frame) == len(daily_600519)
    for col in ("ma5", "ma10", "ma20", "vol_ma5", "golden_bull"):
        assert col in frame.columns, f"missing column {col}"


def test_render_chart_html_output(daily_600519):
    config = ChartSeriesConfig(ma_periods=(5, 10, 20), volume_ma=True, golden_bull=True, requested=("ma", "golden_bull"))
    frame = build_chart_frame(daily_600519, config)
    html = render_chart_html(frame, {}, config)
    assert len(html) > 1000
    assert "<html" in html.lower() or "<!DOCTYPE" in html.lower()


def test_render_chart_html_with_markers(daily_600519):
    from tech_indicators.chart import build_reburn_markers

    config = ChartSeriesConfig(ma_periods=(5, 10, 20), volume_ma=True, golden_bull=True, requested=("ma", "golden_bull"))
    frame = build_chart_frame(daily_600519, config)
    markers = build_reburn_markers(frame, timeframe_label="1d")
    html = render_chart_html(frame, {}, config, markers=markers)
    assert len(html) > 1000