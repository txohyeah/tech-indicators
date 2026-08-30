from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import DataInsufficientError, ReportWriteError, UserInputError
from .indicators import compute_golden_bull_lines
from .reburn import reburn_signal


def format_symbol(code: str) -> str:
    """人类可读符号：A 股 600519.SH → 600519，其余（如 BTCUSDT）原样返回。

    仅用于展示/文件名；ts_code 的规范化解析由调用方负责。"""
    base, sep, suffix = code.rpartition(".")
    if sep and suffix in {"SH", "SZ", "BJ"}:
        return base
    return code


DEFAULT_INDICATORS = "ma,golden_bull"
KNOWN_INDICATORS = {"ma", "ma5", "ma10", "ma20", "ma60", "volume_ma", "golden_bull"}
TRADE_MARKER_HIT_RADIUS = 9
REBURN_MARKER_HIT_RADIUS = 11
TRADE_MARKER_Y_OFFSET = 18
REBURN_MARKER_Y_OFFSET = 30
TRADE_MARKER_STACK_GAP = 15
REBURN_MARKER_STACK_GAP = 14


@dataclass(frozen=True)
class ChartSeriesConfig:
    ma_periods: tuple[int, ...]
    volume_ma: bool
    golden_bull: bool
    requested: tuple[str, ...]


def run_chart(
    repository,
    *,
    code: str,
    ts_code: str | None = None,
    requested_date: str = "latest",
    lookback_days: int = 120,
    indicators: str = DEFAULT_INDICATORS,
    output_path: str | None = None,
) -> dict[str, object]:
    if lookback_days <= 0:
        raise UserInputError("--lookback-days must be greater than 0")

    fetch_code = ts_code or code
    display_code = format_symbol(code)
    indicator_config = parse_chart_indicators(indicators)
    trade_date = repository.resolve_trade_date(_compact_chart_date(requested_date))
    start_date = repository.resolve_start_date(trade_date, lookback_days)
    rows = repository.fetch_stock_history([fetch_code], start_date, trade_date)
    if not rows:
        raise DataInsufficientError(
            f"No daily bars found for {fetch_code}",
            payload={"code": fetch_code, "trade_date": trade_date},
        )

    frame = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    latest_date = str(frame.iloc[-1]["trade_date"])
    if latest_date != trade_date:
        raise DataInsufficientError(
            f"Daily bars for {fetch_code} are not synced to {trade_date}",
            payload={
                "code": fetch_code,
                "trade_date": trade_date,
                "latest_bar_date": latest_date,
            },
        )

    chart_frame = build_chart_frame(frame, indicator_config)
    markers = build_reburn_markers(chart_frame, timeframe_label="1d")
    meta = _chart_meta(frame, fetch_code, trade_date, start_date)
    output = Path(output_path) if output_path else _default_chart_output(display_code, trade_date)
    html = render_chart_html(chart_frame, meta, indicator_config, markers=markers)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
    except OSError as exc:
        raise ReportWriteError(str(exc), payload={"output": str(output)}) from exc

    return {
        "ok": True,
        "command": "chart",
        "code": fetch_code,
        "trade_date": trade_date,
        "start_date": start_date,
        "bar_count": int(len(chart_frame)),
        "marker_count": len(markers),
        "indicators": list(indicator_config.requested),
        "output": str(output.resolve()),
    }


def parse_chart_indicators(value: str | None) -> ChartSeriesConfig:
    tokens = tuple(item.strip().lower() for item in (value or DEFAULT_INDICATORS).split(",") if item.strip())
    if not tokens:
        raise UserInputError("--indicators must include at least one indicator")
    unknown = sorted(set(tokens) - KNOWN_INDICATORS)
    if unknown:
        raise UserInputError(f"Unsupported chart indicators: {', '.join(unknown)}")

    periods: set[int] = set()
    if "ma" in tokens:
        periods.update([5, 10, 20, 60])
    for token in tokens:
        if token.startswith("ma") and token != "ma":
            periods.add(int(token[2:]))
    return ChartSeriesConfig(
        ma_periods=tuple(sorted(periods)),
        volume_ma="volume_ma" in tokens,
        golden_bull="golden_bull" in tokens,
        requested=tokens,
    )


def build_chart_frame(frame: pd.DataFrame, config: ChartSeriesConfig) -> pd.DataFrame:
    data = frame.sort_values("trade_date").copy().reset_index(drop=True)
    for column in ["open", "high", "low", "close", "vol"]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    required = ["trade_date", "open", "high", "low", "close", "vol"]
    missing = [column for column in required if column not in data]
    if missing:
        raise DataInsufficientError(f"Daily bars missing required columns: {', '.join(missing)}")
    if data[["open", "high", "low", "close"]].isna().any().any():
        raise DataInsufficientError("Daily bars contain incomplete OHLC values")

    for period in config.ma_periods:
        data[f"ma{period}"] = data["close"].rolling(period, min_periods=period).mean()
    if config.volume_ma:
        data["vol_ma5"] = data["vol"].rolling(5, min_periods=5).mean()
    if config.golden_bull:
        lines = compute_golden_bull_lines(data)
        for column in lines.columns:
            data[column] = lines[column].values
    return data


def render_chart_html(
    frame: pd.DataFrame,
    meta: dict[str, object],
    config: ChartSeriesConfig,
    *,
    markers: list[dict[str, Any]] | None = None,
    kline_details: dict[int, dict[str, Any]] | None = None,
) -> str:
    markers = _markers_with_ids(markers or [])
    kline_details = kline_details or {}
    svg = _render_svg(frame, meta, config, markers=markers)
    chart_block = _chart_block(svg, markers, frame=frame, meta=meta, config=config, kline_details=kline_details)
    title = f"{meta.get('code')} {meta.get('name') or ''} 日K"
    latest = frame.iloc[-1]
    summary = (
        f"最新 {latest['trade_date']} "
        f"开 {latest['open']:.2f} 高 {latest['high']:.2f} 低 {latest['low']:.2f} 收 {latest['close']:.2f}"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; font-family: Arial, "Microsoft YaHei", sans-serif; }}
    body {{ margin: 0; background: #050505; color: #d9d9d9; }}
    main {{ width: min(1500px, calc(100vw - 24px)); margin: 0 auto; padding: 18px 0 24px; }}
    h1 {{ margin: 0 0 4px; font-size: 20px; font-weight: 700; letter-spacing: 0; }}
    p {{ margin: 0 0 12px; color: #a8a8a8; font-size: 13px; }}
    svg {{ width: 100%; height: auto; display: block; background: #050505; border: 1px solid #2a0000; }}
    .chart-layout {{ display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 14px; align-items: start; }}
    .chart-tools {{ display: flex; justify-content: flex-end; align-items: center; gap: 10px; margin: 0 0 8px; color: #b8b8b8; font-size: 12px; }}
    .chart-tools label {{ display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }}
    .chart-tools input {{ width: 150px; accent-color: #ff3030; }}
    .chart-window-label {{ min-width: 120px; text-align: right; color: #d8d8d8; }}
    .trade-detail {{ min-height: 220px; border: 1px solid #2a0000; background: #090909; padding: 12px; box-sizing: border-box; }}
    .trade-detail-header {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin: 0 0 10px; }}
    .trade-detail h2 {{ margin: 0; font-size: 14px; color: #e8e8e8; letter-spacing: 0; }}
    .trade-nav {{ display: inline-flex; gap: 6px; }}
    .trade-nav button {{ width: 28px; height: 24px; border: 1px solid #4a0000; background: #120000; color: #e8e8e8; cursor: pointer; font-size: 14px; line-height: 1; }}
    .trade-nav button:disabled {{ color: #555; cursor: default; opacity: 0.55; }}
    .trade-nav button:not(:disabled):hover, .trade-nav button:not(:disabled):focus {{ border-color: #ff3030; outline: none; }}
    .trade-detail dl {{ margin: 0; display: grid; grid-template-columns: 86px minmax(0, 1fr); gap: 8px 10px; font-size: 12px; }}
    .trade-detail dt {{ color: #888; }}
    .trade-detail dd {{ margin: 0; color: #d8d8d8; word-break: break-word; white-space: pre-wrap; }}
    .trade-marker {{ cursor: pointer; outline: none; }}
    .trade-marker:focus text, .trade-marker.is-selected text {{ stroke: #ffffff; stroke-width: 0.8; paint-order: stroke; }}
    .kline-candle {{ cursor: crosshair; outline: none; }}
    .kline-candle:focus rect, .kline-candle.is-selected rect {{ stroke: #ffffff; stroke-width: 2; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0 0; font-size: 12px; color: #c8c8c8; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 18px; height: 3px; display: inline-block; }}
    @media (max-width: 900px) {{ .chart-layout {{ grid-template-columns: 1fr; }} .trade-detail {{ min-height: 0; }} }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(str(title))}</h1>
    <p>{escape(summary)}</p>
    {chart_block}
    <div class="legend">{_legend_html(config, has_markers=bool(markers))}</div>
  </main>
  {_trade_detail_script(markers, frame=frame, meta=meta, config=config, kline_details=kline_details)}
</body>
</html>
"""


def _chart_block(
    svg: str,
    markers: list[dict[str, Any]],
    *,
    frame: pd.DataFrame,
    meta: dict[str, object],
    config: ChartSeriesConfig,
    kline_details: dict[int, dict[str, Any]] | None = None,
) -> str:
    if not markers and not kline_details:
        return svg
    count = len(frame)
    min_window = min(20, count)
    default_window = min(160, count)
    start_value = max(count - default_window, 0)
    return f"""
    <div class="chart-layout">
      <div class="chart-area">
        <div class="chart-tools">
          <label for="chart-window-size">\u663e\u793aK\u7ebf
            <input id="chart-window-size" type="range" min="{min_window}" max="{count}" value="{default_window}" step="1">
          </label>
          <label for="chart-window-start">\u7a97\u53e3
            <input id="chart-window-start" type="range" min="0" max="{start_value}" value="{start_value}" step="1">
          </label>
          <span class="chart-window-label" id="chart-window-label"></span>
        </div>
        <div id="chart-svg-container">{svg}</div>
      </div>
      <aside class="trade-detail" id="trade-detail" aria-live="polite">
        <div class="trade-detail-header">
          <h2>\u590d\u76d8\u8be6\u60c5</h2>
          <div class="trade-nav" aria-label="\u4ea4\u6613\u5207\u6362">
            <button id="trade-prev" type="button" title="\u4e0a\u4e00\u7b14\u4ea4\u6613" aria-label="\u4e0a\u4e00\u7b14\u4ea4\u6613">\u2039</button>
            <button id="trade-next" type="button" title="\u4e0b\u4e00\u7b14\u4ea4\u6613" aria-label="\u4e0b\u4e00\u7b14\u4ea4\u6613">\u203a</button>
          </div>
        </div>
        <dl id="trade-detail-list">
          <dt>\u63d0\u793a</dt><dd>\u70b9\u51fbK\u7ebf\u6216\u4e70\u5356\u7bad\u5934\u67e5\u770b\u660e\u7ec6</dd>
        </dl>
      </aside>
    </div>
"""


def _markers_with_ids(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        item = dict(marker)
        item["id"] = str(item.get("id") or f"trade-{index}")
        result.append(item)
    return result


def build_reburn_markers(frame: pd.DataFrame, *, timeframe_label: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    if frame.empty or "close" not in frame:
        return markers
    for idx in range(len(frame)):
        history = frame.iloc[: idx + 1]
        if not reburn_signal(history):
            continue
        row = frame.iloc[idx]
        price = row.get("close")
        if not _is_finite_number(price):
            continue
        marker: dict[str, Any] = {
            "marker_kind": "reburn",
            "label": "R",
            "marker_name": "Reburn low point",
            "time": str(row.get("trade_date")),
            "side": "reburn",
            "action": "signal",
            "price": float(price),
            "timeframe": timeframe_label,
        }
        if "open_time_ms" in row and pd.notna(row.get("open_time_ms")):
            marker["open_time_ms"] = int(row.get("open_time_ms"))
        markers.append(marker)
    return markers


def _trade_detail_script(
    markers: list[dict[str, Any]],
    *,
    frame: pd.DataFrame,
    meta: dict[str, object],
    config: ChartSeriesConfig,
    kline_details: dict[int, dict[str, Any]] | None = None,
) -> str:
    if not markers and not kline_details:
        return ""
    payload = json.dumps([_marker_detail_payload(marker) for marker in markers], ensure_ascii=False)
    kline_payload = json.dumps({str(key): value for key, value in (kline_details or {}).items()}, ensure_ascii=False)
    rows_payload = json.dumps(_chart_rows_payload(frame, config), ensure_ascii=False)
    meta_payload = json.dumps(
        {
            "code": meta.get("code"),
            "name": meta.get("name") or "",
            "start_date": meta.get("start_date"),
            "trade_date": meta.get("trade_date"),
            "golden_bull": config.golden_bull,
            "volume_ma": config.volume_ma,
            "ma_periods": list(config.ma_periods),
        },
        ensure_ascii=False,
    )
    return f"""<script>
  const tradeMarkers = {payload};
  const klineDetails = {kline_payload};
  const chartRows = {rows_payload};
  const chartMeta = {meta_payload};
  const tradeMarkerById = new Map(tradeMarkers.map((item) => [String(item.id), item]));
  const detailList = document.getElementById("trade-detail-list");
  const chartContainer = document.getElementById("chart-svg-container");
  const windowSizeInput = document.getElementById("chart-window-size");
  const windowStartInput = document.getElementById("chart-window-start");
  const windowLabel = document.getElementById("chart-window-label");
  const tradePrevButton = document.getElementById("trade-prev");
  const tradeNextButton = document.getElementById("trade-next");
  const svgLayout = {{ width: 1200, height: 720, left: 66, right: 72, top: 42, priceHeight: 454, gap: 24, volumeHeight: 152 }};
  const tradeMarkerHitRadius = {TRADE_MARKER_HIT_RADIUS};
  const reburnMarkerHitRadius = {REBURN_MARKER_HIT_RADIUS};
  const tradeMarkerYOffset = {TRADE_MARKER_Y_OFFSET};
  const reburnMarkerYOffset = {REBURN_MARKER_Y_OFFSET};
  const tradeMarkerStackGap = {TRADE_MARKER_STACK_GAP};
  const reburnMarkerStackGap = {REBURN_MARKER_STACK_GAP};
  svgLayout.volumeTop = svgLayout.top + svgLayout.priceHeight + svgLayout.gap;
  svgLayout.plotWidth = svgLayout.width - svgLayout.left - svgLayout.right;
  let selectedMarkerId = null;
  let selectedKlineOpenTimeMs = null;
  function formatTradeValue(value) {{
    if (Array.isArray(value) || (value && typeof value === "object")) return JSON.stringify(value, null, 2);
    return value;
  }}
  function esc(value) {{
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[char]));
  }}
  function finiteNumber(value) {{
    const number = Number(value);
    return Number.isFinite(number);
  }}
  function num(value) {{
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }}
  function cssIdent(value) {{
    if (window.CSS && typeof window.CSS.escape === "function") return CSS.escape(String(value));
    return String(value).replace(/["\\\\]/g, "\\\\$&");
  }}
  function extent(rows, columns) {{
    const values = [];
    rows.forEach((row) => {{
      columns.forEach((column) => {{
        const value = num(row[column]);
        if (value !== null) values.push(value);
      }});
    }});
    if (!values.length) return [0, 1];
    return [Math.min(...values), Math.max(...values)];
  }}
  function grid(left, top, width, height, minValue, maxValue, scale, rightAxisX, rows = 5) {{
    const parts = [];
    for (let index = 0; index <= rows; index += 1) {{
      const value = minValue + (maxValue - minValue) * index / rows;
      const y = scale(value);
      parts.push(`<line x1="${{left}}" y1="${{y.toFixed(2)}}" x2="${{left + width}}" y2="${{y.toFixed(2)}}" stroke="#260000" stroke-width="1"/>`);
      parts.push(`<text x="${{rightAxisX + 8}}" y="${{(y + 4).toFixed(2)}}" fill="#ff3030" font-size="12">${{value.toFixed(2)}}</text>`);
    }}
    parts.push(`<rect x="${{left}}" y="${{top}}" width="${{width}}" height="${{height}}" fill="none" stroke="#a00000" stroke-width="1"/>`);
    return parts.join("\\n");
  }}
  function linePath(rows, column, xAt, yAt, color, width = 1.5, dash = null) {{
    const segments = [];
    let current = [];
    rows.forEach((row, idx) => {{
      const value = num(row[column]);
      if (value === null) {{
        if (current.length) segments.push(current);
        current = [];
        return;
      }}
      current.push(`${{current.length ? "L" : "M"}}${{xAt(idx).toFixed(2)}},${{yAt(value).toFixed(2)}}`);
    }});
    if (current.length) segments.push(current);
    const dashAttr = dash ? ` stroke-dasharray="${{dash}}"` : "";
    return segments.filter((segment) => segment.length >= 2)
      .map((segment) => `<path d="${{segment.join(" ")}}" fill="none" stroke="${{color}}" stroke-width="${{width}}"${{dashAttr}}/>`)
      .join("\\n");
  }}
  function channelRegime(lifeLine, trendConfirmationLine) {{
    if (trendConfirmationLine < lifeLine) return "bull";
    if (lifeLine < trendConfirmationLine) return "bear";
    return "transition";
  }}
  function channelFill(rows, xAt, yAt) {{
    const required = ["channel_upper", "channel_lower", "golden_bull_trend", "golden_bull_2"];
    if (!required.every((column) => rows.some((row) => row[column] !== null && row[column] !== undefined))) return "";
    const fills = [];
    let upper = [];
    let lower = [];
    let currentRegime = null;
    function flush() {{
      if (currentRegime && upper.length >= 2) {{
        const colors = {{ bull: "#4d0000", bear: "#004d22", transition: "#303030" }};
        fills.push(`<polygon data-regime="${{currentRegime}}" points="${{upper.concat([...lower].reverse()).join(" ")}}" fill="${{colors[currentRegime]}}" opacity="0.42"/>`);
      }}
      upper = [];
      lower = [];
    }}
    rows.forEach((row, idx) => {{
      const values = required.map((column) => num(row[column]));
      if (values.some((value) => value === null)) {{
        flush();
        currentRegime = null;
        return;
      }}
      const regime = channelRegime(num(row.golden_bull_trend), num(row.golden_bull_2));
      if (regime !== currentRegime) {{
        flush();
        currentRegime = regime;
      }}
      upper.push(`${{xAt(idx).toFixed(2)}},${{yAt(num(row.channel_upper)).toFixed(2)}}`);
      lower.push(`${{xAt(idx).toFixed(2)}},${{yAt(num(row.channel_lower)).toFixed(2)}}`);
    }});
    flush();
    return fills.join("\\n");
  }}
  function dateLabels(rows, xAt, y) {{
    if (!rows.length) return "";
    const indexes = [...new Set([0, Math.floor(rows.length / 4), Math.floor(rows.length / 2), Math.floor(rows.length * 3 / 4), rows.length - 1])].sort((a, b) => a - b);
    return indexes.map((idx) => `<text x="${{xAt(idx).toFixed(2)}}" y="${{y.toFixed(2)}}" fill="#888" font-size="11" text-anchor="middle">${{esc(rows[idx].trade_date)}}</text>`).join("\\n");
  }}
  function markerTitle(marker) {{
    return [
      ["marker_kind", marker.marker_kind],
      ["label", marker.label],
      ["marker_name", marker.marker_name],
      ["time", marker.time],
      ["side", marker.side],
      ["action", marker.action],
      ["signal_action", marker.signal_action],
      ["channel_regime", marker.channel_regime],
      ["price", marker.price],
      ["qty", marker.qty],
      ["quote_value", marker.quote_value],
      ["fee", marker.fee],
    ].filter(([, value]) => value !== null && value !== undefined).map(([key, value]) => `${{key}}=${{value}}`).join("; ");
  }}
  function renderTradeMarkers(rows, visibleMarkers, xAt, yAt) {{
    const indexByOpenTime = new Map();
    const indexByTradeDate = new Map();
    rows.forEach((row, idx) => {{
      if (row.open_time_ms !== null && row.open_time_ms !== undefined) indexByOpenTime.set(Number(row.open_time_ms), idx);
      indexByTradeDate.set(String(row.trade_date), idx);
    }});
    const stacked = new Map();
    return visibleMarkers.map((marker) => {{
      let idx = null;
      if (marker.open_time_ms !== null && marker.open_time_ms !== undefined) idx = indexByOpenTime.get(Number(marker.open_time_ms));
      if ((idx === null || idx === undefined) && marker.time) idx = indexByTradeDate.get(String(marker.time));
      if (idx === null || idx === undefined || !finiteNumber(marker.price)) return "";
      const kind = String(marker.marker_kind || "trade").toLowerCase();
      const side = String(marker.side || "").toLowerCase();
      const isBuy = side === "buy";
      const isReburn = kind === "reburn";
      const color = isReburn ? "#ff9f1a" : (isBuy ? "#ff3030" : "#00c853");
      const x = xAt(idx);
      const y = yAt(Number(marker.price));
      const stackKey = `${{idx}}:${{kind}}:${{side}}`;
      const offsetCount = stacked.get(stackKey) || 0;
      stacked.set(stackKey, offsetCount + 1);
      const yOffset = isReburn
        ? reburnMarkerYOffset + offsetCount * reburnMarkerStackGap
        : (isBuy ? -tradeMarkerYOffset - offsetCount * tradeMarkerStackGap : tradeMarkerYOffset + offsetCount * tradeMarkerStackGap);
      const markerY = y + yOffset;
      const label = marker.label || (isBuy ? "B" : "S");
      const fontSize = isReburn ? 11 : 13;
      const weight = isReburn ? 700 : 800;
      const anchor = "middle";
      const hitRadius = isReburn ? reburnMarkerHitRadius : tradeMarkerHitRadius;
      return `<g class="trade-marker trade-${{esc(side)}}" data-marker-id="${{esc(marker.id)}}" tabindex="0"><title>${{esc(markerTitle(marker))}}</title><circle cx="${{x.toFixed(2)}}" cy="${{markerY.toFixed(2)}}" r="${{hitRadius}}" fill="transparent" pointer-events="all"/><text x="${{x.toFixed(2)}}" y="${{markerY.toFixed(2)}}" fill="${{color}}" font-size="${{fontSize}}" font-weight="${{weight}}" text-anchor="${{anchor}}" dominant-baseline="middle">${{esc(label)}}</text></g>`;
    }}).join("\\n");
  }}
  function renderChartWindow() {{
    if (!chartContainer || !chartRows.length) return;
    const size = Math.max(1, Math.min(Number(windowSizeInput?.value || chartRows.length), chartRows.length));
    const maxStart = Math.max(chartRows.length - size, 0);
    if (windowStartInput) {{
      windowStartInput.max = String(maxStart);
      if (Number(windowStartInput.value) > maxStart) windowStartInput.value = String(maxStart);
    }}
    const start = Math.max(0, Math.min(Number(windowStartInput?.value || maxStart), maxStart));
    const rows = chartRows.slice(start, start + size);
    const openTimes = new Set(rows.map((row) => Number(row.open_time_ms)).filter((value) => Number.isFinite(value)));
    const dates = new Set(rows.map((row) => String(row.trade_date)));
    const visibleMarkers = tradeMarkers.filter((marker) => (
      marker.open_time_ms !== null && marker.open_time_ms !== undefined && openTimes.has(Number(marker.open_time_ms))
    ) || dates.has(String(marker.time)));
    const step = svgLayout.plotWidth / Math.max(rows.length, 1);
    const candleWidth = Math.max(3.0, Math.min(12.0, step * 0.58));
    const priceColumns = ["open", "high", "low", "close"];
    (chartMeta.ma_periods || []).forEach((period) => priceColumns.push(`ma${{period}}`));
    if (chartMeta.golden_bull) priceColumns.push("golden_bull", "golden_bull_trend", "golden_bull_2", "channel_upper", "channel_lower");
    let [priceMin, priceMax] = extent(rows, priceColumns);
    const markerPrices = visibleMarkers.map((marker) => num(marker.price)).filter((value) => value !== null);
    if (markerPrices.length) {{
      priceMin = Math.min(priceMin, ...markerPrices);
      priceMax = Math.max(priceMax, ...markerPrices);
    }}
    const pad = Math.max((priceMax - priceMin) * 0.08, priceMax * 0.01, 0.01);
    priceMin -= pad;
    priceMax += pad;
    let volMax = Math.max(...rows.map((row) => num(row.vol) || 0), 1.0);
    if (chartMeta.volume_ma) volMax = Math.max(volMax, ...rows.map((row) => num(row.vol_ma5) || 0), 1.0);
    const xAt = (idx) => svgLayout.left + step * idx + step / 2;
    const yPrice = (value) => svgLayout.top + (priceMax - value) / (priceMax - priceMin) * svgLayout.priceHeight;
    const yVol = (value) => svgLayout.volumeTop + (volMax - value) / volMax * svgLayout.volumeHeight;
    const parts = [
      `<svg viewBox="0 0 ${{svgLayout.width}} ${{svgLayout.height}}" role="img" aria-label="${{esc(chartMeta.code)}} kline chart">`,
      '<rect x="0" y="0" width="1200" height="720" fill="#050505"/>',
      grid(svgLayout.left, svgLayout.top, svgLayout.plotWidth, svgLayout.priceHeight, priceMin, priceMax, yPrice, svgLayout.width - svgLayout.right),
      grid(svgLayout.left, svgLayout.volumeTop, svgLayout.plotWidth, svgLayout.volumeHeight, 0, volMax, yVol, svgLayout.width - svgLayout.right, 3),
      `<text x="${{svgLayout.left}}" y="24" fill="#e8e8e8" font-size="18" font-weight="700">${{esc(chartMeta.code)}} ${{esc(chartMeta.name || "")}}</text>`,
      `<text x="${{svgLayout.width - svgLayout.right}}" y="24" fill="#c4c4c4" font-size="13" text-anchor="end">${{esc(rows[0]?.trade_date)}} - ${{esc(rows[rows.length - 1]?.trade_date)}}</text>`,
    ];
    if (chartMeta.golden_bull) {{
      parts.push(channelFill(rows, xAt, yPrice));
      parts.push(linePath(rows, "golden_bull", xAt, yPrice, "#fff000", 2, "5 7"));
      parts.push(linePath(rows, "golden_bull_trend", xAt, yPrice, "#ff3030", 2, "7 6"));
      parts.push(linePath(rows, "golden_bull_2", xAt, yPrice, "#00d7d7", 2, "9 7"));
    }}
    const maStyles = {{ 5: "#ffffff", 10: "#ffd84d", 20: "#ff5d5d", 60: "#37d7ff" }};
    (chartMeta.ma_periods || []).forEach((period) => parts.push(linePath(rows, `ma${{period}}`, xAt, yPrice, maStyles[period] || "#d8d8d8", 1.6)));
    rows.forEach((row, idx) => {{
      const x = xAt(idx);
      const open = num(row.open);
      const high = num(row.high);
      const low = num(row.low);
      const close = num(row.close);
      if ([open, high, low, close].some((value) => value === null)) return;
      const up = close >= open;
      const color = up ? "#ff3030" : "#42edf0";
      const yOpen = yPrice(open);
      const yClose = yPrice(close);
      const bodyTop = Math.min(yOpen, yClose);
      const bodyHeight = Math.max(Math.abs(yClose - yOpen), 1.2);
      const fill = up ? "#050505" : color;
      const candleParts = [];
      candleParts.push(`<line x1="${{x.toFixed(2)}}" y1="${{yPrice(high).toFixed(2)}}" x2="${{x.toFixed(2)}}" y2="${{yPrice(low).toFixed(2)}}" stroke="${{color}}" stroke-width="1.4"/>`);
      candleParts.push(`<rect x="${{(x - candleWidth / 2).toFixed(2)}}" y="${{bodyTop.toFixed(2)}}" width="${{candleWidth.toFixed(2)}}" height="${{bodyHeight.toFixed(2)}}" fill="${{fill}}" stroke="${{color}}" stroke-width="1.5"/>`);
      parts.push(`<g class="kline-candle" data-open-time-ms="${{esc(row.open_time_ms)}}" tabindex="0"><title>${{esc(row.trade_date)}} O=${{open}} H=${{high}} L=${{low}} C=${{close}}</title>${{candleParts.join("")}}</g>`);
      const vol = num(row.vol) || 0;
      const volY = yVol(vol);
      parts.push(`<rect x="${{(x - candleWidth / 2).toFixed(2)}}" y="${{volY.toFixed(2)}}" width="${{candleWidth.toFixed(2)}}" height="${{(svgLayout.volumeTop + svgLayout.volumeHeight - volY).toFixed(2)}}" fill="${{fill}}" stroke="${{color}}" stroke-width="1"/>`);
    }});
    if (chartMeta.volume_ma) parts.push(linePath(rows, "vol_ma5", xAt, yVol, "#f5f5f5", 1.4));
    parts.push(renderTradeMarkers(rows, visibleMarkers, xAt, yPrice));
    parts.push(dateLabels(rows, xAt, svgLayout.volumeTop + svgLayout.volumeHeight + 20));
    parts.push("</svg>");
    chartContainer.innerHTML = parts.filter(Boolean).join("\\n");
    if (windowLabel) windowLabel.textContent = `${{start + 1}}-${{start + rows.length}} / ${{chartRows.length}}`;
    bindTradeMarkerEvents();
    bindKlineEvents(rows);
    applySelectionClasses();
    updateTradeNav();
  }}
  function chartRowForMarker(marker) {{
    if (!marker) return null;
    if (marker.open_time_ms !== null && marker.open_time_ms !== undefined) {{
      const openTime = Number(marker.open_time_ms);
      const row = chartRows.find((item) => Number(item.open_time_ms) === openTime);
      if (row) return row;
    }}
    if (marker.time) return chartRows.find((item) => String(item.trade_date) === String(marker.time)) || null;
    return null;
  }}
  function renderFields(fields) {{
    detailList.innerHTML = fields
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .map(([name, value]) => `<dt>${{esc(name)}}</dt><dd>${{esc(formatTradeValue(value))}}</dd>`)
      .join("");
  }}
  function klineDetailFields(row) {{
    const context = klineDetails[String(row.open_time_ms)] || {{}};
    return [
      ["K\u7ebf\u65f6\u95f4", row.trade_date],
      ["\u5f53\u524d\u6536\u76ca\u7387", context.return_pct],
      ["open_time_ms", row.open_time_ms],
      ["\u5f00", row.open],
      ["\u9ad8", row.high],
      ["\u4f4e", row.low],
      ["\u6536", row.close],
      ["\u6210\u4ea4\u91cf", row.vol],
      ["MA20", context.decision_ma20 ?? row.ma20],
      ["MA60", context.decision_ma60 ?? row.ma60],
      ["MA60\u659c\u7387", context.ma60_slope_pct],
      ["MA20/MA60\u5f20\u53e3", context.ma20_ma60_spread_pct],
      ["\u5f53\u65f6\u51b3\u7b56\u91d1\u725b", context.decision_golden_bull ?? row.golden_bull],
      ["\u5f53\u65f6\u51b3\u7b56\u91d1\u725b\u8d8b\u52bf", context.decision_golden_bull_trend ?? row.golden_bull_trend],
      ["\u5f53\u65f6\u51b3\u7b56\u91d1\u725b2", context.decision_golden_bull_2 ?? row.golden_bull_2],
      ["\u6700\u7ec8\u4ea4\u6613\u4fe1\u53f7", context.final_trade_signal],
      ["\u6700\u7ec8\u4ea4\u6613\u52a8\u4f5c", context.final_trade_action || context.action],
      ["\u76ee\u6807\u4ed3\u4f4d", context.target_position_pct],
      ["\u5f53\u524d\u4ed3\u4f4d", context.current_position_pct],
      ["\u5f53\u524d\u603b\u5e02\u503c", context.equity],
      ["\u73b0\u91d1", context.cash],
      ["BTC", context.base_qty],
      ["\u6301\u4ed3\u5e02\u503c", context.position_value],
      ["\u6b62\u76c8\u51cf\u4ed3\u72b6\u6001", context.risk],
      ["\u5165\u573a\u5747\u4ef7", context.entry_price],
      ["\u5165\u573a\u540e\u6700\u9ad8\u4ef7", context.entry_high_price],
      ["\u672c\u8f6e\u6700\u9ad8\u6d6e\u76c8", context.entry_peak_gain_pct],
      ["\u4e70\u5165K\u7ebfLow", context.entry_candle_low_price],
      ["\u901a\u9053\u6027\u8d28", context.channel_regime],
      ["\u901a\u9053\u5f3a\u5ea6", context.channel_strength],
      ["\u89e6\u53d1\u573a\u666f", context.scenes],
      ["\u4ea4\u6613\u5019\u9009\u4fe1\u53f7", context.trade_candidates],
      ["\u91d1\u725b\u5168\u90e8\u8bc4\u7ea7", context.golden_bull_ratings],
      ["\u91d1\u725b\u5ea6\u91cf", context.golden_bull_metrics],
      ["\u51b3\u7b56\u539f\u56e0", context.reason],
      ["\u6700\u7ec8\u4fe1\u53f7\u539f\u56e0", context.final_trade_reason],
    ];
  }}
  function renderTradeDetail(marker) {{
    if (!detailList || !marker) return;
    const markerFields = [
      ["\u65f6\u95f4", marker.time],
      ["\u6807\u8bb0", marker.label],
      ["\u6807\u8bb0\u540d\u79f0", marker.marker_name],
      ["\u6807\u8bb0\u7c7b\u578b", marker.marker_kind],
      ["\u5f53\u524d\u6536\u76ca\u7387", marker.return_pct_after ?? marker.return_pct_before],
      ["\u65b9\u5411", marker.side],
      ["\u52a8\u4f5c", marker.action],
      ["\u91d1\u725b\u4fe1\u53f7", marker.signal_action],
      ["\u6700\u7ec8\u4ea4\u6613\u4fe1\u53f7", marker.final_trade_signal],
      ["\u6700\u7ec8\u4ea4\u6613\u52a8\u4f5c", marker.final_trade_action],
      ["\u4fe1\u53f7\u5f3a\u5ea6", marker.signal_strength],
      ["\u901a\u9053\u6027\u8d28", marker.channel_regime],
      ["\u901a\u9053\u5f3a\u5ea6", marker.channel_strength],
      ["\u89e6\u53d1\u573a\u666f", marker.scenes],
      ["\u4ea4\u6613\u5019\u9009\u4fe1\u53f7", marker.trade_candidates],
      ["\u91d1\u725b\u5168\u90e8\u8bc4\u7ea7", marker.golden_bull_ratings],
      ["\u91d1\u725b\u5ea6\u91cf", marker.golden_bull_metrics],
      ["\u98ce\u63a7\u6807\u8bb0", marker.risk_flags],
      ["\u5931\u6548\u6761\u4ef6", marker.invalidations],
      ["\u89e6\u53d1\u539f\u56e0", marker.reason],
      ["\u6700\u7ec8\u4fe1\u53f7\u539f\u56e0", marker.final_trade_reason],
      ["\u4ef7\u683c", marker.price],
      ["\u6570\u91cf", marker.qty],
      ["\u91d1\u989d", marker.quote_value],
      ["\u624b\u7eed\u8d39", marker.fee],
      ["\u76ee\u6807\u4ed3\u4f4d", marker.target_position_pct],
      ["\u6210\u4ea4\u524d\u6536\u76ca\u7387", marker.return_pct_before],
      ["\u6210\u4ea4\u524d\u603b\u5e02\u503c", marker.equity_before],
      ["\u6210\u4ea4\u524d\u73b0\u91d1", marker.cash_before],
      ["\u6210\u4ea4\u524dBTC", marker.base_qty_before],
      ["\u6210\u4ea4\u524d\u4ed3\u4f4d", marker.position_pct_before],
      ["\u6210\u4ea4\u540e\u6536\u76ca\u7387", marker.return_pct_after],
      ["\u6210\u4ea4\u540e\u603b\u5e02\u503c", marker.equity_after],
      ["\u6210\u4ea4\u540e\u73b0\u91d1", marker.cash_after],
      ["\u6210\u4ea4\u540eBTC", marker.base_qty_after],
      ["\u6210\u4ea4\u540e\u4ed3\u4f4d", marker.position_pct_after],
      ["\u6b62\u635f\u7ebf", marker.stop_line_name],
      ["\u6b62\u635f\u4ef7", marker.stop_line_price],
      ["\u5165\u573a\u7c7b\u578b", marker.entry_signal_type],
      ["\u5165\u573a\u5747\u4ef7", marker.entry_price],
      ["\u5165\u573a\u540e\u6700\u9ad8\u4ef7", marker.entry_high_price],
      ["\u672c\u8f6e\u6700\u9ad8\u6d6e\u76c8", marker.entry_peak_gain_pct],
      ["\u4e70\u5165K\u7ebfLow", marker.entry_candle_low_price],
    ];
    const row = chartRowForMarker(marker);
    renderFields(row ? markerFields.concat(klineDetailFields(row)) : markerFields);
  }}
  function renderKlineDetail(row) {{
    if (!detailList || !row) return;
    renderFields(klineDetailFields(row));
  }}
  function bindTradeMarkerEvents() {{
    document.querySelectorAll(".trade-marker").forEach((node) => {{
      node.addEventListener("click", () => {{
        selectTradeMarkerById(node.dataset.markerId, {{ ensureVisible: false }});
      }});
      node.addEventListener("keydown", (event) => {{
        if (event.key === "Enter" || event.key === " ") {{
          event.preventDefault();
          node.dispatchEvent(new MouseEvent("click", {{ bubbles: true }}));
        }}
      }});
    }});
  }}
  function bindKlineEvents(rows) {{
    const rowByOpenTime = new Map(rows.map((row) => [String(row.open_time_ms), row]));
    document.querySelectorAll(".kline-candle").forEach((node) => {{
      node.addEventListener("click", () => {{
        selectedMarkerId = null;
        selectedKlineOpenTimeMs = String(node.dataset.openTimeMs);
        applySelectionClasses();
        updateTradeNav();
        renderKlineDetail(rowByOpenTime.get(String(node.dataset.openTimeMs)));
      }});
      node.addEventListener("keydown", (event) => {{
        if (event.key === "Enter" || event.key === " ") {{
          event.preventDefault();
          node.dispatchEvent(new MouseEvent("click", {{ bubbles: true }}));
        }}
      }});
    }});
  }}
  function markerChartIndex(marker) {{
    if (!marker) return -1;
    if (marker.open_time_ms !== null && marker.open_time_ms !== undefined) {{
      const openTime = Number(marker.open_time_ms);
      const index = chartRows.findIndex((row) => Number(row.open_time_ms) === openTime);
      if (index >= 0) return index;
    }}
    if (marker.time) return chartRows.findIndex((row) => String(row.trade_date) === String(marker.time));
    return -1;
  }}
  function ensureMarkerVisible(marker) {{
    const index = markerChartIndex(marker);
    if (index < 0 || !windowStartInput || !windowSizeInput) return false;
    const size = Math.max(1, Math.min(Number(windowSizeInput.value || chartRows.length), chartRows.length));
    const currentStart = Math.max(0, Number(windowStartInput.value || 0));
    if (index >= currentStart && index < currentStart + size) return false;
    const maxStart = Math.max(chartRows.length - size, 0);
    const nextStart = Math.max(0, Math.min(index - Math.floor(size / 2), maxStart));
    windowStartInput.value = String(nextStart);
    return true;
  }}
  function selectTradeMarkerById(markerId, options = {{}}) {{
    const marker = tradeMarkerById.get(String(markerId));
    if (!marker) return;
    selectedMarkerId = String(marker.id);
    selectedKlineOpenTimeMs = null;
    renderTradeDetail(marker);
    updateTradeNav();
    if (options.ensureVisible && ensureMarkerVisible(marker)) {{
      renderChartWindow();
      return;
    }}
    applySelectionClasses();
    if (options.focus) {{
      document.querySelector(`.trade-marker[data-marker-id="${{cssIdent(selectedMarkerId)}}"]`)?.focus();
    }}
  }}
  function applySelectionClasses() {{
    document.querySelectorAll(".trade-marker.is-selected").forEach((item) => item.classList.remove("is-selected"));
    document.querySelectorAll(".kline-candle.is-selected").forEach((item) => item.classList.remove("is-selected"));
    if (selectedMarkerId) {{
      document.querySelector(`.trade-marker[data-marker-id="${{cssIdent(selectedMarkerId)}}"]`)?.classList.add("is-selected");
    }}
    if (selectedKlineOpenTimeMs) {{
      document.querySelector(`.kline-candle[data-open-time-ms="${{cssIdent(selectedKlineOpenTimeMs)}}"]`)?.classList.add("is-selected");
    }}
  }}
  function selectedTradeIndex() {{
    if (!tradeMarkers.length) return -1;
    if (!selectedMarkerId) return -1;
    return tradeMarkers.findIndex((marker) => String(marker.id) === String(selectedMarkerId));
  }}
  function updateTradeNav() {{
    const index = selectedTradeIndex();
    if (tradePrevButton) tradePrevButton.disabled = !tradeMarkers.length || index === 0;
    if (tradeNextButton) tradeNextButton.disabled = !tradeMarkers.length || index === tradeMarkers.length - 1;
  }}
  function moveTradeSelection(delta) {{
    if (!tradeMarkers.length) return;
    const index = selectedTradeIndex();
    const baseIndex = index >= 0 ? index : (delta > 0 ? -1 : tradeMarkers.length);
    const nextIndex = Math.max(0, Math.min(baseIndex + delta, tradeMarkers.length - 1));
    selectTradeMarkerById(tradeMarkers[nextIndex].id, {{ ensureVisible: true, focus: true }});
  }}
  windowSizeInput?.addEventListener("input", () => {{
    renderChartWindow();
  }});
  windowStartInput?.addEventListener("input", () => {{
    renderChartWindow();
  }});
  tradePrevButton?.addEventListener("click", () => moveTradeSelection(-1));
  tradeNextButton?.addEventListener("click", () => moveTradeSelection(1));
  document.addEventListener("keydown", (event) => {{
    const tag = String(event.target?.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (event.key === "ArrowLeft") {{
      event.preventDefault();
      moveTradeSelection(-1);
    }}
    if (event.key === "ArrowRight") {{
      event.preventDefault();
      moveTradeSelection(1);
    }}
  }});
  renderChartWindow();
</script>"""


def _marker_detail_payload(marker: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "marker_kind",
        "label",
        "marker_name",
        "time",
        "open_time_ms",
        "side",
        "action",
        "price",
        "qty",
        "quote_value",
        "fee",
        "target_position_pct",
        "return_pct_before",
        "equity_before",
        "cash_before",
        "base_qty_before",
        "position_pct_before",
        "return_pct_after",
        "equity_after",
        "cash_after",
        "base_qty_after",
        "position_pct_after",
        "stop_line_name",
        "stop_line_price",
        "entry_signal_type",
        "entry_price",
        "entry_price_before",
        "entry_high_price",
        "entry_high_price_before",
        "entry_peak_gain_pct",
        "entry_candle_low_price",
        "entry_candle_low_price_before",
        "signal_action",
        "signal_strength",
        "final_trade_signal",
        "final_trade_action",
        "final_trade_side",
        "final_trade_reason",
        "trade_candidates",
        "golden_bull_ratings",
        "golden_bull_metrics",
        "channel_regime",
        "channel_strength",
        "scenes",
        "risk_flags",
        "invalidations",
        "reason",
    ]
    return {key: marker.get(key) for key in keys}


def _chart_rows_payload(frame: pd.DataFrame, config: ChartSeriesConfig) -> list[dict[str, Any]]:
    columns = ["trade_date", "open_time_ms", "open", "high", "low", "close", "vol"]
    columns.extend(f"ma{period}" for period in config.ma_periods)
    if config.volume_ma:
        columns.append("vol_ma5")
    if config.golden_bull:
        columns.extend(["golden_bull", "golden_bull_trend", "golden_bull_2", "channel_upper", "channel_lower"])
    existing = [column for column in columns if column in frame]
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        item: dict[str, Any] = {}
        for column in existing:
            value = row[column]
            if pd.isna(value):
                item[column] = None
            elif column == "trade_date":
                item[column] = str(value)
            elif column == "open_time_ms":
                item[column] = int(value)
            else:
                item[column] = float(value)
        rows.append(item)
    return rows


def _render_svg(
    frame: pd.DataFrame,
    meta: dict[str, object],
    config: ChartSeriesConfig,
    *,
    markers: list[dict[str, Any]] | None = None,
) -> str:
    markers = markers or []
    width = 1200
    height = 720
    left = 66
    right = 72
    top = 42
    price_height = 454
    gap = 24
    volume_top = top + price_height + gap
    volume_height = 152
    plot_width = width - left - right
    count = len(frame)
    step = plot_width / max(count, 1)
    candle_width = max(3.0, min(12.0, step * 0.58))

    price_columns = ["open", "high", "low", "close"]
    if config.ma_periods:
        price_columns.extend([f"ma{period}" for period in config.ma_periods])
    if config.golden_bull:
        price_columns.extend(["golden_bull", "golden_bull_trend", "golden_bull_2", "channel_upper", "channel_lower"])
    price_min, price_max = _series_extent(frame, price_columns)
    marker_prices = [float(item["price"]) for item in markers if _is_finite_number(item.get("price"))]
    if marker_prices:
        price_min = min(price_min, min(marker_prices))
        price_max = max(price_max, max(marker_prices))
    pad = max((price_max - price_min) * 0.08, price_max * 0.01, 0.01)
    price_min -= pad
    price_max += pad
    vol_max = max(float(frame["vol"].max() or 0), 1.0)
    if config.volume_ma and "vol_ma5" in frame:
        vol_max = max(vol_max, float(frame["vol_ma5"].max() or 0), 1.0)

    def x_at(idx: int) -> float:
        return left + step * idx + step / 2

    def y_price(value: float) -> float:
        return top + (price_max - value) / (price_max - price_min) * price_height

    def y_vol(value: float) -> float:
        return volume_top + (vol_max - value) / vol_max * volume_height

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(str(meta.get("code")))} daily kline chart">',
        '<rect x="0" y="0" width="1200" height="720" fill="#050505"/>',
        _grid(left, top, plot_width, price_height, price_min, price_max, y_price, right_axis_x=width - right),
        _grid(left, volume_top, plot_width, volume_height, 0, vol_max, y_vol, right_axis_x=width - right, rows=3),
        f'<text x="{left}" y="24" fill="#e8e8e8" font-size="18" font-weight="700">{escape(str(meta.get("code")))} {escape(str(meta.get("name") or ""))}</text>',
        f'<text x="{width - right}" y="24" fill="#c4c4c4" font-size="13" text-anchor="end">{escape(str(meta.get("start_date")))} - {escape(str(meta.get("trade_date")))}</text>',
    ]

    if config.golden_bull:
        parts.append(_channel_fill(frame, x_at, y_price))
        parts.extend(
            [
                _line_path(frame, "golden_bull", x_at, y_price, "#fff000", dash="5 7", width=2),
                _line_path(frame, "golden_bull_trend", x_at, y_price, "#ff3030", dash="7 6", width=2),
                _line_path(frame, "golden_bull_2", x_at, y_price, "#00d7d7", dash="9 7", width=2),
            ]
        )

    ma_styles = {5: "#ffffff", 10: "#ffd84d", 20: "#ff5d5d", 60: "#37d7ff"}
    for period in config.ma_periods:
        parts.append(_line_path(frame, f"ma{period}", x_at, y_price, ma_styles.get(period, "#d8d8d8"), width=1.6))

    for idx, row in frame.iterrows():
        x = x_at(idx)
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        up = close >= open_
        color = "#ff3030" if up else "#42edf0"
        y_open = y_price(open_)
        y_close = y_price(close)
        body_top = min(y_open, y_close)
        body_height = max(abs(y_close - y_open), 1.2)
        fill = "#050505" if up else color
        parts.append(
            f'<line x1="{x:.2f}" y1="{y_price(high):.2f}" x2="{x:.2f}" y2="{y_price(low):.2f}" stroke="{color}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<rect x="{x - candle_width / 2:.2f}" y="{body_top:.2f}" width="{candle_width:.2f}" height="{body_height:.2f}" fill="{fill}" stroke="{color}" stroke-width="1.5"/>'
        )
        vol = float(row["vol"]) if pd.notna(row["vol"]) else 0.0
        vol_y = y_vol(vol)
        parts.append(
            f'<rect x="{x - candle_width / 2:.2f}" y="{vol_y:.2f}" width="{candle_width:.2f}" height="{volume_top + volume_height - vol_y:.2f}" fill="{fill}" stroke="{color}" stroke-width="1"/>'
        )

    if config.volume_ma and "vol_ma5" in frame:
        parts.append(_line_path(frame, "vol_ma5", x_at, y_vol, "#f5f5f5", width=1.4))

    if markers:
        parts.append(_trade_markers(frame, markers, x_at, y_price))

    parts.append(_date_labels(frame, x_at, volume_top + volume_height + 20))
    parts.append("</svg>")
    return "\n".join(part for part in parts if part)


def _grid(
    left: int,
    top: int,
    width: int,
    height: int,
    min_value: float,
    max_value: float,
    scale,
    *,
    right_axis_x: int,
    rows: int = 5,
) -> str:
    parts: list[str] = []
    for index in range(rows + 1):
        value = min_value + (max_value - min_value) * index / rows
        y = scale(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" stroke="#260000" stroke-width="1"/>')
        parts.append(
            f'<text x="{right_axis_x + 8}" y="{y + 4:.2f}" fill="#ff3030" font-size="12">{value:.2f}</text>'
        )
    parts.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#a00000" stroke-width="1"/>')
    return "\n".join(parts)


def _line_path(frame: pd.DataFrame, column: str, x_at, y_at, color: str, *, width: float = 1.5, dash: str | None = None) -> str:
    if column not in frame:
        return ""
    segments: list[list[str]] = []
    current: list[str] = []
    for idx, value in enumerate(frame[column]):
        if pd.isna(value):
            if current:
                segments.append(current)
                current = []
            continue
        command = "M" if not current else "L"
        current.append(f"{command}{x_at(idx):.2f},{y_at(float(value)):.2f}")
    if current:
        segments.append(current)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return "\n".join(
        f'<path d="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>'
        for segment in segments
        if len(segment) >= 2
    )


def _channel_fill(frame: pd.DataFrame, x_at, y_at) -> str:
    required = ["channel_upper", "channel_lower", "golden_bull_trend", "golden_bull_2"]
    if any(column not in frame for column in required):
        return ""

    fills: list[str] = []
    upper: list[str] = []
    lower: list[str] = []
    current_regime: str | None = None

    def flush_segment() -> None:
        nonlocal upper, lower
        if current_regime is None or len(upper) < 2:
            upper = []
            lower = []
            return
        colors = {
            "bull": "#4d0000",
            "bear": "#004d22",
            "transition": "#303030",
        }
        points = " ".join(upper + list(reversed(lower)))
        fills.append(
            f'<polygon data-regime="{current_regime}" points="{points}" fill="{colors[current_regime]}" opacity="0.42"/>'
        )
        upper = []
        lower = []

    for idx, row in frame.iterrows():
        values = [row[column] for column in required]
        if any(pd.isna(value) for value in values):
            flush_segment()
            current_regime = None
            continue

        regime = _channel_regime_for_chart(float(row["golden_bull_trend"]), float(row["golden_bull_2"]))
        if regime != current_regime:
            flush_segment()
            current_regime = regime

        upper.append(f"{x_at(idx):.2f},{y_at(float(row['channel_upper'])):.2f}")
        lower.append(f"{x_at(idx):.2f},{y_at(float(row['channel_lower'])):.2f}")

    flush_segment()
    return "\n".join(fills)


def _channel_regime_for_chart(life_line: float, trend_confirmation_line: float) -> str:
    if trend_confirmation_line < life_line:
        return "bull"
    if life_line < trend_confirmation_line:
        return "bear"
    return "transition"


def _date_labels(frame: pd.DataFrame, x_at, y: float) -> str:
    if frame.empty:
        return ""
    indexes = sorted(set([0, len(frame) // 4, len(frame) // 2, len(frame) * 3 // 4, len(frame) - 1]))
    parts = []
    for idx in indexes:
        label = str(frame.iloc[idx]["trade_date"])
        parts.append(f'<text x="{x_at(idx):.2f}" y="{y:.2f}" fill="#888" font-size="11" text-anchor="middle">{escape(label)}</text>')
    return "\n".join(parts)


def _legend_html(config: ChartSeriesConfig, *, has_markers: bool = False) -> str:
    items = [
        ("#ff3030", "上涨K线"),
        ("#42edf0", "下跌K线"),
    ]
    ma_styles = {5: "#ffffff", 10: "#ffd84d", 20: "#ff5d5d", 60: "#37d7ff"}
    for period in config.ma_periods:
        items.append((ma_styles.get(period, "#d8d8d8"), f"MA{period}"))
    if config.volume_ma:
        items.append(("#f5f5f5", "成交量MA5"))
    if config.golden_bull:
        items.extend([("#fff000", "金牛"), ("#ff3030", "金牛趋势"), ("#00d7d7", "金牛2")])
    if has_markers:
        items.extend([("#ff3030", "B"), ("#00c853", "S"), ("#ff9f1a", "火=低位复燃点")])
    return "".join(f'<span><i class="swatch" style="background:{color}"></i>{escape(label)}</span>' for color, label in items)


def _trade_markers(frame: pd.DataFrame, markers: list[dict[str, Any]], x_at, y_at) -> str:
    index_by_open_time = {}
    if "open_time_ms" in frame:
        for idx, value in enumerate(frame["open_time_ms"]):
            if pd.notna(value):
                index_by_open_time[int(value)] = idx
    index_by_trade_date = {str(value): idx for idx, value in enumerate(frame["trade_date"])}
    parts: list[str] = []
    stacked: dict[tuple[int, str], int] = {}
    for marker in markers:
        idx = None
        if marker.get("open_time_ms") is not None:
            idx = index_by_open_time.get(int(marker["open_time_ms"]))
        if idx is None and marker.get("time"):
            idx = index_by_trade_date.get(str(marker["time"]))
        if idx is None or not _is_finite_number(marker.get("price")):
            continue
        kind = str(marker.get("marker_kind") or "trade").lower()
        side = str(marker.get("side") or "").lower()
        is_buy = side == "buy"
        is_reburn = kind == "reburn"
        color = "#ff9f1a" if is_reburn else ("#ff3030" if is_buy else "#00c853")
        x = x_at(idx)
        y = y_at(float(marker["price"]))
        key = (idx, f"{kind}:{side}")
        offset_count = stacked.get(key, 0)
        stacked[key] = offset_count + 1
        y_offset = (
            REBURN_MARKER_Y_OFFSET + offset_count * REBURN_MARKER_STACK_GAP
            if is_reburn
            else (
                -TRADE_MARKER_Y_OFFSET - offset_count * TRADE_MARKER_STACK_GAP
                if is_buy
                else TRADE_MARKER_Y_OFFSET + offset_count * TRADE_MARKER_STACK_GAP
            )
        )
        marker_y = y + y_offset
        label = str(marker.get("label") or ("B" if is_buy else "S"))
        font_size = 11 if is_reburn else 13
        font_weight = 700 if is_reburn else 800
        title = _marker_title(marker)
        marker_id = escape(str(marker.get("id") or ""))
        parts.append(f'<g class="trade-marker trade-{escape(side)}" data-marker-id="{marker_id}" tabindex="0">')
        parts.append(f"<title>{escape(title)}</title>")
        hit_radius = REBURN_MARKER_HIT_RADIUS if is_reburn else TRADE_MARKER_HIT_RADIUS
        parts.append(f'<circle cx="{x:.2f}" cy="{marker_y:.2f}" r="{hit_radius}" fill="transparent" pointer-events="all"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{marker_y:.2f}" fill="{color}" font-size="{font_size}" '
            f'font-weight="{font_weight}" text-anchor="middle" dominant-baseline="middle">{escape(label)}</text>'
        )
        parts.append("</g>")
    return "\n".join(parts)


def _marker_title(marker: dict[str, Any]) -> str:
    fields = [
        f"marker_kind={marker.get('marker_kind')}",
        f"label={marker.get('label')}",
        f"marker_name={marker.get('marker_name')}",
        f"time={marker.get('time')}",
        f"side={marker.get('side')}",
        f"action={marker.get('action')}",
        f"signal_action={marker.get('signal_action')}",
        f"channel_regime={marker.get('channel_regime')}",
        f"price={marker.get('price')}",
        f"qty={marker.get('qty')}",
        f"quote_value={marker.get('quote_value')}",
        f"fee={marker.get('fee')}",
    ]
    return "; ".join(item for item in fields if not item.endswith("=None"))


def _series_extent(frame: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    values: list[float] = []
    for column in columns:
        if column in frame:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            values.extend(float(value) for value in series)
    if not values:
        return 0.0, 1.0
    return min(values), max(values)


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return pd.notna(number)


def _chart_meta(frame: pd.DataFrame, ts_code: str, trade_date: str, start_date: str) -> dict[str, object]:
    first = frame.iloc[0]
    return {
        "code": ts_code,
        "symbol": first.get("symbol"),
        "name": first.get("name"),
        "trade_date": trade_date,
        "start_date": start_date,
    }


def _default_chart_output(code: str, trade_date: str) -> Path:
    return Path("reports") / "charts" / f"{code}_{trade_date}.html"


def _compact_chart_date(value: str) -> str:
    text = str(value or "").strip()
    if text == "latest":
        return text
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = text.replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise UserInputError("Date must use YYYYMMDD or YYYY-MM-DD format")
    return text
