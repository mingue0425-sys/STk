"""관심종목(워치리스트) 일괄 예측 + 대시보드 HTML.

    python -m stk watch AAPL MSFT NVDA TSLA          # 4종목 병렬 예측 → output/watch/index.html
    python -m stk watch --file watchlist.txt -H 3m   # 파일에서 종목 읽기 (한 줄에 하나, # 주석 가능)
"""

from __future__ import annotations

import html
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DataError, get_history, normalize_symbol
from .forecast import Forecast, run_forecast
from .html_report import CSS as REPORT_CSS
from .html_report import render_html
from .report import DISCLAIMER, money, pct

DIRECTION = {
    2: ("▲▲", "up"),
    1: ("▲", "up"),
    0: ("■", "flat"),
    -1: ("▼", "down"),
    -2: ("▼▼", "down"),
}


def read_watchlist(path: str | Path) -> list[str]:
    """한 줄에 하나 또는 쉼표/공백 구분. '#' 이후는 주석."""
    tickers: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        for tok in line.replace(",", " ").split():
            sym = normalize_symbol(tok)
            if sym and sym not in tickers:
                tickers.append(sym)
    return tickers


@dataclass
class WatchItem:
    symbol: str
    forecast: Forecast | None
    report: str | None      # 개별 HTML 리포트 파일명
    error: str | None = None
    history: list[float] | None = None


def _forecast_one(symbol: str, opts: dict) -> WatchItem:
    """작업 프로세스에서 실행: 데이터 로드 → 예측 → 개별 HTML 저장."""
    try:
        prices = get_history(symbol, years=opts["years"], cache_dir=opts["cache_dir"], refresh=opts["refresh"])
        fc = run_forecast(prices, symbol, **opts["forecast"])
        out = Path(opts["out"])
        name = f"{symbol}_{fc.stats.last_date:%Y%m%d}_{fc.horizon}d.html"
        (out / name).write_text(render_html(fc, prices), encoding="utf-8")
        hist = prices["adj_close"].dropna().iloc[-63:].round(4).tolist()
        fc.mix_samples = None  # 프로세스 간 전송량 절감
        return WatchItem(symbol, fc, name, history=hist)
    except (DataError, ValueError, OSError) as err:
        return WatchItem(symbol, None, None, error=str(err))


def run_watchlist(
    symbols: list[str],
    out_dir: str | Path,
    forecast_kwargs: dict,
    years: float = 10,
    cache_dir=None,
    refresh: bool = False,
    jobs: int | None = None,
    progress: bool = False,
) -> list[WatchItem]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"years": years, "cache_dir": cache_dir, "refresh": refresh, "out": str(out), "forecast": forecast_kwargs}
    jobs = max(1, min(jobs or os.cpu_count() or 1, len(symbols)))
    items: dict[str, WatchItem] = {}
    if jobs == 1:
        for i, sym in enumerate(symbols, 1):
            items[sym] = _forecast_one(sym, opts)
            if progress:
                print(f"\r  예측 진행 {i}/{len(symbols)}", end="", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_forecast_one, sym, opts): sym for sym in symbols}
            for i, fut in enumerate(as_completed(futures), 1):
                items[futures[fut]] = fut.result()
                if progress:
                    print(f"\r  예측 진행 {i}/{len(symbols)}", end="", flush=True)
    if progress:
        print()
    return [items[s] for s in symbols]


# ------------------------------------------------------------------ 대시보드
def _spark(item: WatchItem, width: int = 168, height: int = 44) -> str:
    """최근 3개월 주가 + 예측 분포(50%/90%) + 중앙값을 그린 작은 SVG."""
    fc = item.forecast
    hist = item.history or []
    n_hist, n_fut = len(hist), fc.horizon + 1
    total = n_hist + n_fut - 1
    q = fc.mix_quantiles
    vals = hist + list(q[0.05]) + list(q[0.95])
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1
    x = lambda i: 2 + i / max(total - 1, 1) * (width - 4)
    y = lambda v: 2 + (1 - (v - lo) / (hi - lo)) * (height - 4)

    def poly(series, offset):
        return " ".join(f"{x(offset + i):.1f},{y(v):.1f}" for i, v in enumerate(series))

    def area(top, bottom, offset):
        pts = [f"{x(offset + i):.1f},{y(v):.1f}" for i, v in enumerate(top)]
        pts += [f"{x(offset + i):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(bottom)))]
        return " ".join(pts)

    off = n_hist - 1
    title = html.escape(f"{fc.symbol}: 최근 3개월 + {fc.horizon}거래일 예측 (음영 50%·90% 구간)")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">'
        f"<title>{title}</title>"
        f'<polygon points="{area(q[0.95], q[0.05], off)}" class="b90"/>'
        f'<polygon points="{area(q[0.75], q[0.25], off)}" class="b50"/>'
        f'<polyline points="{poly(hist, 0)}" class="hist"/>'
        f'<polyline points="{poly(q[0.5], off)}" class="med"/>'
        f'<line x1="{x(off):.1f}" x2="{x(off):.1f}" y1="2" y2="{height - 2}" class="now"/>'
        "</svg>"
    )


def _row(item: WatchItem) -> str:
    e = html.escape
    if item.forecast is None:
        msg = (item.error or "").removeprefix(f"{item.symbol}: ")
        return (f'<tr class="err"><td class="sym">{e(item.symbol)}</td>'
                f'<td colspan="9">오류: {e(msg)}</td></tr>')
    fc, ol = item.forecast, item.forecast.outlook
    icon, cls = DIRECTION[ol.direction]
    regime = "-"
    regime_key = -1.0
    if fc.regime is not None:
        st = fc.regime.state
        regime = f"{st.names[st.current_regime]} {st.current[st.current_regime]:.0%}"
        regime_key = float(st.current_regime)
    var95 = fc.risk.var95 if fc.risk is not None else float("nan")
    cells = [
        (f'<a href="{e(item.report)}">{e(fc.symbol)}</a>', fc.symbol, "sym"),
        (money(fc.s0), fc.s0, "num"),
        (f'<span class="dir {cls}"><span aria-hidden="true">{icon}</span> {e(ol.label)}</span>', ol.direction * 10 + ol.prob_up, ""),
        (pct(ol.prob_up, False, 0), ol.prob_up, "num"),
        (pct(ol.expected_return), ol.expected_return, "num"),
        (f"{money(ol.band90[0])} ~ {money(ol.band90[1])}", (ol.band90[1] - ol.band90[0]) / fc.s0, "num"),
        (pct(-var95), -var95 if math.isfinite(var95) else 0, "num"),
        (e(regime), regime_key, ""),
        (f"{fc.stats.trend_score:+.2f}", fc.stats.trend_score, "num"),
        (_spark(item), 0, "sp"),
    ]
    return "<tr>" + "".join(
        f'<td class="{c}" data-v="{html.escape(str(v))}">{t}</td>' for t, v, c in cells
    ) + "</tr>"


def render_dashboard(items: list[WatchItem], horizon: int, generated: datetime | None = None) -> str:
    ok = [it for it in items if it.forecast is not None]
    generated = generated or datetime.now()
    counts = {"up": 0, "flat": 0, "down": 0}
    for it in ok:
        counts[DIRECTION[it.forecast.outlook.direction][1]] += 1
    as_of = max((it.forecast.stats.last_date for it in ok), default=None)
    rows = "".join(_row(it) for it in items)
    summary = [
        ("분석 종목", f"{len(ok)} / {len(items)}"),
        ("상승 우위", f"{counts['up']}"),
        ("횡보", f"{counts['flat']}"),
        ("하락 우위", f"{counts['down']}"),
    ]
    avg_up = np.mean([it.forecast.outlook.prob_up for it in ok]) if ok else float("nan")
    summary.append(("평균 상승확률", pct(float(avg_up), False, 0) if ok else "-"))
    market = ok[0].forecast.market if ok else None
    market_txt = f"시장 스트레스 {market.stress:+.2f} ({market.label}) · VIX {market.vix:.1f}" if market and market.vix else ""
    tiles = "".join(f'<div class="kpi"><div class="kpi-label">{a}</div><div class="kpi-value">{b}</div></div>'
                    for a, b in summary)
    headers = ["종목", "현재가", "전체 흐름", "상승확률", "기대수익률", "90% 범위", "95% VaR", "현재 국면", "추세", "흐름"]
    th = "".join(
        f'<th class="{"num" if i in (1, 3, 4, 5, 6, 8) else ""}"><button type="button" data-col="{i}">{h}</button></th>'
        if i < 9 else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>관심종목 흐름 예측</title>
<style>{REPORT_CSS}{DASH_CSS}</style>
</head>
<body>
<main class="wrap">
<header class="top">
  <h1>관심종목 흐름 예측 <span class="h-sub">{horizon}거래일 · 시나리오 + 국면 앙상블</span></h1>
  <p class="meta">기준일 {as_of:%Y-%m-%d} · 생성 {generated:%Y-%m-%d %H:%M} {('· ' + html.escape(market_txt)) if market_txt else ''}</p>
</header>
<section class="card"><div class="kpis five">{tiles}</div></section>
<section class="card">
  <h2>종목별 요약 <span class="h-sub">열 제목을 누르면 정렬 · 종목명을 누르면 상세 리포트</span></h2>
  <div class="tbl-wrap"><table id="watch"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
  <p class="note">흐름: 최근 3개월 주가(실선) + 예측 중앙값(점선) · 음영은 50%·90% 구간. 방향 판정은 상승 마감 확률 기준이며 방향 예측력은 제한적입니다 (README의 백테스트 참고).</p>
</section>
<footer class="foot">{html.escape(DISCLAIMER)}</footer>
</main>
<script>{DASH_JS}</script>
</body>
</html>
"""


DASH_CSS = r"""
.kpis.five{grid-template-columns:repeat(5,1fr)}
#watch th button{all:unset;cursor:pointer;color:var(--text2);font-weight:600;font-size:12px}
#watch th button:focus-visible{outline:2px solid var(--s0);outline-offset:2px}
#watch th button[data-dir="asc"]::after{content:" ▲";font-size:9px}
#watch th button[data-dir="desc"]::after{content:" ▼";font-size:9px}
#watch td{vertical-align:middle}
#watch td.sym a{color:var(--text);font-weight:650;text-decoration:none;border-bottom:1px solid var(--axis)}
#watch td.sym a:hover{border-bottom-color:var(--text)}
#watch tr.err td{color:var(--muted)}
.dir{display:inline-flex;gap:6px;align-items:center;white-space:nowrap}
.dir.up span{color:var(--s0)}.dir.down span{color:var(--s7)}.dir.flat span{color:var(--muted)}
.spark{display:block}
.spark .b90{fill:var(--band);fill-opacity:var(--band90)}
.spark .b50{fill:var(--band);fill-opacity:var(--band50)}
.spark .hist{fill:none;stroke:var(--text);stroke-width:1.3}
.spark .med{fill:none;stroke:var(--text);stroke-width:1.3;stroke-dasharray:3 2}
.spark .now{stroke:var(--muted);stroke-width:1;stroke-dasharray:1 2}
@media (max-width:860px){.kpis.five{grid-template-columns:repeat(2,1fr)}}
"""

DASH_JS = r"""
(function(){
const table = document.getElementById('watch'); if (!table) return;
const body = table.tBodies[0];
table.querySelectorAll('th button').forEach(btn => btn.addEventListener('click', () => {
  const col = +btn.dataset.col; const dir = btn.dataset.dir === 'desc' ? 'asc' : 'desc';
  table.querySelectorAll('th button').forEach(b => delete b.dataset.dir); btn.dataset.dir = dir;
  const rows = Array.from(body.rows).filter(r => !r.classList.contains('err'));
  const errs = Array.from(body.rows).filter(r => r.classList.contains('err'));
  rows.sort((a, b) => { const va = a.cells[col].dataset.v, vb = b.cells[col].dataset.v;
    const na = parseFloat(va), nb = parseFloat(vb);
    const c = (!isNaN(na) && !isNaN(nb)) ? na - nb : String(va).localeCompare(String(vb));
    return dir === 'asc' ? c : -c; });
  rows.concat(errs).forEach(r => body.appendChild(r));
}));
})();
"""


def summary_json(items: list[WatchItem]) -> str:
    out = []
    for it in items:
        if it.forecast is None:
            out.append({"symbol": it.symbol, "error": it.error})
            continue
        d = it.forecast.to_dict()
        d.pop("path", None)
        d["report"] = it.report
        out.append(d)
    return json.dumps(out, ensure_ascii=False, indent=2, default=_default)


def _default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(type(obj))
