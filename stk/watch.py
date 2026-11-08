"""관심종목(워치리스트) 일괄 예측 + 대시보드 HTML.

    python -m stk watch AAPL MSFT NVDA TSLA          # 4종목 병렬 예측 → output/watch/index.html
    python -m stk watch --file watchlist.txt -H 3m   # 파일에서 종목 읽기 (한 줄에 하나, # 주석 가능)
"""

from __future__ import annotations

import html
import json
import math
import os
import time
from collections import Counter
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
from .narrative import narrative_text
from .report import DISCLAIMER, money, pct

FLIP_MIN_DELTA = 0.02  # 흐름 판정이 바뀌었다고 표시할 최소 상승확률 변화
STALE_RETRY_WAIT = 5.0  # 다른 종목보다 데이터가 늦은 종목을 다시 받기 전 대기 (초)

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
class Change:
    """같은 종목·기간·모델의 직전 예측 대비 변화."""

    prev_as_of: str
    prev_price: float
    price_change: float
    d_prob_up: float
    d_expected: float
    prev_outlook: str
    outlook_changed: bool
    prev_regime: str | None
    regime_changed: bool


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
        expect = opts.get("expect_date")
        if expect is not None and fc.stats.last_date < expect:
            lag = int(np.busday_count(fc.stats.last_date.date(), expect.date()))
            fc.data_warnings.append(
                f"다른 종목보다 데이터가 {lag}거래일 늦습니다 (마지막 {fc.stats.last_date:%Y-%m-%d}, 다른 종목 {expect:%Y-%m-%d}) "
                "— 데이터 제공처 지연으로 이전 날짜 기준 예측입니다")
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
    _retry_stale(items, opts)
    return [items[s] for s in symbols]


def reference_date(dates: list[pd.Timestamp]) -> pd.Timestamp | None:
    """종목들의 마지막 데이터 날짜 중 가장 흔한 날짜 (3종목 미만이면 None)."""
    if len(dates) < 3:
        return None
    (ref, _), = Counter(dates).most_common(1)
    return ref


def _retry_stale(items: dict[str, WatchItem], opts: dict) -> None:
    """다른 종목보다 마지막 데이터가 이른 종목은 캐시 없이 한 번 더 받고, 그래도 늦으면 경고를 붙입니다."""
    ref = reference_date([it.forecast.stats.last_date for it in items.values() if it.forecast is not None])
    if ref is None:
        return
    stale = [s for s, it in items.items() if it.forecast is not None and it.forecast.stats.last_date < ref]
    if not stale:
        return
    time.sleep(STALE_RETRY_WAIT)
    retry_opts = dict(opts, refresh=True, expect_date=ref)
    for sym in stale:
        old = items[sym]
        new = _forecast_one(sym, retry_opts)
        if new.forecast is None:  # 재시도 자체가 실패 → 기존 결과에 경고만 추가
            lag = int(np.busday_count(old.forecast.stats.last_date.date(), ref.date()))
            old.forecast.data_warnings.append(
                f"다른 종목보다 데이터가 {lag}거래일 늦습니다 (마지막 {old.forecast.stats.last_date:%Y-%m-%d}, "
                f"다른 종목 {ref:%Y-%m-%d})")
            continue
        if old.report and old.report != new.report:
            (Path(opts["out"]) / old.report).unlink(missing_ok=True)
        items[sym] = new


def compute_changes(items: list[WatchItem], records: list[dict]) -> dict[str, Change]:
    """예측 기록에서 직전 예측(기준일이 더 이른 것 중 최신)을 찾아 변화를 계산합니다."""
    out: dict[str, Change] = {}
    for it in items:
        fc = it.forecast
        if fc is None:
            continue
        as_of = fc.stats.last_date.strftime("%Y-%m-%d")
        prev = [r for r in records if r.get("symbol") == fc.symbol and r.get("horizon") == fc.horizon
                and r.get("model") == fc.model and r.get("as_of", "") < as_of]
        if not prev:
            continue
        p = max(prev, key=lambda r: (r["as_of"], r.get("created", "")))
        regime_now = fc.regime.state.names[fc.regime.state.current_regime] if fc.regime is not None else None
        out[fc.symbol] = Change(
            prev_as_of=p["as_of"],
            prev_price=float(p["price"]),
            price_change=fc.s0 / float(p["price"]) - 1,
            d_prob_up=fc.mix_prob_up - float(p["prob_up"]),
            d_expected=(fc.mix_expected_price / fc.s0 - 1) - float(p["expected_return"]),
            prev_outlook=p.get("outlook", ""),
            # 경계값 부근의 미세한 흔들림(예: 55%↔54%)은 흐름 변화로 보지 않음
            outlook_changed=p.get("outlook") != fc.outlook.label and abs(fc.mix_prob_up - float(p["prob_up"])) >= FLIP_MIN_DELTA,
            prev_regime=p.get("regime"),
            regime_changed=bool(p.get("regime")) and p.get("regime") != regime_now,
        )
    return out


def _delta_html(ch: Change | None) -> tuple[str, float]:
    if ch is None:
        return '<span class="muted">신규</span>', 0.0
    d = ch.d_prob_up
    arrow, cls = ("▲", "up") if d > 0.005 else ("▼", "down") if d < -0.005 else ("–", "flat")
    title = (f"직전 예측 {ch.prev_as_of}: 주가 {money(ch.prev_price)} ({pct(ch.price_change)}) · "
             f"상승확률 {pct(d, digits=1)}p · 기대수익률 {pct(ch.d_expected, digits=1)}p"
             + (f" · 흐름: {ch.prev_outlook} → 현재" if ch.outlook_changed else ""))
    return (f'<span class="delta {cls}" title="{html.escape(title)}"><span aria-hidden="true">{arrow}</span> '
            f'{d * 100:+.0f}%p</span>'), d


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


def _sector_cell(fc: Forecast) -> str:
    s = fc.sector
    if s is None:
        return '<span class="muted">-</span>'
    rel = f" {s.rel_3m * 100:+.0f}%p" if math.isfinite(s.rel_3m) else ""
    weak = " muted" if s.method == "raw" else ""
    title = (f"{s.name}({s.etf}) 연관도 {s.fit:.2f} · 위험 분해 시장 {s.share_market:.0%} / 섹터 {s.share_sector:.0%} / "
             f"고유 {s.share_idio:.0%} · 섹터 추세 {s.sector_label}")
    return f'<span class="sec{weak}" title="{html.escape(title)}">{html.escape(s.etf)} {html.escape(s.name)}{rel}</span>'


def _row(item: WatchItem, change: Change | None = None) -> str:
    e = html.escape
    if item.forecast is None:
        msg = (item.error or "").removeprefix(f"{item.symbol}: ")
        return (f'<tr class="err"><td class="sym">{e(item.symbol)}</td>'
                f'<td colspan="11">오류: {e(msg)}</td></tr>')
    fc, ol = item.forecast, item.forecast.outlook
    icon, cls = DIRECTION[ol.direction]
    regime = "-"
    regime_key = -1.0
    if fc.regime is not None:
        st = fc.regime.state
        regime = f"{st.names[st.current_regime]} {st.current[st.current_regime]:.0%}"
        regime_key = float(st.current_regime)
        if change is not None and change.regime_changed:
            regime += f" ↺ (직전 {change.prev_regime})"
    delta_html, delta_key = _delta_html(change)
    flow_badge = ' <span class="badge" title="직전 예측과 흐름 판정이 달라짐">흐름 변화</span>' if change and change.outlook_changed else ""
    var95 = fc.risk.var95 if fc.risk is not None else float("nan")
    cells = [
        (f'<a href="{e(item.report)}">{e(fc.symbol)}</a>'
         + (f' <span class="dw" title="{e(" / ".join(fc.data_warnings))}" aria-label="데이터 경고">⚠</span>' if fc.data_warnings else ""),
         fc.symbol, "sym"),
        (money(fc.s0), fc.s0, "num"),
        (f'<span class="dir {cls}" title="{e(narrative_text(fc))}"><span aria-hidden="true">{icon}</span> {e(ol.label)}</span>{flow_badge}',
         ol.direction * 10 + ol.prob_up, ""),
        (pct(ol.prob_up, False, 0), ol.prob_up, "num"),
        (delta_html, delta_key, "num"),
        (pct(ol.expected_return), ol.expected_return, "num"),
        (f"{money(ol.band90[0])} ~ {money(ol.band90[1])}", (ol.band90[1] - ol.band90[0]) / fc.s0, "num"),
        (pct(-var95), -var95 if math.isfinite(var95) else 0, "num"),
        (e(regime), regime_key, ""),
        (_sector_cell(fc), fc.sector.rel_3m if fc.sector is not None and math.isfinite(fc.sector.rel_3m) else -9, ""),
        (f"{fc.stats.trend_score:+.2f}", fc.stats.trend_score, "num"),
        (_spark(item), 0, "sp"),
    ]
    return "<tr>" + "".join(
        f'<td class="{c}" data-v="{html.escape(str(v))}">{t}</td>' for t, v, c in cells
    ) + "</tr>"


def _track_card(track) -> str:
    if track is None:
        return ""
    s = track.summary
    if not s.get("evaluated"):
        body = (f'<p class="lead">기록 {s["records"]}건 · 평가 대기 {s["pending"]}개 — 체크포인트(1주·1개월…) 날짜가 지나면 '
                "실제 주가와 비교한 적중률이 여기에 쌓입니다.</p>")
    else:
        tiles = [
            ("평가된 체크포인트", f"{s['evaluated']}개"),
            ("50% 구간 적중", f"{pct(s['coverage50'], False, 0)} (목표 50%)"),
            ("90% 구간 적중", f"{pct(s['coverage90'], False, 0)} (목표 90%)"),
            ("방향 적중 (만기)", f"{pct(s['hit_rate'], False, 0)} · {s['final_n']}건" if s.get("final_n") else "만기 도달 전"),
            ("평균 PIT", f"{s['mean_pit']:.2f} (목표 0.50)"),
        ]
        body = '<div class="kpis five">' + "".join(
            f'<div class="kpi"><div class="kpi-label">{a}</div><div class="kpi-value small">{b}</div></div>' for a, b in tiles
        ) + "</div>"
        if s["evaluated"] < 30:
            body += '<p class="note">표본이 30개 미만이라 아직 통계적으로 의미 있는 평가는 아닙니다.</p>'
    return f'<section class="card"><h2>실전 성과 <span class="h-sub">기록해 둔 과거 예측 vs 실제 주가</span></h2>{body}</section>'


def render_dashboard(
    items: list[WatchItem],
    horizon: int,
    generated: datetime | None = None,
    changes: dict[str, Change] | None = None,
    track=None,
    nav: list[tuple[str, str, bool]] | None = None,
) -> str:
    """nav: [(라벨, 링크, 현재 페이지 여부)] — 여러 예측 기간 대시보드를 오갈 때 사용."""
    changes = changes or {}
    ok = [it for it in items if it.forecast is not None]
    nav_html = ""
    if nav:
        current = ' aria-current="page"'
        links = [f'<a href="{html.escape(href)}"{current if cur else ""}>{html.escape(lbl)}</a>' for lbl, href, cur in nav]
        nav_html = '<nav class="hnav" aria-label="예측 기간">' + "".join(links) + "</nav>"
    generated = generated or datetime.now()
    counts = {"up": 0, "flat": 0, "down": 0}
    for it in ok:
        counts[DIRECTION[it.forecast.outlook.direction][1]] += 1
    as_of = max((it.forecast.stats.last_date for it in ok), default=None)
    rows = "".join(_row(it, changes.get(it.symbol)) for it in items)
    n_flow = sum(1 for c in changes.values() if c.outlook_changed)
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
    headers = ["종목", "현재가", "전체 흐름", "상승확률", "직전 대비", "기대수익률", "90% 범위", "95% VaR", "현재 국면", "섹터 (3M 상대)",
               "추세", "흐름"]
    th = "".join(
        f'<th class="{"num" if i in (1, 3, 4, 5, 6, 7, 10) else ""}"><button type="button" data-col="{i}">{h}</button></th>'
        if i < 11 else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    change_note = (f" · 직전 예측 대비 흐름 판정이 바뀐 종목 {n_flow}개" if changes else "")
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
  {nav_html}
</header>
<section class="card"><div class="kpis five">{tiles}</div></section>
{_track_card(track)}
<section class="card">
  <h2>종목별 요약 <span class="h-sub">열 제목을 누르면 정렬 · 종목명을 누르면 상세 리포트</span></h2>
  <div class="tbl-wrap"><table id="watch"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>
  <p class="note">흐름: 최근 3개월 주가(실선) + 예측 중앙값(점선) · 음영은 50%·90% 구간. 직전 대비: 같은 기간 직전 예측과의 상승확률 차이(마우스를 올리면 상세){change_note}. 방향 판정은 상승 마감 확률 기준이며 방향 예측력은 제한적입니다 (README의 백테스트 참고).</p>
</section>
<footer class="foot">{html.escape(DISCLAIMER)}</footer>
</main>
<script>{DASH_JS}</script>
</body>
</html>
"""


DASH_CSS = r"""
.wrap{max-width:1440px}
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
.dw{cursor:help;color:var(--s3);font-size:12px}
.delta{white-space:nowrap;cursor:help}.delta.up span{color:var(--s0)}.delta.down span{color:var(--s7)}.delta.flat span{color:var(--muted)}
.muted{color:var(--muted)}
.badge{display:inline-block;margin-left:6px;padding:0 6px;border:1px solid var(--axis);border-radius:999px;font-size:11px;color:var(--text2)}
.kpi-value.small{font-size:15px}
.sec{white-space:nowrap;cursor:help}
.hnav{display:flex;gap:6px;flex-wrap:wrap;margin:-6px 0 16px}
.hnav a{padding:4px 12px;border:1px solid var(--grid);border-radius:999px;color:var(--text2);text-decoration:none;font-size:13px}
.hnav a[aria-current="page"]{background:var(--text);color:var(--surface);border-color:var(--text)}
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


def render_markdown_summary(
    items: list[WatchItem],
    horizon: int,
    changes: dict[str, Change] | None = None,
    track=None,
    generated: datetime | None = None,
) -> str:
    """GitHub 에서 바로 보이는 요약 (reports 브랜치 README · Actions 작업 요약)."""
    changes = changes or {}
    ok = [it for it in items if it.forecast is not None]
    generated = generated or datetime.now()
    as_of = max((it.forecast.stats.last_date for it in ok), default=None)
    lines = [f"# 관심종목 흐름 예측 ({horizon}거래일)", ""]
    meta = f"기준일 **{as_of:%Y-%m-%d}** · 생성 {generated:%Y-%m-%d %H:%M}" if as_of is not None else f"생성 {generated:%Y-%m-%d %H:%M}"
    market = ok[0].forecast.market if ok else None
    if market is not None and market.vix is not None:
        meta += f" · 시장 스트레스 {market.stress:+.2f} ({market.label}) · VIX {market.vix:.1f}"
    lines += [meta, "", "| 종목 | 현재가 | 전체 흐름 | 상승확률 | 직전 대비 | 기대수익률 | 90% 범위 | 95% VaR | 현재 국면 | 섹터 | 다음 실적(추정) |",
              "|---|---:|---|---:|---:|---:|---:|---:|---|---|---|"]
    for it in items:
        if it.forecast is None:
            lines.append(f"| {it.symbol} | 오류: {(it.error or '').removeprefix(it.symbol + ': ')} | | | | | | | | | |")
            continue
        fc, ol = it.forecast, it.forecast.outlook
        ch = changes.get(fc.symbol)
        delta = "신규" if ch is None else f"{ch.d_prob_up * 100:+.0f}%p" + (" ⟳흐름" if ch.outlook_changed else "")
        regime = "-"
        if fc.regime is not None:
            st = fc.regime.state
            regime = f"{st.names[st.current_regime]} {st.current[st.current_regime]:.0%}"
        warn = " ⚠" if fc.data_warnings else ""
        link = f"[{fc.symbol}]({it.report})" if it.report else fc.symbol
        lines.append(
            f"| {link}{warn} | {money(fc.s0)} | {DIRECTION[ol.direction][0]} {ol.label} | {pct(ol.prob_up, False, 0)} | {delta} | "
            f"{pct(ol.expected_return)} | {money(ol.band90[0])} ~ {money(ol.band90[1])} | "
            f"{pct(-fc.risk.var95) if fc.risk else '-'} | {regime} | {fc.sector.etf if fc.sector else '-'} | "
            f"{fc.next_earnings[0].strftime('%m-%d') if fc.next_earnings else '-'} |"
        )
    if track is not None:
        s = track.summary
        lines += ["", "## 실전 성과 (기록된 예측 vs 실제)", ""]
        if s.get("evaluated"):
            lines.append(
                f"평가 {s['evaluated']}개 · 50% 구간 적중 {pct(s['coverage50'], False, 0)} (목표 50%) · "
                f"90% 구간 적중 {pct(s['coverage90'], False, 0)} (목표 90%) · 평균 PIT {s['mean_pit']:.2f}"
                + (f" · 만기 방향 적중 {pct(s['hit_rate'], False, 0)} ({s['final_n']}건)" if s.get("final_n") else "")
            )
        else:
            lines.append(f"기록 {s['records']}건 · 평가 대기 {s['pending']}개 (체크포인트 날짜가 지나면 집계)")
    warns = [(it.symbol, w) for it in ok for w in it.forecast.data_warnings]
    if warns:
        lines += ["", "## 데이터 점검", ""] + [f"- ⚠ {sym}: {w}" for sym, w in warns]
    lines += ["", f"_{DISCLAIMER}_", ""]
    return "\n".join(lines)
