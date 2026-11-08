"""여러 예측 기간(예: 1개월·6개월) 대시보드를 한 번에 만들어 정적 사이트로 묶습니다.

    python -m stk publish --file watchlist.txt --horizons 1m,6m --site site

site/
  index.html              첫 화면: 기간별 요약 카드 · 흐름 변화 · 실전 성과
  README.md               GitHub 에서 바로 보이는 요약
  latest/<1m|6m>/         기간별 대시보드(index.html) · 종목별 리포트 · summary.md/json
  history/YYYY-MM-DD.md   날짜별 요약 (파일 이름 = 데이터 기준일, 실행한 날 아님) + 같은 이름의 .html
  history/index.html      지난 기록 목록 (GitHub Pages 는 폴더 목록을 보여 주지 않으므로 직접 생성)
  data/forecasts.jsonl    누적 예측 기록 (실전 성과 추적용)
  data/latest.json        이번 실행 상태 (기준일 · 기간별 종목 수 · 오류 · 데이터 경고)
  alerts.md               흐름 변화·국면 급변·데이터 경고가 있을 때만 생성 (알림용)
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .forecast import horizon_label
from .html_report import CSS as REPORT_CSS
from .report import DISCLAIMER, pct
from .tracking import TrackResult, append_log, evaluate, load_log
from .watch import (
    DASH_CSS,
    DIRECTION,
    WatchItem,
    compute_changes,
    render_dashboard,
    render_markdown_summary,
    run_watchlist,
    summary_json,
)


def horizon_code(days: int) -> str:
    return {21: "1m", 63: "3m", 126: "6m", 252: "1y", 504: "2y"}.get(days, f"{days}d")


@dataclass
class HorizonRun:
    horizon: int
    code: str
    items: list[WatchItem]
    changes: dict = field(default_factory=dict)

    @property
    def ok(self) -> list[WatchItem]:
        return [it for it in self.items if it.forecast is not None]


@dataclass
class PublishResult:
    runs: list[HorizonRun]
    track: TrackResult | None
    alerts: list[str]
    site: Path


def _track_for(records: list[dict], horizon: int, loader) -> TrackResult:
    return evaluate([r for r in records if r.get("horizon") == horizon], loader)


def build_alerts(runs: list[HorizonRun]) -> list[str]:
    """알림거리: 흐름 판정 변화(상승확률 2%p 이상), 고변동 국면 진입, 데이터 경고."""
    alerts = []
    regime_seen: set[str] = set()  # 국면은 기간과 무관한 종목 상태 → 종목당 한 번만
    for run in runs:
        label = horizon_label(run.horizon)
        for it in run.ok:
            fc = it.forecast
            ch = run.changes.get(fc.symbol)
            if ch is not None and ch.outlook_changed:
                alerts.append(f"[{label}] {fc.symbol}: 흐름 판정 '{ch.prev_outlook}' → '{fc.outlook.label}' "
                              f"(상승확률 {ch.d_prob_up * 100:+.0f}%p, 직전 예측 {ch.prev_as_of})")
            if ch is not None and ch.regime_changed and fc.regime is not None and fc.symbol not in regime_seen:
                st = fc.regime.state
                if st.current_regime == len(st.names) - 1:
                    regime_seen.add(fc.symbol)
                    alerts.append(f"{fc.symbol}: '{st.names[-1]}' 국면 진입 ({st.current[-1]:.0%}, 직전 {ch.prev_regime})")
        if run is runs[0]:  # 데이터 경고는 한 번만
            for it in run.ok:
                for w in it.forecast.data_warnings:
                    alerts.append(f"{it.symbol}: ⚠ {w}")
    return alerts


def _landing(runs: list[HorizonRun], track: TrackResult | None, per_h: dict[int, TrackResult], generated: datetime) -> str:
    e = html.escape
    ok_all = [it for r in runs for it in r.ok]
    as_of = max((it.forecast.stats.last_date for it in ok_all), default=None)
    market = ok_all[0].forecast.market if ok_all else None
    meta = f"기준일 {as_of:%Y-%m-%d} · 생성 {generated:%Y-%m-%d %H:%M}" if as_of is not None else ""
    if market is not None and market.vix is not None:
        meta += f" · 시장 스트레스 {market.stress:+.2f} ({e(market.label)}) · VIX {market.vix:.1f}"
    cards = []
    for run in runs:
        counts = {"up": 0, "flat": 0, "down": 0}
        for it in run.ok:
            counts[DIRECTION[it.forecast.outlook.direction][1]] += 1
        top = sorted(run.ok, key=lambda it: -it.forecast.mix_prob_up)[:3]
        flips = [s for s, c in run.changes.items() if c.outlook_changed]
        tr = per_h.get(run.horizon)
        tr_txt = (f"실전 성과: 평가 {tr.summary['evaluated']}개 · 90% 적중 {pct(tr.summary['coverage90'], False, 0)}"
                  if tr is not None and tr.summary.get("evaluated") else "실전 성과: 평가 대기")
        cards.append(f"""
<a class="card hcard" href="latest/{run.code}/index.html">
  <h2>{e(horizon_label(run.horizon))} 전망 <span class="h-sub">{run.horizon}거래일 · {len(run.ok)}종목</span></h2>
  <div class="kpis three">
    <div class="kpi"><div class="kpi-label">상승 우위</div><div class="kpi-value">{counts['up']}</div></div>
    <div class="kpi"><div class="kpi-label">횡보</div><div class="kpi-value">{counts['flat']}</div></div>
    <div class="kpi"><div class="kpi-label">하락 우위</div><div class="kpi-value">{counts['down']}</div></div>
  </div>
  <p class="lead">상승확률 상위: {e(', '.join(f"{it.symbol} {it.forecast.mix_prob_up:.0%}" for it in top))}</p>
  <p class="note">{('흐름 변화: ' + e(', '.join(flips))) if flips else '직전 대비 흐름 변화 없음'} · {e(tr_txt)}</p>
  <span class="go">대시보드 열기 →</span>
</a>""")
    track_txt = ""
    if track is not None:
        s = track.summary
        if s.get("evaluated"):
            track_txt = (f"<p>평가된 체크포인트 {s['evaluated']}개 · 50% 구간 적중 {pct(s['coverage50'], False, 0)} (목표 50%) · "
                         f"90% 구간 적중 {pct(s['coverage90'], False, 0)} (목표 90%) · 평균 PIT {s['mean_pit']:.2f}"
                         + (f" · 만기 방향 적중 {pct(s['hit_rate'], False, 0)} ({s['final_n']}건)" if s.get("final_n") else "")
                         + "</p>")
        else:
            track_txt = f"<p>기록 {s['records']}건 · 평가 대기 {s['pending']}개 — 체크포인트 날짜가 지나면 실제 주가와 비교한 적중률이 쌓입니다.</p>"
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>관심종목 흐름 예측</title>
<style>{REPORT_CSS}{DASH_CSS}{LANDING_CSS}</style>
</head>
<body>
<main class="wrap">
<header class="top">
  <h1>관심종목 흐름 예측 <span class="h-sub">시나리오 + 국면 앙상블</span></h1>
  <p class="meta">{meta}</p>
</header>
<div class="hgrid">{''.join(cards)}</div>
<section class="card"><h2>실전 성과 <span class="h-sub">기록해 둔 과거 예측 vs 실제 주가</span></h2>{track_txt}</section>
<p class="note">지난 기록: <a href="history/index.html">날짜별 요약</a> · 예측 기록: <a href="data/forecasts.jsonl">forecasts.jsonl</a> · 실행 상태: <a href="data/latest.json">latest.json</a></p>
<footer class="foot">{e(DISCLAIMER)}</footer>
</main>
</body>
</html>
"""


LANDING_CSS = r"""
.hgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:16px}
.hgrid .card{margin:0}
.hcard{display:block;color:inherit;text-decoration:none;transition:border-color .15s}
.hcard:hover,.hcard:focus-visible{border-color:var(--text2)}
.kpis.three{grid-template-columns:repeat(3,1fr);margin:6px 0 10px}
.go{font-size:13px;font-weight:600;color:var(--s0)}
"""


def md_to_html(md: str) -> str:
    """이 프로그램이 만드는 요약 마크다운(제목·표·목록·굵게·링크)만 다루는 작은 변환기.

    링크는 글자만 남깁니다 — 지난 날짜의 종목별 리포트 파일은 매일 새로 만들어져 남아 있지 않기 때문.
    """

    def inline(t: str) -> str:
        t = html.escape(t, quote=False)
        t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"^_(.+)_$", r"<em>\1</em>", t)

    out: list[str] = []
    rows: list[list[str]] = []
    items: list[str] = []

    def flush():
        if rows:
            head, body = rows[0], rows[1:]
            out.append('<div class="tbl-wrap"><table><thead><tr>' + "".join(f"<th>{c}</th>" for c in head)
                       + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table></div>")
            rows.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            if items:
                flush()
            if not re.fullmatch(r"\|[\s:|-]+\|", line):
                rows.append([inline(c.strip()) for c in line.strip("|").split("|")])
            continue
        if line.startswith("- "):
            if rows:
                flush()
            items.append(inline(line[2:]))
            continue
        flush()
        m = re.match(r"(#{1,3}) (.*)", line)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
        elif line == "---":
            out.append("<hr>")
        elif line:
            out.append(f"<p>{inline(line)}</p>")
    flush()
    return "\n".join(out)


def _history_page(title: str, body: str, back: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{REPORT_CSS}{DASH_CSS}.hist h2{{margin-top:22px}}.hist ul.days li{{margin:4px 0}}</style>
</head>
<body>
<main class="wrap hist">
<p class="note">{back}</p>
{body}
<footer class="foot">{html.escape(DISCLAIMER)}</footer>
</main>
</body>
</html>
"""


def write_history(site: Path) -> None:
    """history/*.md 마다 .html 을 만들고(없거나 오래된 것만) 날짜 목록 index.html 을 다시 씁니다."""
    hist = site / "history"
    days = sorted(hist.glob("*.md"), reverse=True)
    for md in days:
        page = md.with_suffix(".html")
        if page.exists() and page.stat().st_mtime >= md.stat().st_mtime:
            continue
        text = md.read_text(encoding="utf-8").replace(f"_{DISCLAIMER}_", "")
        text = re.sub(r"\n첫 화면: .*\n", "\n", text)
        text = re.sub(r" — \[대시보드\]\([^)]*\)", "", text)   # 지난 날짜 대시보드는 남아 있지 않음
        body = md_to_html(text.replace("# 관심종목 흐름 예측", f"# 관심종목 흐름 예측 — {md.stem}", 1))
        page.write_text(_history_page(f"흐름 예측 {md.stem}", body,
                                      '<a href="index.html">← 지난 기록 목록</a> · <a href="../index.html">처음으로</a>'),
                        encoding="utf-8")
    links = "".join(f'<li><a href="{md.stem}.html">{md.stem}</a></li>' for md in days)
    (hist / "index.html").write_text(
        _history_page("지난 기록", f'<h1>지난 기록 <span class="h-sub">데이터 기준일별 요약</span></h1><ul class="days">{links}</ul>',
                      '<a href="../index.html">← 처음으로</a>'), encoding="utf-8")


def _prefix_links(md: str, prefix: str) -> str:
    return re.sub(r"\]\(([A-Za-z0-9._-]+\.html)\)", lambda m: f"]({prefix}{m.group(1)})", md)


def publish_site(
    symbols: list[str],
    site: str | Path,
    horizons: list[int],
    forecast_kwargs: dict,
    log_path: str | Path,
    loader,
    years: float = 10,
    cache_dir=None,
    refresh: bool = False,
    jobs: int | None = None,
    progress: bool = False,
    generated: datetime | None = None,
) -> PublishResult:
    """기간별 예측 → 기록 → 실전 성과 → 대시보드·첫 화면·README·history·alerts 생성."""
    site = Path(site)
    generated = generated or datetime.now()
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "history").mkdir(parents=True, exist_ok=True)
    runs: list[HorizonRun] = []
    for h in horizons:
        code = horizon_code(h)
        out = site / "latest" / code
        if out.exists():
            for f in out.iterdir():
                if f.is_file():
                    f.unlink()
        kwargs = dict(forecast_kwargs, horizon=h)
        items = run_watchlist(symbols, out, kwargs, years=years, cache_dir=cache_dir, refresh=refresh, jobs=jobs,
                              progress=progress)
        run = HorizonRun(h, code, items)
        run.changes = compute_changes(items, load_log(log_path))  # 기록 전에 직전 대비 계산
        for it in run.ok:
            append_log(it.forecast, log_path)
        runs.append(run)

    records = load_log(log_path)
    track = evaluate(records, loader)
    per_h = {r.horizon: _track_for(records, r.horizon, loader) for r in runs}
    nav = [(horizon_label(r.horizon), f"../{r.code}/index.html", False) for r in runs]
    md_parts = []
    for run in runs:
        if not run.ok:
            continue
        out = site / "latest" / run.code
        this_nav = [(lbl, href, href == f"../{run.code}/index.html") for lbl, href, _ in nav] + [("처음으로", "../../index.html", False)]
        (out / "index.html").write_text(
            render_dashboard(run.items, run.horizon, generated=generated, changes=run.changes, track=per_h[run.horizon],
                             nav=this_nav), encoding="utf-8")
        md = render_markdown_summary(run.items, run.horizon, changes=run.changes, track=per_h[run.horizon], generated=generated)
        (out / "summary.md").write_text(md, encoding="utf-8")
        (out / "summary.json").write_text(summary_json(run.items), encoding="utf-8")
        body = md.split("\n", 1)[1] if "\n" in md else md
        body = body.replace(f"_{DISCLAIMER}_", "").rstrip()
        md_parts.append(f"## {horizon_label(run.horizon)} 전망 ({run.horizon}거래일) — [대시보드](latest/{run.code}/index.html)\n"
                        + _prefix_links(body, f"latest/{run.code}/"))

    alerts = build_alerts(runs)
    ok_dates = [it.forecast.stats.last_date for r in runs for it in r.ok]
    as_of = max(ok_dates) if ok_dates else None
    (site / "index.html").write_text(_landing(runs, track, per_h, generated), encoding="utf-8")
    readme = "# 관심종목 흐름 예측\n\n" + "\n\n".join(md_parts)
    if alerts:
        readme += "\n\n## 알림\n\n" + "\n".join(f"- {a}" for a in alerts)
    readme += f"\n\n---\n첫 화면: [index.html](index.html) · 지난 요약: [history/](history/) · 예측 기록: [data/forecasts.jsonl](data/forecasts.jsonl)\n\n_{DISCLAIMER}_\n"
    (site / "README.md").write_text(readme, encoding="utf-8")
    day = as_of if as_of is not None else generated  # 지연 실행(예: 토요일 새벽)이어도 데이터 날짜로 저장
    (site / "history" / f"{day:%Y-%m-%d}.md").write_text(readme, encoding="utf-8")
    write_history(site)
    status = {
        "as_of": f"{day:%Y-%m-%d}",
        "generated": generated.isoformat(timespec="seconds"),
        "horizons": {r.code: {"ok": len(r.ok), "total": len(r.items)} for r in runs},
        "errors": [it.error for it in (runs[0].items if runs else []) if it.error],
        "data_warnings": [f"{it.symbol}: {w}" for it in (runs[0].ok if runs else []) for w in it.forecast.data_warnings],
        "alerts": len(alerts),
    }
    (site / "data" / "latest.json").write_text(json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    alert_file = site / "alerts.md"
    if alerts:
        alert_file.write_text(
            f"### {generated:%Y-%m-%d} 관심종목 알림\n\n" + "\n".join(f"- {a}" for a in alerts) + "\n", encoding="utf-8")
    elif alert_file.exists():
        alert_file.unlink()
    (site / ".nojekyll").touch()
    return PublishResult(runs, track, alerts, site)
