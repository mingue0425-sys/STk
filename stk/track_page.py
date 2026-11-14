"""실전 성과 페이지 (site/track.html): 기록해 둔 과거 예측을 실제 주가와 비교해 보여 줍니다.

- 요약: 50%·90% 구간 적중률, 평균 PIT, 서로 다른 평가 날짜 수(실질 표본)
- 예측 기간 × 체크포인트별 표 (1개월·6개월 예측의 1주 체크포인트가 겹쳐 합계가 부풀지 않도록)
- PIT 분포: 잘 보정된 예측이면 10개 칸이 고르게 채워집니다
- 종목별 예측 vs 실제: 날짜마다 그날 예측한 1주 뒤 50%·90% 범위(막대)와 실제 주가(선)

외부 라이브러리 없이 인라인 SVG 로 그리며, 라이트/다크 모드를 따릅니다.
"""

from __future__ import annotations

import html
import math
from datetime import datetime

import numpy as np
import pandas as pd

from .html_report import CSS as REPORT_CSS
from .report import DISCLAIMER, pct
from .tracking import TrackResult, breakdown, by_symbol
from .watch import DASH_CSS

e = html.escape
MAX_DATES = 60

TRACK_CSS = r"""
.wrap{max-width:1180px}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:4px 0 6px}
.tile{border:1px solid var(--grid);border-radius:10px;padding:10px 12px;background:var(--surface)}
.tile .l{font-size:12px;color:var(--text2)}.tile .v{font-size:24px;font-weight:650;letter-spacing:-.01em}
.tile .t{font-size:12px;color:var(--muted)}
.gauge{position:relative;height:8px;border-radius:4px;background:var(--grid);margin-top:6px}
.gauge i{position:absolute;top:0;bottom:0;left:0;border-radius:4px;background:var(--text2)}
.gauge b{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--text)}
.scroll{overflow-x:auto}
table.tr{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
table.tr th,table.tr td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
table.tr th{font-size:12px;color:var(--text2);font-weight:600}
table.tr th:first-child,table.tr td:first-child,table.tr td.l,table.tr th.l{text-align:left}
.off{color:var(--s7)}
.pit .bar{fill:var(--s0)}.pit .bar:hover{fill-opacity:.8}
.ax text{fill:var(--muted);font-size:11px}.ax line,.ax path{stroke:var(--axis)}
.ref{stroke:var(--text);stroke-width:1.2;stroke-dasharray:4 3}
.grid-sm{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.sm{border:1px solid var(--grid);border-radius:10px;padding:8px 10px 6px;background:var(--surface)}
.sm h3{margin:0;font-size:14px;display:flex;justify-content:space-between;gap:8px}
.sm h3 span{font-weight:400;font-size:12px;color:var(--text2)}
.sm .b90{fill:var(--band);fill-opacity:calc(var(--band90) * 2)}
.sm .b50{fill:var(--band);fill-opacity:calc(var(--band50) * 2)}
.sm .px{fill:none;stroke:var(--text);stroke-width:1.5}
.sm .hitpt{fill:var(--text);stroke:var(--surface);stroke-width:2}
.sm .misspt{fill:var(--s7);stroke:var(--surface);stroke-width:2}
.sm .hov:hover .b50,.sm .hov:hover .b90{fill:var(--s0)}
.tkey{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text2);margin:0 0 10px}
.tkey i{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:5px;background:var(--band)}
.tkey i.k90{opacity:.35}.tkey i.k50{opacity:.65}.tkey i.kpx{height:2px;background:var(--text);border-radius:0;vertical-align:3px}
.tkey i.kmiss{width:9px;height:9px;border-radius:50%;background:var(--s7)}
@media (max-width:760px){.tiles{grid-template-columns:repeat(2,1fr)}}
"""


def _tile(label: str, value: str, target: str, frac: float | None = None, goal: float | None = None) -> str:
    gauge = ""
    if frac is not None and math.isfinite(frac):
        g = f'<b style="left:calc({goal * 100:.1f}% - 1px)"></b>' if goal is not None else ""
        gauge = f'<div class="gauge" aria-hidden="true"><i style="width:{min(max(frac, 0), 1) * 100:.1f}%"></i>{g}</div>'
    return f'<div class="tile"><div class="l">{e(label)}</div><div class="v">{e(value)}</div><div class="t">{e(target)}</div>{gauge}</div>'


def _flag(v: float, goal: float, tol: float) -> str:
    """목표에서 tol 이상 벗어나면 강조 (색 + '!' 표시로 색만으로 구분하지 않음)."""
    if not math.isfinite(v):
        return "-"
    txt = pct(v, False, 0)
    return f'<span class="off">{txt} !</span>' if abs(v - goal) >= tol else txt


def _pit_svg(pits: np.ndarray, width: int = 520, height: int = 190) -> str:
    bins = np.histogram(np.clip(pits, 0, 0.999999), bins=10, range=(0, 1))[0]
    n = int(bins.sum())
    expect = n / 10
    top = max(bins.max(), expect) * 1.15 or 1
    ml, mr, mt, mb = 34, 8, 10, 26
    pw, ph = width - ml - mr, height - mt - mb
    bw = pw / 10
    y = lambda v: mt + ph - v / top * ph  # noqa: E731
    parts = [f'<svg class="pit" viewBox="0 0 {width} {height}" role="img" aria-label="PIT 분포 막대그래프" '
             f'style="width:100%;max-width:{width}px;height:auto">']
    ticks = sorted({0, int(round(expect)), int(bins.max())})
    parts.append('<g class="ax">')
    for t in ticks:
        parts.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke-opacity=".5"/>'
                     f'<text x="{ml - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for i in range(0, 11, 2):
        parts.append(f'<text x="{ml + i * bw:.1f}" y="{height - 8}" text-anchor="middle">{i / 10:.1f}</text>')
    parts.append("</g>")
    for i, c in enumerate(bins):
        x0 = ml + i * bw + 1
        h = ph * c / top
        r = min(4, h / 2, (bw - 2) / 2)
        x1, yb = x0 + bw - 2, mt + ph
        yt = yb - h
        # 위쪽 모서리만 둥글게 (기준선에 붙은 쪽은 각지게)
        d = (f"M{x0:.1f},{yb:.1f}V{yt + r:.1f}Q{x0:.1f},{yt:.1f} {x0 + r:.1f},{yt:.1f}H{x1 - r:.1f}"
             f"Q{x1:.1f},{yt:.1f} {x1:.1f},{yt + r:.1f}V{yb:.1f}Z") if c else ""
        parts.append(f'<g><title>PIT {i / 10:.1f}–{(i + 1) / 10:.1f}: {c}건 ({c / n:.0%})</title>'
                     f'<rect x="{x0 - 1:.1f}" y="{mt}" width="{bw:.1f}" height="{ph}" fill="transparent"/>'
                     + (f'<path class="bar" d="{d}"/>' if d else "") + "</g>")
    parts.append(f'<line class="ref" x1="{ml}" x2="{ml + pw}" y1="{y(expect):.1f}" y2="{y(expect):.1f}">'
                 f'<title>고르게 맞을 때 칸마다 {expect:.1f}건</title></line>')
    parts.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" stroke="var(--axis)"/></svg>')
    return "".join(parts)


def _symbol_svg(sym: str, ev: pd.DataFrame, prices: pd.Series | None, width: int = 260, height: int = 120) -> str:
    """그날 예측한 1주 뒤 범위(막대, 목표 날짜에 표시)와 실제 종가(선)."""
    rows = ev.sort_values("date")
    keep = sorted(rows["date"].unique())[-MAX_DATES:]  # 오래되면 최근 날짜만 (막대가 너무 가늘어지지 않게)
    rows = rows[rows["date"].isin(keep)]
    dates = pd.to_datetime(rows["date"])
    lo90 = (rows["lo90"] + 1) * rows["base_price"]
    hi90 = (rows["hi90"] + 1) * rows["base_price"]
    lo50 = (rows["lo50"] + 1) * rows["base_price"]
    hi50 = (rows["hi50"] + 1) * rows["base_price"]
    real = (rows["realized_return"] + 1) * rows["base_price"]
    start = pd.to_datetime(rows["as_of"]).min()
    end = dates.max()
    px = prices.loc[start:end] if prices is not None else pd.Series(dtype=float)
    vals = np.concatenate([lo90.values, hi90.values, px.values if len(px) else real.values])
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = (vmax - vmin) * 0.06 or vmax * 0.02
    vmin, vmax = vmin - pad, vmax + pad
    ml, mr, mt, mb = 6, 40, 6, 16
    pw, ph = width - ml - mr, height - mt - mb
    t0, t1 = start.value, max(end.value, start.value + 1)
    X = lambda d: ml + (pd.Timestamp(d).value - t0) / (t1 - t0) * pw  # noqa: E731
    Y = lambda v: mt + (vmax - v) / (vmax - vmin) * ph  # noqa: E731
    nd = max(len(set(dates)), 1)
    bw = max(2.0, min(8.0, pw / (nd * 1.8)))
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{e(sym)} 예측 범위와 실제 주가" '
           f'style="width:100%;height:auto">']
    for v in (vmin + pad, vmax - pad):
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{width - mr + 4}" y="{Y(v) + 4:.1f}" font-size="10" fill="var(--muted)">{v:,.0f}</text>')
    # 같은 목표 날짜에 기간만 다른 예측(1개월·6개월)이 겹치면 첫 것만 그림
    seen = set()
    for i, (_, r) in enumerate(rows.iterrows()):
        key = r["date"]
        if key in seen:
            continue
        seen.add(key)
        x = X(dates.iloc[i]) - bw / 2
        inside = "50%·90% 안" if r["in50"] else ("90% 안" if r["in90"] else "90% 밖")
        tip = (f"{r['as_of']} 예측 → {r['date']} ({r['checkpoint']})\n90% {lo90.iloc[i]:,.2f}–{hi90.iloc[i]:,.2f} · "
               f"50% {lo50.iloc[i]:,.2f}–{hi50.iloc[i]:,.2f}\n실제 {real.iloc[i]:,.2f} ({inside})")
        out.append(f'<g class="hov"><title>{e(tip)}</title>'
                   f'<rect class="b90" x="{x:.1f}" y="{Y(hi90.iloc[i]):.1f}" width="{bw:.1f}" '
                   f'height="{max(Y(lo90.iloc[i]) - Y(hi90.iloc[i]), 1):.1f}" rx="1.5"/>'
                   f'<rect class="b50" x="{x:.1f}" y="{Y(hi50.iloc[i]):.1f}" width="{bw:.1f}" '
                   f'height="{max(Y(lo50.iloc[i]) - Y(hi50.iloc[i]), 1):.1f}" rx="1.5"/></g>')
    if len(px) > 1:
        pts = " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in px.items())
        out.append(f'<polyline class="px" points="{pts}"/>')
    seen = set()
    for i, (_, r) in enumerate(rows.iterrows()):
        if r["date"] in seen:
            continue
        seen.add(r["date"])
        cls = "hitpt" if r["in90"] else "misspt"
        rad = 3.2 if r["in90"] else 4.2
        out.append(f'<circle class="{cls}" cx="{X(dates.iloc[i]):.1f}" cy="{Y(real.iloc[i]):.1f}" r="{rad}" pointer-events="none"/>')
    out.append(f'<text x="{ml}" y="{height - 3}" font-size="10" fill="var(--muted)">{start:%m-%d}</text>'
               f'<text x="{ml + pw}" y="{height - 3}" font-size="10" fill="var(--muted)" text-anchor="end">{end:%m-%d}</text>')
    out.append("</svg>")
    return "".join(out)


def render_track_page(track: TrackResult, generated: datetime | None = None, back: str = "index.html") -> str:
    generated = generated or datetime.now()
    s = track.summary
    ev = track.evaluations
    head = (f'<header class="top"><p class="note"><a href="{e(back)}">← 처음으로</a></p>'
            '<h1>실전 성과 <span class="h-sub">기록해 둔 과거 예측 vs 실제 주가</span></h1>'
            f'<p class="meta">생성 {generated:%Y-%m-%d %H:%M} · 예측 기록 {s["records"]}건 · 평가 {s["evaluated"]}개 · '
            f'대기 {s["pending"]}개</p></header>')
    if not s.get("evaluated"):
        body = ('<section class="card"><p>아직 날짜가 지난 체크포인트가 없습니다. 예측 후 1주가 지나면 '
                '실제 주가와 비교한 결과가 여기에 쌓입니다.</p></section>')
        return _page(head + body)

    nd = s.get("target_dates", 0)
    # 같은 종목·기준일·날짜를 여러 기간으로 예측한 평가는 한 번만 (요약·PIT·종목별 표용)
    uniq = ev.sort_values("horizon").drop_duplicates(["symbol", "as_of", "date"])
    s = dict(s, coverage50=float(uniq["in50"].mean()), coverage90=float(uniq["in90"].mean()),
             mean_pit=float(uniq["pit"].mean()))
    tiles = "".join([
        _tile("50% 구간 적중률", pct(s["coverage50"], False, 1), "목표 50% — 실제가 예측 범위 가운데 절반 안에 든 비율",
              s["coverage50"], 0.5),
        _tile("90% 구간 적중률", pct(s["coverage90"], False, 1), "목표 90% — 넓은 범위 안에 든 비율", s["coverage90"], 0.9),
        _tile("평균 PIT", f"{s['mean_pit']:.2f}", "목표 0.50 — 높으면 실제가 예측보다 위쪽", s["mean_pit"], 0.5),
        _tile("서로 다른 평가 날짜", f"{nd}일", f"중복 제외 평가 {len(uniq)}개가 이 날짜들에 몰려 있음"),
    ])
    caution = ""
    if nd < 20:
        caution = (f'<p class="note">※ 평가가 {nd}개 날짜에만 몰려 있습니다. 같은 날 20종목이 함께 움직이므로 '
                   '실질 표본은 날짜 수에 가깝고, 최소 몇 주는 쌓여야 의미 있는 판단이 가능합니다.</p>')

    bd = breakdown(ev)
    brow = "".join(
        f'<tr><td class="l">{r.horizon}거래일</td><td class="l">{e(r.checkpoint)}</td><td>{r.n}</td><td>{r.dates}</td>'
        f'<td>{_flag(r.cov50, 0.5, 0.15 if r.dates >= 5 else 9)}</td><td>{_flag(r.cov90, 0.9, 0.10 if r.dates >= 5 else 9)}</td><td>{r.pit:.2f}</td>'
        f'<td>{pct(r.hit, False, 0) if r.hit == r.hit else "-"}</td></tr>' for r in bd.itertuples())
    btable = ('<div class="scroll"><table class="tr"><thead><tr><th class="l">예측 기간</th><th class="l">체크포인트</th>'
              '<th>평가 수</th><th>날짜 수</th><th>50% 적중</th><th>90% 적중</th><th>평균 PIT</th><th>방향 적중(만기)</th>'
              f'</tr></thead><tbody>{brow}</tbody></table></div>'
              '<p class="note">"!" = 목표에서 크게 벗어남 (50% 구간 ±15%p, 90% 구간 −10%p 이상). 평가 날짜가 5일 미만인 줄은 표시하지 않습니다.</p>')

    sym = by_symbol(uniq)
    srow = "".join(
        f'<tr><td class="l">{e(r.symbol)}</td><td>{r.n}</td><td>{pct(r.cov50, False, 0)}</td>'
        f'<td>{pct(r.cov90, False, 0)}</td><td>{r.pit:.2f}</td><td>{e(str(r.last))}</td></tr>' for r in sym.itertuples())
    stable = ('<div class="scroll"><table class="tr"><thead><tr><th class="l">종목</th><th>평가 수</th><th>50% 적중</th>'
              f'<th>90% 적중</th><th>평균 PIT</th><th>마지막 평가일</th></tr></thead><tbody>{srow}</tbody></table></div>')

    # 종목별 차트: 가장 짧은 체크포인트(보통 1주)만
    ev = ev.assign(_days=(pd.to_datetime(ev["date"]) - pd.to_datetime(ev["as_of"])).dt.days)
    first_cp = ev.groupby("checkpoint")["_days"].median().idxmin()
    sub = ev[ev["checkpoint"] == first_cp].sort_values("horizon")
    cards = []
    for symbol, g in sub.groupby("symbol"):
        n90 = g.drop_duplicates("date")["in90"]
        cards.append(f'<div class="sm"><h3>{e(symbol)}<span>90% 안 {int(n90.sum())}/{len(n90)}</span></h3>'
                     f'{_symbol_svg(symbol, g, track.prices.get(symbol))}</div>')
    key = ('<div class="tkey"><span><i class="k90"></i>90% 범위</span><span><i class="k50"></i>50% 범위</span>'
           '<span><i class="kpx"></i>실제 종가</span><span><i class="kmiss"></i>90% 범위 밖</span></div>')

    body = (f'<section class="card"><h2>요약 <span class="h-sub">1개월·6개월 예측이 같은 날짜를 평가한 경우 한 번만 셈</span></h2><div class="tiles">{tiles}</div>{caution}</section>'
            f'<section class="card"><h2>예측 기간 × 체크포인트별 <span class="h-sub">1개월·6개월 예측의 1주 평가가 겹치지 않게 나눠 봄</span></h2>{btable}</section>'
            '<section class="card"><h2>PIT 분포 <span class="h-sub">실제 주가가 예측 분포의 어디쯤 왔는지 (0 = 최하단, 1 = 최상단)</span></h2>'
            f'{_pit_svg(uniq["pit"].to_numpy())}'
            '<p class="note">점선 = 고르게 맞을 때의 칸당 건수. 잘 보정된 예측이면 막대 높이가 점선 근처에 고르게 모입니다. 오른쪽에 몰리면 실제가 예측보다 높게, '
            '왼쪽이면 낮게, 양 끝에 몰리면 범위가 너무 좁았다는 뜻입니다.</p></section>'
            f'<section class="card"><h2>종목별 예측 vs 실제 <span class="h-sub">{e(first_cp)} 뒤 범위 (막대를 가리키면 상세)</span></h2>'
            f'{key}<div class="grid-sm">{"".join(cards)}</div></section>'
            f'<section class="card"><h2>종목별 요약</h2>{stable}</section>')
    return _page(head + body)


def _page(body: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>실전 성과</title>
<style>{REPORT_CSS}{DASH_CSS}{TRACK_CSS}</style>
</head>
<body>
<main class="wrap">
{body}
<footer class="foot">{e(DISCLAIMER)}</footer>
</main>
</body>
</html>
"""
