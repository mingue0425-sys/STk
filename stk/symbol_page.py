"""종목별 예측 변화 기록 페이지 (site/symbols/<종목>.html, site/symbols/index.html).

예측 기록(forecasts.jsonl)만으로 만듭니다:
- 상승확률 추이: 기준일마다 예측 기간(1개월·6개월)별 상승확률 — 판단이 언제 바뀌었는지
- 지난 예측 vs 실제: 그날 예측한 1주 뒤 범위(막대)와 실제 종가(선) — track_page 와 같은 그림
- 날짜별 표: 종가 · 기간별 흐름 판정 · 상승확률 · 기대수익 · 국면 (판정·국면이 바뀐 날 표시)

기준일은 실제 기록된 날짜만 씁니다 (데이터 지연으로 빠진 날은 건너뜀).
"""

from __future__ import annotations

import html
import math
from collections import defaultdict
from datetime import datetime

import pandas as pd

from .forecast import horizon_label
from .html_report import CSS as REPORT_CSS
from .report import DISCLAIMER, pct
from .track_page import TRACK_CSS, _symbol_svg
from .tracking import TrackResult
from .watch import DASH_CSS

e = html.escape
SERIES = ("var(--s0)", "var(--s1)", "var(--s2)", "var(--s3)")   # 기간 순서대로 고정 배정

SYMBOL_CSS = r"""
.wrap{max-width:1080px}
.minw{min-width:560px;max-width:720px}  /* 휴대폰에서 글자가 너무 작아지지 않게: 좁으면 가로 스크롤 */
.pl{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.pt{stroke:var(--surface);stroke-width:2}
.hv:hover .pt{r:5}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text2);margin:0 0 6px}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:3px;margin-right:6px;border-radius:1px}
.chg{display:inline-block;margin-left:4px;padding:0 5px;border:1px solid var(--axis);border-radius:999px;font-size:11px;color:var(--text2)}
.up{color:var(--s0)}.down{color:var(--s7)}
ul.syms{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;padding:0;list-style:none}
ul.syms a{display:block;padding:8px 10px;border:1px solid var(--grid);border-radius:8px;background:var(--surface);
color:var(--text);text-decoration:none;font-weight:600}
ul.syms a span{display:block;font-weight:400;font-size:12px;color:var(--text2)}
ul.syms a:hover{border-color:var(--text2)}
"""


def _by_symbol(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("symbol") and r.get("as_of"):
            out[r["symbol"]].append(r)
    return out


def _horizons(recs: list[dict]) -> list[int]:
    """최근 기준일에 쓰인 예측 기간만 (예전 버전의 3개월 기록 등은 표에서만 보임)."""
    last = max(r["as_of"] for r in recs)
    hs = sorted({int(r["horizon"]) for r in recs if r["as_of"] == last})
    return hs or sorted({int(r["horizon"]) for r in recs})


def _prob_svg(recs: list[dict], horizons: list[int], width: int = 680, height: int = 220) -> str:
    dates = sorted({r["as_of"] for r in recs if int(r["horizon"]) in horizons})
    if not dates:
        return ""
    ml, mr, mt, mb = 40, 70, 12, 24
    pw, ph = width - ml - mr, height - mt - mb
    ts = [pd.Timestamp(d) for d in dates]
    t0, t1 = ts[0].value, max(ts[-1].value, ts[0].value + 1)
    X = (lambda d: ml + (pd.Timestamp(d).value - t0) / (t1 - t0) * pw) if len(ts) > 1 else (lambda d: ml + pw / 2)  # noqa: E731
    probs = [float(r["prob_up"]) for r in recs if int(r["horizon"]) in horizons and r.get("prob_up") is not None]
    lo = min(0.3, math.floor(min(probs) * 10) / 10)
    hi = max(0.7, math.ceil(max(probs) * 10) / 10)
    Y = lambda p: mt + (hi - p) / (hi - lo) * ph  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="기준일별 상승확률 추이" '
           f'style="width:100%;max-width:{width}px;height:auto">']
    p = lo
    while p <= hi + 1e-9:
        strong = abs(p - 0.5) < 1e-9
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(p):.1f}" y2="{Y(p):.1f}" stroke="var(--{"axis" if strong else "grid"})"'
                   + (' stroke-dasharray="4 3"' if strong else "") + "/>"
                   f'<text x="{ml - 6}" y="{Y(p) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{p:.0%}</text>')
        p += 0.1
    labels = [(dates[0], "start"), (dates[-1], "end")] if len(dates) > 1 else [(dates[0], "middle")]
    for d, anchor in labels:
        out.append(f'<text x="{X(d):.1f}" y="{height - 6}" text-anchor="{anchor}" font-size="11" fill="var(--muted)">{d[5:]}</text>')
    ends = []
    for i, h in enumerate(horizons):
        col = SERIES[i % len(SERIES)]
        pts = sorted((r["as_of"], float(r["prob_up"])) for r in recs
                     if int(r["horizon"]) == h and r.get("prob_up") is not None)
        if not pts:
            continue
        if len(pts) > 1:
            out.append(f'<polyline class="pl" stroke="{col}" points="'
                       + " ".join(f"{X(d):.1f},{Y(v):.1f}" for d, v in pts) + '"/>')
        for d, v in pts:
            out.append(f'<g class="hv"><title>{e(horizon_label(h))} 예측 · {d} · 상승확률 {v:.0%}</title>'
                       f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="9" fill="transparent"/>'
                       f'<circle class="pt" cx="{X(d):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}"/></g>')
        ends.append([Y(pts[-1][1]), f"{horizon_label(h)} {pts[-1][1]:.0%}"])
    # 끝 라벨이 겹치지 않게 최소 간격 확보
    ends.sort()
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 13)
    for y, txt in ends:
        out.append(f'<text x="{ml + pw + 8}" y="{y + 4:.1f}" font-size="11" fill="var(--text2)">{e(txt)}</text>')
    out.append("</svg>")
    return "".join(out)


def _table(recs: list[dict], horizons_all: list[int]) -> str:
    by_day: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in recs:
        by_day[r["as_of"]][int(r["horizon"])] = r
    days = sorted(by_day)
    prev: dict[int, dict] = {}
    rows = []
    for d in days:
        cells = []
        price = next((r.get("price") for r in by_day[d].values() if r.get("price")), None)
        for h in horizons_all:
            r = by_day[d].get(h)
            if r is None:
                cells.append('<td class="muted">-</td><td class="muted">-</td>')
                continue
            p = prev.get(h)
            flip = p is not None and p.get("direction", p.get("outlook")) != r.get("direction", r.get("outlook"))
            pu = r.get("prob_up")
            er = r.get("expected_return")
            cells.append(
                f'<td class="l">{e(r.get("outlook") or "-")}' + ('<span class="chg">판정 변화</span>' if flip else "") + "</td>"
                f'<td>{pct(pu, False, 0) if pu is not None else "-"}'
                + (f' <span class="muted">({pct(er, True, 1)})</span>' if er is not None else "") + "</td>")
            prev[h] = r
        if not by_day[d]:
            continue
        regime_rec = by_day[d].get(horizons_all[0]) or next(iter(by_day[d].values()))
        regime = regime_rec.get("regime") or "-"
        rows.append((d, price, "".join(cells), regime))
    body = []
    last_regime = None
    for d, price, cells, regime in rows:
        rchg = last_regime is not None and regime != last_regime
        body.append(f'<tr><td class="l">{d}</td><td>{f"{price:,.2f}" if price else "-"}</td>{cells}'
                    f'<td class="l">{e(regime)}' + ('<span class="chg">국면 변화</span>' if rchg else "") + "</td></tr>")
        last_regime = regime
    head = "".join(f'<th class="l">{e(horizon_label(h))} 판정</th><th>상승확률 (기대수익)</th>' for h in horizons_all)
    return ('<div class="scroll"><table class="tr"><thead><tr><th class="l">기준일</th><th>종가</th>'
            f'{head}<th class="l">국면</th></tr></thead><tbody>{"".join(reversed(body))}</tbody></table></div>')


def render_symbol_page(symbol: str, recs: list[dict], track: TrackResult | None,
                       generated: datetime | None = None) -> str:
    generated = generated or datetime.now()
    recs = sorted(recs, key=lambda r: (r["as_of"], int(r["horizon"])))
    hs = _horizons(recs)
    last = recs[-1]["as_of"]
    current = [r for r in recs if int(r["horizon"]) in hs]
    days = sorted({r["as_of"] for r in recs})
    legend = "".join(f'<span><i style="background:{SERIES[i % len(SERIES)]}"></i>{e(horizon_label(h))} 예측</span>'
                     for i, h in enumerate(hs))
    prob = (f'<section class="card"><h2>상승확률 추이 <span class="h-sub">기준일마다 예측한 만기 시점 상승확률 '
            '(점선 = 50%)</span></h2>'
            f'<div class="legend">{legend}</div><div class="scroll"><div class="minw">{_prob_svg(recs, hs)}</div></div></section>')
    band = ""
    if track is not None and not track.evaluations.empty:
        ev = track.evaluations[track.evaluations["symbol"] == symbol]
        if len(ev):
            ev = ev.assign(_days=(pd.to_datetime(ev["date"]) - pd.to_datetime(ev["as_of"])).dt.days)
            first_cp = ev.groupby("checkpoint")["_days"].median().idxmin()
            sub = ev[ev["checkpoint"] == first_cp].sort_values("horizon")
            n90 = sub.drop_duplicates("date")["in90"]
            band = (f'<section class="card"><h2>지난 예측 vs 실제 <span class="h-sub">그날 예측한 {e(first_cp)} 뒤 범위와 실제 종가 · '
                    f'90% 범위 안 {int(n90.sum())}/{len(n90)}</span></h2>'
                    '<div class="tkey"><span><i class="k90"></i>90% 범위</span><span><i class="k50"></i>50% 범위</span>'
                    '<span><i class="kpx"></i>실제 종가</span><span><i class="kmiss"></i>90% 범위 밖</span></div>'
                    f'<div class="scroll"><div class="sm big minw">{_symbol_svg(symbol, sub, track.prices.get(symbol), width=680, height=220)}</div></div></section>')
    if not band:
        band = ('<section class="card"><h2>지난 예측 vs 실제</h2><p class="note">아직 평가할 날짜가 지나지 않았습니다 '
                '(예측 후 1주 뒤부터 표시).</p></section>')
    body = (f'<header class="top"><p class="note"><a href="index.html">← 종목 목록</a> · <a href="../index.html">처음으로</a> · '
            '<a href="../track.html">실전 성과</a></p>'
            f'<h1>{e(symbol)} <span class="h-sub">예측 변화 기록</span></h1>'
            f'<p class="meta">기록 {len(days)}일 ({days[0]} ~ {last}) · 생성 {generated:%Y-%m-%d %H:%M}</p></header>'
            + prob + band
            + f'<section class="card"><h2>날짜별 예측 <span class="h-sub">최근 순 · 괄호 = 기대수익</span></h2>{_table(current, hs)}</section>')
    return _page(f"{symbol} 예측 기록", body)


def render_symbol_index(by_sym: dict[str, list[dict]], generated: datetime | None = None) -> str:
    generated = generated or datetime.now()
    items = []
    for sym in sorted(by_sym):
        recs = by_sym[sym]
        last = max(r["as_of"] for r in recs)
        latest = sorted((r for r in recs if r["as_of"] == last), key=lambda r: int(r["horizon"]))
        txt = " · ".join(f"{horizon_label(int(r['horizon']))} {r['prob_up']:.0%}" for r in latest if r.get("prob_up") is not None)
        items.append(f'<li><a href="{e(sym)}.html">{e(sym)}<span>{e(txt)} 상승</span></a></li>')
    body = ('<header class="top"><p class="note"><a href="../index.html">← 처음으로</a></p>'
            '<h1>종목별 예측 기록 <span class="h-sub">날짜마다 예측이 어떻게 바뀌었는지</span></h1>'
            f'<p class="meta">생성 {generated:%Y-%m-%d %H:%M}</p></header>'
            f'<section class="card"><ul class="syms">{"".join(items)}</ul></section>')
    return _page("종목별 예측 기록", body)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<style>{REPORT_CSS}{DASH_CSS}{TRACK_CSS}{SYMBOL_CSS}</style>
</head>
<body>
<main class="wrap">
{body}
<footer class="foot">{e(DISCLAIMER)}</footer>
</main>
</body>
</html>
"""


def write_symbol_pages(site, records: list[dict], track: TrackResult | None, symbols: list[str] | None = None,
                       generated: datetime | None = None) -> int:
    """기록이 있는 종목마다 페이지를 쓰고 목록 페이지를 만듭니다. symbols 를 주면 그 종목만 (관심종목에서 빠진 종목 제외)."""
    from pathlib import Path

    by_sym = _by_symbol(records)
    if symbols is not None:
        by_sym = {s: v for s, v in by_sym.items() if s in set(symbols)}
    out = Path(site) / "symbols"
    out.mkdir(parents=True, exist_ok=True)
    for sym, recs in by_sym.items():
        (out / f"{sym}.html").write_text(render_symbol_page(sym, recs, track, generated), encoding="utf-8")
    (out / "index.html").write_text(render_symbol_index(by_sym, generated), encoding="utf-8")
    return len(by_sym)
