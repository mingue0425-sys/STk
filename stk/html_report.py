"""대화형 HTML 리포트 (단일 파일, 외부 의존성 없음 · 오프라인 동작).

- 팬 차트: 마우스를 올리면 날짜별 시나리오 가격·분포 구간이 표시되고, 범례로 계열을 켜고 끌 수 있습니다.
- 국면 흐름: 기간별 국면(저변동/중간/고변동) 확률 100% 누적 막대
- 만기 수익률 분포, 시나리오/기간별/진단/시장 환경 표
- 라이트/다크 모드 자동 전환
"""

from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd

from .forecast import QUANTILES, Forecast
from .report import DISCLAIMER, drift_text, market_lines, model_text, money, pct, regime_summary_line, trend_note

HISTORY_DAYS = 252


def _e(text) -> str:
    return html.escape(str(text), quote=True)


def _r(values, nd: int = 2) -> list:
    return [None if not math.isfinite(float(v)) else round(float(v), nd) for v in values]


def _payload(fc: Forecast, prices) -> dict:
    series = prices["adj_close"] if hasattr(prices, "columns") else prices
    hist = series.dropna().iloc[-HISTORY_DAYS:]
    s0 = fc.s0
    n_sc = len(fc.results)
    lines = [
        {
            "key": r.scenario.key,
            "name": r.scenario.name,
            "slot": r.scenario.color_slot % 8,
            "dash": "",
            "values": _r(r.quantiles[0.5]),
            "note": f"{r.scenario.probability:.0%}",
        }
        for r in fc.results
    ]
    if fc.regime is not None:
        lines.append(
            {
                "key": "regime",
                "name": "국면 모델",
                "slot": n_sc % 8,
                "dash": "2 3",
                "values": _r(fc.regime.quantiles[0.5]),
                "note": f"비중 {fc.regime_weight:.0%}" if fc.regime_weight > 0 else "참고",
            }
        )

    # 만기 수익률 분포 (확률 가중 히스토그램)
    groups = [(r.terminal, r.scenario.probability * (1 - fc.regime_weight), r.scenario.name,
               r.scenario.color_slot % 8, "") for r in fc.results]
    if fc.regime is not None and fc.regime_weight > 0:
        groups.append((fc.regime.terminal, fc.regime_weight, "국면 모델", n_sc % 8, "2 3"))
    all_ret = np.concatenate([g[0] / s0 - 1 for g in groups])
    lo, hi = np.quantile(all_ret, [0.005, 0.995])
    bins = np.linspace(lo, hi, 61)
    dist_groups = []
    for terminal, weight, name, slot, dash in groups:
        h, _ = np.histogram(terminal / s0 - 1, bins=bins)
        dist_groups.append({"name": name, "slot": slot, "dash": dash,
                            "mass": _r(h / len(terminal) * weight, 5)})

    regime_flow = None
    if fc.regime is not None:
        st = fc.regime.state
        regime_flow = {
            "names": st.names,
            "rows": [{"label": r["기간"], "probs": [round(r[n], 4) for n in st.names]} for r in fc.regime_flow()],
        }

    return {
        "symbol": fc.symbol,
        "s0": s0,
        "hist": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index], "prices": _r(hist.to_numpy())},
        "fut": {
            "dates": [d.strftime("%Y-%m-%d") for d in fc.dates],
            "bands": {f"p{int(q * 100)}": _r(fc.mix_quantiles[q]) for q in QUANTILES},
            "lines": lines,
        },
        "events": [d.strftime("%Y-%m-%d") for d, _ in fc.events],
        "dist": {"bins": _r(bins, 4), "groups": dist_groups},
        "scenarios": [
            {"name": r.scenario.name, "slot": r.scenario.color_slot % 8, "p": round(r.scenario.probability, 4),
             "desc": r.scenario.description}
            for r in fc.results
        ],
        "regimeFlow": regime_flow,
    }


def _kpis(fc: Forecast) -> str:
    ol = fc.outlook
    tiles = [
        ("상승 마감 확률", pct(ol.prob_up, False, 0), f"{fc.dates[-1]:%Y-%m-%d} 기준"),
        ("기대수익률", pct(ol.expected_return), f"중앙값 {pct(ol.median_return)}"),
        ("90% 가격 범위", f"{money(ol.band90[0])} ~ {money(ol.band90[1])}", ol.volatility),
        ("기간 중 ±20% 도달", f"{pct(fc.mix_prob_hit_up, False, 0)} / {pct(fc.mix_prob_hit_down, False, 0)}",
         "+20% 이상 / −20% 이하"),
    ]
    return "".join(
        f'<div class="kpi"><div class="kpi-label">{_e(a)}</div><div class="kpi-value">{_e(b)}</div>'
        f'<div class="kpi-sub">{_e(c)}</div></div>'
        for a, b, c in tiles
    )


def _table(headers: list[str], rows: list[list[str]], num_from: int = 1, swatches: list[int | None] | None = None) -> str:
    head = "".join(f'<th class="{"num" if i >= num_from else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = []
    for ri, row in enumerate(rows):
        cells = []
        for i, c in enumerate(row):
            content = _e(c)
            if i == 0 and swatches and swatches[ri] is not None:
                content = f'<span class="sw s{swatches[ri]}"></span>{content}'
            cells.append(f'<td class="{"num" if i >= num_from else ""}">{content}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tbl-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _scenario_table(fc: Forecast) -> str:
    s0 = fc.s0
    rows, sw = [], []
    for r in fc.results:
        sc = r.scenario
        rows.append([sc.name, pct(sc.probability, False, 0), pct(sc.avg_drift + sc.jump_drag),
                     f"{money(r.terminal_mean)} ({pct(r.terminal_mean / s0 - 1, digits=0)})",
                     money(r.terminal_quantiles[0.5]),
                     f"{money(r.terminal_quantiles[0.05])} ~ {money(r.terminal_quantiles[0.95])}",
                     pct(r.prob_up, False, 0), pct(r.median_max_drawdown, digits=0)])
        sw.append(sc.color_slot % 8)
    if fc.regime is not None:
        rg = fc.regime
        rows.append(["국면 모델(HMM)", pct(rg.weight, False, 0) if rg.weight > 0 else "참고",
                     pct(float(rg.state.current @ rg.state.drift)),
                     f"{money(rg.terminal_mean)} ({pct(rg.terminal_mean / s0 - 1, digits=0)})",
                     money(rg.terminal_quantiles[0.5]),
                     f"{money(rg.terminal_quantiles[0.05])} ~ {money(rg.terminal_quantiles[0.95])}",
                     pct(rg.prob_up, False, 0), pct(rg.median_max_drawdown, digits=0)])
        sw.append(len(fc.results) % 8)
    rows.append(["확률가중 전체", "100%", "", f"{money(fc.mix_expected_price)} ({pct(fc.mix_expected_price / s0 - 1, digits=0)})",
                 money(fc.mix_terminal_quantiles[0.5]),
                 f"{money(fc.mix_terminal_quantiles[0.05])} ~ {money(fc.mix_terminal_quantiles[0.95])}",
                 pct(fc.mix_prob_up, False, 0), ""])
    sw.append(None)
    note = ""
    if fc.regime is not None and 0 < fc.regime_weight < 1:
        note = (f'<p class="note">시나리오 확률은 시나리오 모델 내부 비중입니다. 전체 분포 = 시나리오 모델 '
                f"{1 - fc.regime_weight:.0%} + 국면 모델 {fc.regime_weight:.0%}</p>")
    return _table(["시나리오", "확률", "연 기대수익률", "기대가격", "중앙값", "5%~95% 범위", "상승확률", "중앙 MDD"],
                  rows, swatches=sw) + note


def _checkpoint_table(fc: Forecast) -> str:
    tbl = fc.checkpoint_table()
    names = [r.scenario.name for r in fc.results] + (["국면 모델"] if fc.regime is not None else [])
    rows = []
    for _, row in tbl.iterrows():
        rows.append([row["기간"], row["날짜"], *[money(row[n]) for n in names],
                     f"{money(row['전체 p50'])} ({pct(row['전체 p50'] / fc.s0 - 1, digits=0)})",
                     f"{money(row['전체 p25'])} ~ {money(row['전체 p75'])}"])
    return _table(["기간", "날짜", *names, "전체 중앙값", "전체 25~75%"], rows, num_from=2)


def _diagnostics(fc: Forecast) -> str:
    st, cb = fc.stats, fc.calib
    rows = [
        ["추세 점수", f"{st.trend_score:+.2f} ({st.trend_label}, {trend_note(fc)})"],
        ["이동평균 20 / 50 / 200일", f"{money(st.ma20)} / {money(st.ma50)} / {money(st.ma200)}"],
        ["RSI(14)", f"{st.rsi14:.0f}"],
        ["1년 수익률 / 12-1 모멘텀", f"{pct(st.return_1y)} / {pct(st.momentum_12_1)}"],
        ["52주 고점 대비", pct(st.drawdown_now)],
        ["변동성 현재(EWMA) / 1년 / 장기", f"{pct(st.vol_now, False)} / {pct(st.vol_1y, False)} / {pct(st.vol_long, False)}"],
        ["베타 (vs S&P500)", f"{st.beta:.2f}" if st.beta is not None else "N/A"],
        ["꼬리 두께 (t 자유도)", f"{st.tail_df:.1f}"],
        ["기본 기대수익률", drift_text(fc)],
        ["시나리오 폭", f"±{pct(cb.spread, False)}"],
    ]
    return _table(["지표", "값"], rows, num_from=9)


def _regime_table(fc: Forecast) -> str:
    st = fc.regime.state
    rows = [[st.names[i], pct(st.drift[i]), pct(st.vol[i], False, 0), f"{st.duration[i]:.0f}일",
             pct(st.stationary[i], False, 0), pct(st.current[i], False, 0)] for i in range(st.params.k)]
    return _table(["국면", "연 기대수익률", "연 변동성", "평균 지속", "장기 비중", "현재 확률"], rows,
                  swatches=[f"r{i}" for i in range(st.params.k)])  # type: ignore[list-item]


def render_html(fc: Forecast, prices) -> str:
    data = json.dumps(_payload(fc, prices), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    ol = fc.outlook
    events = "".join(f"<li>{d:%Y-%m-%d} — 하루 충격 ±{s:.1%}</li>" for d, s in fc.events)
    market = "".join(f"<li>{_e(line.strip())}</li>" for line in market_lines(fc) if line.strip())
    regime_block = ""
    if fc.regime is not None:
        regime_block = f"""
<section class="card">
  <h2>국면 흐름 <span class="h-sub">은닉 마르코프 모델 · 데이터 기반</span></h2>
  <p class="lead">{_e(regime_summary_line(fc).strip())}</p>
  <div id="regime-flow" class="flow" role="img" aria-label="기간별 국면 확률"></div>
  <div id="regime-legend" class="legend static"></div>
  {_regime_table(fc)}
  <p class="note">국면별 기대수익률은 표본 차이를 추정 정밀도만큼 축소한 뒤 기본 기대수익률에 맞춘 값입니다.</p>
</section>"""
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(fc.symbol)} 주가 흐름 예측</title>
<style>{CSS}</style>
</head>
<body>
<main class="wrap">
<header class="top">
  <div>
    <h1>{_e(fc.symbol)} <span class="h-sub">시나리오별 주가 흐름 예측</span></h1>
    <p class="meta">기준일 {fc.stats.last_date:%Y-%m-%d} · 현재가 {_e(money(fc.s0))} · {fc.horizon}거래일 예측 (~{fc.dates[-1]:%Y-%m-%d}) · {_e(model_text(fc))}</p>
  </div>
</header>

<section class="card hero">
  <div class="verdict">
    <div class="verdict-label">전체 흐름</div>
    <div class="verdict-value">{_e(ol.label)}</div>
    <div class="verdict-sub">{_e(ol.shape)} · 최유력 시나리오: {_e(ol.dominant)}</div>
  </div>
  <div class="kpis">{_kpis(fc)}</div>
</section>

<section class="card">
  <h2>가격 경로 <span class="h-sub">최근 1년 실제 주가 + 예측 분포</span></h2>
  <div id="fan-legend" class="legend" role="group" aria-label="계열 표시 전환"></div>
  <div class="chart-wrap" id="fan-wrap">
    <svg id="fan" class="chart" tabindex="0" role="img" aria-label="{_e(fc.symbol)} 가격 경로 팬 차트"></svg>
    <div class="tip" id="fan-tip" hidden></div>
  </div>
  <p class="note">음영: 전체(확률가중) 분포의 50%·90% 구간 · 선: 각 시나리오의 중앙값 경로. 차트 위에서 마우스(또는 ←/→ 키)로 날짜별 값을 확인하세요.</p>
</section>

<div class="grid2">
<section class="card">
  <h2>시나리오 확률</h2>
  <div id="probs" class="probs"></div>
</section>
<section class="card">
  <h2>만기 수익률 분포 <span class="h-sub">확률 가중</span></h2>
  <div class="chart-wrap" id="dist-wrap">
    <svg id="dist" class="chart" role="img" aria-label="만기 수익률 분포"></svg>
    <div class="tip" id="dist-tip" hidden></div>
  </div>
</section>
</div>

<section class="card">
  <h2>시나리오별 결과</h2>
  {_scenario_table(fc)}
</section>

<section class="card">
  <h2>기간별 흐름 <span class="h-sub">중앙값 가격 경로</span></h2>
  {_checkpoint_table(fc)}
</section>
{regime_block}
<div class="grid2">
<section class="card">
  <h2>현재 상태 진단</h2>
  {_diagnostics(fc)}
</section>
<section class="card">
  <h2>시장 환경</h2>
  <ul class="plain">{market}</ul>
  {"<h3>이벤트</h3><ul class='plain'>" + events + "</ul>" if events else ""}
</section>
</div>

<footer class="foot">{_e(DISCLAIMER)}</footer>
</main>
<script id="data" type="application/json">{data}</script>
<script>{JS}</script>
</body>
</html>
"""


CSS = r"""
:root{color-scheme:light;
--page:#f4f3ef;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--muted:#7a7974;--grid:#e4e3df;--axis:#cfcec9;
--band:#8a8984;--band90:.16;--band50:.30;
--s0:#2a78d6;--s1:#eb6834;--s2:#1baf7a;--s3:#eda100;--s4:#e87ba4;--s5:#008300;--s6:#6250d6;--s7:#e34948;
--r0:#86b6ef;--r1:#2a78d6;--r2:#104281;--r0t:#0b0b0b;--r1t:#ffffff;--r2t:#ffffff;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#121211;--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--muted:#97968e;--grid:#2e2e2c;--axis:#3d3d3a;
--band:#a3a29b;--band90:.14;--band50:.26;
--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181;--s5:#008300;--s6:#9085e9;--s7:#e66767;
--r0:#9ec5f4;--r1:#3987e5;--r2:#184f95;--r0t:#0b0b0b;--r1t:#0b0b0b;--r2t:#ffffff;}}
:root[data-theme="dark"]{color-scheme:dark;
--page:#121211;--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--muted:#97968e;--grid:#2e2e2c;--axis:#3d3d3a;
--band:#a3a29b;--band90:.14;--band50:.26;
--s0:#3987e5;--s1:#d95926;--s2:#199e70;--s3:#c98500;--s4:#d55181;--s5:#008300;--s6:#9085e9;--s7:#e66767;
--r0:#9ec5f4;--r1:#3987e5;--r2:#184f95;--r0t:#0b0b0b;--r1t:#0b0b0b;--r2t:#ffffff;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);
font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,-apple-system,"Segoe UI",sans-serif;
font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:26px;margin:0;letter-spacing:-.01em}
h2{font-size:16px;margin:0 0 12px}
h3{font-size:14px;margin:16px 0 6px}
.h-sub{font-size:.62em;font-weight:500;color:var(--text2);margin-left:6px}
h2 .h-sub{font-size:12px}
.meta{margin:4px 0 16px;color:var(--text2);font-size:13px}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:18px 18px 14px;margin:0 0 16px}
.hero{display:grid;grid-template-columns:minmax(220px,1fr) 2.4fr;gap:18px;align-items:center}
.verdict-label,.kpi-label{font-size:12px;color:var(--text2)}
.verdict-value{font-size:24px;font-weight:700;margin:2px 0 4px;letter-spacing:-.01em}
.verdict-sub{color:var(--text2);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi{border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.kpi-value{font-size:18px;font-weight:650;margin:2px 0;font-variant-numeric:tabular-nums;line-height:1.3}
.kpi-sub{font-size:11.5px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid2>.card{margin:0 0 16px}
.card,.hero>*,.grid2>*,.kpi{min-width:0}
.legend{display:flex;flex-wrap:wrap;gap:6px 8px;margin:0 0 10px}
.legend button,.legend span.item{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--grid);background:transparent;
color:var(--text);border-radius:999px;padding:3px 10px;font:inherit;font-size:12.5px;cursor:pointer}
.legend span.item{cursor:default}
.legend button[aria-pressed="false"]{color:var(--muted);opacity:.55}
.legend button:focus-visible{outline:2px solid var(--s0);outline-offset:1px}
.key{display:inline-block;width:16px;height:0;border-top:2px solid currentColor}
.key.dash{border-top-style:dashed}.key.dot{border-top-style:dotted}
.key.area{height:10px;border:0;background:var(--band);opacity:.35;border-radius:2px}
.chart-wrap{position:relative}
.chart{display:block;width:100%;overflow:visible;outline:none}
.chart:focus-visible{outline:2px solid var(--s0);outline-offset:4px;border-radius:4px}
.chart text{fill:var(--text2);font-size:11px;font-variant-numeric:tabular-nums}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--axis);border-radius:8px;
padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);min-width:170px;z-index:5}
.tip .t-date{font-weight:650;margin-bottom:4px}
.tip .row{display:flex;align-items:center;gap:6px;justify-content:space-between;white-space:nowrap}
.tip .row span:last-child{font-variant-numeric:tabular-nums}
.tip .lbl{display:inline-flex;align-items:center;gap:6px;color:var(--text2)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.s0{background:var(--s0)}.s1{background:var(--s1)}.s2{background:var(--s2)}.s3{background:var(--s3)}
.s4{background:var(--s4)}.s5{background:var(--s5)}.s6{background:var(--s6)}.s7{background:var(--s7)}
.sr0{background:var(--r0)}.sr1{background:var(--r1)}.sr2{background:var(--r2)}
.probs{display:flex;flex-direction:column;gap:10px}
.prob-row{display:grid;grid-template-columns:110px 1fr 44px;gap:10px;align-items:center}
.prob-name{font-size:13px}
.prob-track{height:14px;background:transparent;border-radius:4px;position:relative}
.prob-bar{height:14px;border-radius:0 4px 4px 0;min-width:2px}
.prob-val{text-align:right;font-variant-numeric:tabular-nums;font-size:13px}
.prob-desc{grid-column:2/4;font-size:11.5px;color:var(--muted);margin-top:-6px}
.flow{display:flex;flex-direction:column;gap:6px;margin:6px 0 8px}
.flow-row{display:grid;grid-template-columns:52px 1fr;gap:10px;align-items:center}
.flow-label{font-size:12px;color:var(--text2)}
.flow-bar{display:flex;gap:2px;height:22px}
.seg{height:100%;display:flex;align-items:center;justify-content:center;font-size:11px;font-variant-numeric:tabular-nums;overflow:hidden}
.seg:first-child{border-radius:4px 0 0 4px}.seg:last-child{border-radius:0 4px 4px 0}
.seg.r0{background:var(--r0);color:var(--r0t)}.seg.r1{background:var(--r1);color:var(--r1t)}.seg.r2{background:var(--r2);color:var(--r2t)}
.tbl-wrap{overflow-x:auto;margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--text2);font-size:12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
.note{font-size:12px;color:var(--muted);margin:8px 0 0}
.lead{margin:0 0 6px;color:var(--text)}
ul.plain{margin:0;padding-left:18px}ul.plain li{margin:4px 0;color:var(--text)}
.foot{font-size:12px;color:var(--muted);margin-top:8px}
@media (max-width:860px){.hero{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}}
@media (max-width:420px){.kpis{grid-template-columns:1fr}.prob-row{grid-template-columns:88px 1fr 40px}h1{font-size:22px}}
"""


JS = r"""
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const $ = s => document.querySelector(s);
const fmt$ = v => v == null ? '-' : '$' + (v >= 10000 ? Math.round(v).toLocaleString('en-US') : v.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}));
const fmtPct = (v, d=0) => (v >= 0 ? '+' : '') + (v*100).toFixed(d) + '%';
function el(tag, attrs, parent){ const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; }
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ------------------------------------------------------------ 팬 차트
const H = D.hist.prices.length;
const fut = D.fut;
const N = H + fut.dates.length - 1;            // 기준일은 과거 마지막 점과 미래 첫 점이 같음
const dates = D.hist.dates.concat(fut.dates.slice(1));
const fx = j => H - 1 + j;                    // 미래 j번째 → 전체 인덱스
const series = [
  {id:'bands', name:'전체 50%·90% 구간', kind:'area', on:true},
  {id:'hist', name:'실제 주가', kind:'line', color:'var(--text)', on:true},
  ...fut.lines.map(l => ({id:l.key, name:l.name + ' (' + l.note + ')', short:l.name, kind:'line', color:'var(--s'+l.slot+')', dash:l.dash, values:l.values, on:true})),
  {id:'median', name:'전체 중앙값', short:'전체 중앙값', kind:'line', color:'var(--text)', dash:'6 3', values:fut.bands.p50, on:true},
];
const legend = $('#fan-legend');
series.forEach(s => {
  const b = document.createElement('button');
  b.type = 'button'; b.setAttribute('aria-pressed', 'true');
  const key = document.createElement('span');
  key.className = 'key' + (s.kind === 'area' ? ' area' : s.dash === '2 3' ? ' dot' : s.dash ? ' dash' : '');
  if (s.color) key.style.color = s.color;
  b.appendChild(key); b.appendChild(document.createTextNode(s.name));
  b.addEventListener('click', () => { s.on = !s.on; b.setAttribute('aria-pressed', String(s.on)); drawFan(); });
  legend.appendChild(b);
});
const eventIdx = D.events.map(d => dates.indexOf(d)).filter(i => i >= 0);

let fanState = null, hoverIdx = null;
function drawFan(){
  const svg = $('#fan'); svg.innerHTML = '';
  const W = svg.clientWidth || 800;
  const Ht = Math.round(Math.max(280, Math.min(460, W * 0.52)));
  svg.setAttribute('viewBox', `0 0 ${W} ${Ht}`); svg.style.height = Ht + 'px';
  const m = {l: W < 520 ? 46 : 58, r: 12, t: 10, b: 26};
  const pw = W - m.l - m.r, ph = Ht - m.t - m.b;
  const X = i => m.l + i / (N - 1) * pw;
  let lo = Infinity, hi = -Infinity;
  const take = v => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
  if (series[1].on) D.hist.prices.forEach(take);
  if (series[0].on) { fut.bands.p5.forEach(take); fut.bands.p95.forEach(take); }
  series.slice(2).forEach(s => { if (s.on) s.values.forEach(take); });
  if (!isFinite(lo)) { lo = D.s0 * .8; hi = D.s0 * 1.2; }
  const pad = (hi - lo) * .06; lo -= pad; hi += pad;
  const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * ph;
  // 격자 + y축
  const g = el('g', {}, svg);
  const step = niceStep((hi - lo) / 5);
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    el('line', {x1:m.l, x2:W - m.r, y1:Y(v), y2:Y(v), stroke:'var(--grid)', 'stroke-width':1}, g);
    const t = el('text', {x:m.l - 6, y:Y(v) + 4, 'text-anchor':'end'}, g); t.textContent = '$' + Math.round(v).toLocaleString('en-US');
  }
  // x축 (월 단위 눈금)
  const months = []; let prev = '';
  dates.forEach((d, i) => { const ym = d.slice(0, 7); if (ym !== prev) { months.push([i, ym]); prev = ym; } });
  const every = Math.max(1, Math.ceil(months.length / Math.max(3, Math.floor(pw / 80))));
  months.forEach(([i, ym], k) => { if (k % every || i === 0) return;
    el('line', {x1:X(i), x2:X(i), y1:m.t + ph, y2:m.t + ph + 4, stroke:'var(--axis)'}, g);
    const t = el('text', {x:X(i), y:Ht - 6, 'text-anchor':'middle'}, g); t.textContent = ym; });
  el('line', {x1:m.l, x2:W - m.r, y1:m.t + ph, y2:m.t + ph, stroke:'var(--axis)'}, g);
  // 기준일·이벤트 세로선
  el('line', {x1:X(H - 1), x2:X(H - 1), y1:m.t, y2:m.t + ph, stroke:'var(--muted)', 'stroke-dasharray':'2 3'}, g);
  const tt = el('text', {x:X(H - 1) + 4, y:m.t + 10}, g); tt.textContent = '기준일';
  eventIdx.forEach(i => { el('line', {x1:X(i), x2:X(i), y1:m.t, y2:m.t + ph, stroke:'var(--muted)', 'stroke-dasharray':'4 3'}, g);
    const t = el('text', {x:X(i) + 4, y:m.t + 24}, g); t.textContent = '이벤트'; });
  // 분포 구간
  if (series[0].on) {
    const area = (a, b) => { let d = ''; a.forEach((v, j) => d += (j ? 'L' : 'M') + X(fx(j)).toFixed(1) + ',' + Y(v).toFixed(1));
      for (let j = b.length - 1; j >= 0; j--) d += 'L' + X(fx(j)).toFixed(1) + ',' + Y(b[j]).toFixed(1); return d + 'Z'; };
    el('path', {d:area(fut.bands.p95, fut.bands.p5), fill:'var(--band)', 'fill-opacity':css('--band90')}, svg);
    el('path', {d:area(fut.bands.p75, fut.bands.p25), fill:'var(--band)', 'fill-opacity':css('--band50')}, svg);
  }
  const line = (vals, offset, color, dash, width) => { let d = ''; vals.forEach((v, j) => { if (v == null) return; d += (d ? 'L' : 'M') + X(offset + j).toFixed(1) + ',' + Y(v).toFixed(1); });
    const p = el('path', {d, fill:'none', stroke:color, 'stroke-width':width || 2, 'stroke-linejoin':'round', 'stroke-linecap':'round'}, svg);
    if (dash) p.setAttribute('stroke-dasharray', dash); };
  if (series[1].on) line(D.hist.prices, 0, 'var(--text)', '', 1.6);
  series.slice(2).forEach(s => { if (s.on) line(s.values, H - 1, s.color, s.dash, 2); });
  // 호버 레이어
  const cross = el('line', {y1:m.t, y2:m.t + ph, stroke:'var(--text2)', 'stroke-width':1, visibility:'hidden'}, svg);
  const dots = el('g', {}, svg);
  el('rect', {x:m.l, y:m.t, width:pw, height:ph, fill:'transparent', class:'hit'}, svg);
  fanState = {X, Y, m, pw, ph, cross, dots, W};
  if (hoverIdx != null) showFan(hoverIdx);
}
function niceStep(raw){ const p = Math.pow(10, Math.floor(Math.log10(raw))); const f = raw / p; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }
function showFan(i){
  const s = fanState; if (!s) return; hoverIdx = i;
  const x = s.X(i); s.cross.setAttribute('x1', x); s.cross.setAttribute('x2', x); s.cross.setAttribute('visibility', 'visible');
  s.dots.innerHTML = '';
  const rows = [];
  const dot = (v, color) => el('circle', {cx:x, cy:s.Y(v), r:4, fill:color, stroke:'var(--surface)', 'stroke-width':2}, s.dots);
  if (i <= H - 1 && series[1].on) { const v = D.hist.prices[i]; dot(v, 'var(--text)'); rows.push(['실제 주가', v, 'var(--text)']); }
  if (i >= H - 1) {
    const j = i - (H - 1);
    series.slice(2).forEach(se => { if (!se.on) return; const v = se.values[j]; dot(v, se.color); rows.push([se.short, v, se.color]); });
  }
  rows.sort((a, b) => b[1] - a[1]);
  const tip = $('#fan-tip');
  let h = `<div class="t-date">${dates[i]}${i === H - 1 ? ' (기준일)' : ''}</div>`;
  rows.forEach(([n, v, c]) => { h += `<div class="row"><span class="lbl"><span class="sw" style="background:${c}"></span>${n}</span><span>${fmt$(v)} <small>${fmtPct(v / D.s0 - 1)}</small></span></div>`; });
  if (i > H - 1 && series[0].on) { const j = i - (H - 1);
    h += `<div class="row"><span class="lbl">50% 구간</span><span>${fmt$(fut.bands.p25[j])} ~ ${fmt$(fut.bands.p75[j])}</span></div>`;
    h += `<div class="row"><span class="lbl">90% 구간</span><span>${fmt$(fut.bands.p5[j])} ~ ${fmt$(fut.bands.p95[j])}</span></div>`; }
  tip.innerHTML = h; tip.hidden = false;
  const wrap = $('#fan-wrap').getBoundingClientRect();
  const tw = tip.offsetWidth; let left = x + 14; if (left + tw > wrap.width) left = x - tw - 14; if (left < 0) left = 4;
  tip.style.left = left + 'px'; tip.style.top = (s.m.t + 6) + 'px';
}
function hideFan(){ hoverIdx = null; if (fanState) { fanState.cross.setAttribute('visibility', 'hidden'); fanState.dots.innerHTML = ''; } $('#fan-tip').hidden = true; }
const fanSvg = $('#fan');
fanSvg.addEventListener('pointermove', e => { if (!fanState) return; const r = fanSvg.getBoundingClientRect();
  const px = (e.clientX - r.left) * (fanState.W / r.width);
  const i = Math.round((px - fanState.m.l) / fanState.pw * (N - 1)); if (i < 0 || i > N - 1) { hideFan(); return; } showFan(i); });
fanSvg.addEventListener('pointerleave', hideFan);
fanSvg.addEventListener('keydown', e => { if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return; e.preventDefault();
  const step = e.shiftKey ? 21 : 1; const i = hoverIdx == null ? H - 1 : hoverIdx + (e.key === 'ArrowRight' ? step : -step);
  showFan(Math.max(0, Math.min(N - 1, i))); });
fanSvg.addEventListener('blur', hideFan);

// ------------------------------------------------------------ 시나리오 확률 막대
const probs = $('#probs'); const pmax = Math.max(...D.scenarios.map(s => s.p));
D.scenarios.forEach(s => { const row = document.createElement('div'); row.className = 'prob-row';
  row.innerHTML = `<div class="prob-name"><span class="sw s${s.slot}"></span>${s.name}</div><div class="prob-track"><div class="prob-bar s${s.slot}" style="width:${(s.p / pmax * 100).toFixed(1)}%"></div></div><div class="prob-val">${Math.round(s.p * 100)}%</div><div class="prob-desc">${s.desc}</div>`;
  probs.appendChild(row); });

// ------------------------------------------------------------ 만기 수익률 분포
function drawDist(){
  const svg = $('#dist'); svg.innerHTML = '';
  const W = svg.clientWidth || 500, Ht = 230; svg.setAttribute('viewBox', `0 0 ${W} ${Ht}`); svg.style.height = Ht + 'px';
  const m = {l:10, r:10, t:10, b:24}, pw = W - m.l - m.r, ph = Ht - m.t - m.b;
  const bins = D.dist.bins, nb = bins.length - 1;
  const total = Array.from({length:nb}, (_, k) => D.dist.groups.reduce((a, g) => a + g.mass[k], 0));
  const top = Math.max(...total) * 1.08;
  const X = v => m.l + (v - bins[0]) / (bins[nb] - bins[0]) * pw, Y = v => m.t + (1 - v / top) * ph;
  let d = `M${X(bins[0])},${Y(0)}`; total.forEach((v, k) => d += `L${X(bins[k])},${Y(v)}L${X(bins[k + 1])},${Y(v)}`); d += `L${X(bins[nb])},${Y(0)}Z`;
  el('path', {d, fill:'var(--band)', 'fill-opacity':css('--band50')}, svg);
  D.dist.groups.forEach(g => { let p = ''; g.mass.forEach((v, k) => p += (k ? 'L' : 'M') + X(bins[k]) + ',' + Y(v) + 'L' + X(bins[k + 1]) + ',' + Y(v));
    const path = el('path', {d:p, fill:'none', stroke:'var(--s' + g.slot + ')', 'stroke-width':1.6}, svg); if (g.dash) path.setAttribute('stroke-dasharray', g.dash); });
  el('line', {x1:X(0), x2:X(0), y1:m.t, y2:m.t + ph, stroke:'var(--muted)', 'stroke-dasharray':'2 3'}, svg);
  el('line', {x1:m.l, x2:W - m.r, y1:m.t + ph, y2:m.t + ph, stroke:'var(--axis)'}, svg);
  const st = niceStep((bins[nb] - bins[0]) / 5);
  for (let v = Math.ceil(bins[0] / st) * st; v <= bins[nb]; v += st) { const t = el('text', {x:X(v), y:Ht - 6, 'text-anchor':'middle'}, svg); t.textContent = fmtPct(v); }
  const band = el('rect', {y:m.t, height:ph, fill:'var(--text)', 'fill-opacity':.06, visibility:'hidden'}, svg);
  const hit = el('rect', {x:m.l, y:m.t, width:pw, height:ph, fill:'transparent'}, svg);
  const tip = $('#dist-tip');
  hit.addEventListener('pointermove', e => { const r = svg.getBoundingClientRect(); const px = (e.clientX - r.left) * (W / r.width);
    const k = Math.min(nb - 1, Math.max(0, Math.floor((px - m.l) / pw * nb)));
    band.setAttribute('x', X(bins[k])); band.setAttribute('width', X(bins[k + 1]) - X(bins[k])); band.setAttribute('visibility', 'visible');
    let h = `<div class="t-date">수익률 ${fmtPct(bins[k], 1)} ~ ${fmtPct(bins[k + 1], 1)}</div><div class="row"><span class="lbl">전체 확률</span><span>${(total[k] * 100).toFixed(2)}%</span></div>`;
    D.dist.groups.forEach(g => h += `<div class="row"><span class="lbl"><span class="sw s${g.slot}"></span>${g.name}</span><span>${(g.mass[k] * 100).toFixed(2)}%</span></div>`);
    tip.innerHTML = h; tip.hidden = false; const tw = tip.offsetWidth; const x = X(bins[k]);
    tip.style.left = (x + tw + 20 > W ? Math.max(4, x - tw - 10) : x + 14) + 'px'; tip.style.top = '8px'; });
  hit.addEventListener('pointerleave', () => { band.setAttribute('visibility', 'hidden'); tip.hidden = true; });
}

// ------------------------------------------------------------ 국면 흐름
function drawRegime(){
  const rf = D.regimeFlow; if (!rf) return;
  const box = $('#regime-flow'), lg = $('#regime-legend');
  rf.names.forEach((n, i) => { const s = document.createElement('span'); s.className = 'item'; s.innerHTML = `<span class="sw sr${i}"></span>${n}`; lg.appendChild(s); });
  rf.rows.forEach(r => { const row = document.createElement('div'); row.className = 'flow-row';
    const bar = r.probs.map((p, i) => p > 0.004 ? `<div class="seg r${i}" style="flex:${p} 1 0" title="${r.label} · ${rf.names[i]} ${(p * 100).toFixed(0)}%">${p >= 0.09 ? Math.round(p * 100) + '%' : ''}</div>` : '').join('');
    row.innerHTML = `<div class="flow-label">${r.label}</div><div class="flow-bar">${bar}</div>`; box.appendChild(row); });
}

function drawAll(){ drawFan(); drawDist(); }
drawAll(); drawRegime();
let raf = null; window.addEventListener('resize', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(drawAll); });
})();
"""
