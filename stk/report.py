"""콘솔(텍스트)·마크다운·JSON 리포트."""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import numpy as np

from .forecast import Forecast

DISCLAIMER = (
    "※ 본 결과는 과거 가격 통계와 가정된 시나리오에 기반한 확률적 시뮬레이션이며, "
    "실적·뉴스·거시 이벤트를 직접 반영하지 않습니다. 투자 권유가 아니며 투자 판단의 책임은 본인에게 있습니다."
)


# ---------------------------------------------------------------- 포맷 도우미
def disp_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int, align: str = "left") -> str:
    gap = max(0, width - disp_width(text))
    if align == "right":
        return " " * gap + text
    if align == "center":
        return " " * (gap // 2) + text + " " * (gap - gap // 2)
    return text + " " * gap


def table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    widths = [max(disp_width(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    line = "  ".join(pad(h, w, "center") for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(pad(str(c), w, a) for c, w, a in zip(r, widths, aligns)) for r in rows]
    return "\n".join([line, sep, *body])


def money(x: float) -> str:
    if not math.isfinite(x):
        return "-"
    return f"${x:,.2f}" if x < 10_000 else f"${x:,.0f}"


def pct(x: float | None, signed: bool = True, digits: int = 1) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


MODEL_NAMES = {"ensemble": "앙상블", "scenario": "시나리오", "regime": "국면(HMM)"}


def model_text(fc: Forecast) -> str:
    if fc.model == "ensemble" and fc.regime is not None:
        w = fc.regime_weight
        return f"앙상블 (시나리오 모델 {1 - w:.0%} + 국면 모델 {w:.0%})"
    if fc.model == "regime" and fc.regime is not None:
        return "국면(HMM) 모델"
    return "시나리오 모델"


def market_lines(fc: Forecast) -> list[str]:
    m, cb = fc.market, fc.calib
    if m is None:
        return ["  시장 데이터 미사용 (--no-market 또는 데이터 없음)"]
    parts = []
    if m.vix is not None:
        parts.append(f"VIX {m.vix:.1f} (5년 백분위 {m.vix_rank or 0:.0%})")
    if m.term_ratio is not None:
        state = "역전 → 단기 공포" if m.term_ratio > 1 else "정상(우상향)"
        parts.append(f"VIX/VIX3M {m.term_ratio:.2f} ({state})")
    if m.spy_drawdown is not None:
        parts.append(f"S&P500 고점 대비 {pct(m.spy_drawdown)}")
    if m.spy_trend_score is not None:
        parts.append(f"S&P500 추세 {m.spy_trend_score:+.2f}")
    lines = [f"  시장 스트레스 {m.stress:+.2f} ({m.label})  |  " + "  |  ".join(parts)]
    if cb.vol_fwd is not None:
        lines.append(
            f"  변동성 반영  과거 EWMA {pct(cb.vol_ewma, False)} + VIX 기반 선행 {pct(cb.vol_fwd, False)} "
            f"→ 출발 변동성 {pct(cb.vol_now, False)} (장기 {pct(cb.vol_long, False)}로 회귀)"
        )
    return lines


def drift_text(fc: Forecast) -> str:
    """기본 기대수익률 산출 내역."""
    cb = fc.calib
    capm = f"CAPM {pct(cb.capm_drift)} (rf {pct(cb.risk_free, False)} + β {cb.beta:.2f} × ERP {pct(cb.erp, False)})"
    if cb.trend_weight <= 0:
        return f"기본 {pct(cb.base_drift)} = {capm}, 추세 미반영"
    return (f"기본 {pct(cb.base_drift)} = {capm} × {1 - cb.trend_weight:.0%} + 추세 {pct(cb.trend_drift)} × "
            f"{cb.trend_weight:.0%}")


def trend_note(fc: Forecast) -> str:
    return "시나리오 확률에 반영" if fc.trend_tilt else "참고용 · 확률에 미반영"


def regime_summary_line(fc: Forecast) -> str:
    st = fc.regime.state
    cur = st.current_regime
    later_day = fc.checkpoints[-1][1]
    occ = st.occupancy(later_day)
    return (
        f"  현재 '{st.names[cur]}' 국면일 확률 {st.current[cur]:.0%} (평균 지속 {st.duration[cur]:.0f}거래일) → "
        f"{fc.checkpoints[-1][0]} 후 " + " · ".join(f"{n} {p:.0%}" for n, p in zip(st.names, occ))
    )


# ---------------------------------------------------------------- 텍스트 리포트
def render_text(fc: Forecast) -> str:
    st, cb, s0 = fc.stats, fc.calib, fc.s0
    out: list[str] = []
    title = f" {fc.symbol} 시나리오별 주가 흐름 예측 "
    out.append("═" * 78)
    out.append(pad(title, 78, "center"))
    out.append("═" * 78)
    out.append(
        f"기준일 {st.last_date:%Y-%m-%d}  |  현재가 {money(s0)}  |  예측기간 {fc.horizon}거래일 "
        f"(~{fc.dates[-1]:%Y-%m-%d})  |  시뮬레이션 {fc.n_paths:,}회 × {len(fc.scenarios)}개 시나리오"
    )
    out.append(f"모델: {model_text(fc)}")
    if fc.events:
        out.append("이벤트: " + ", ".join(f"{d:%Y-%m-%d} (하루 충격 ±{s:.1%})" for d, s in fc.events))
    out.append("")

    # 1. 전체 흐름
    ol = fc.outlook
    out.append("■ 전체 흐름 전망")
    out.append(f"  ▶ {ol.label}  —  {ol.shape}")
    out.append(
        f"  · 상승 마감 확률 {pct(ol.prob_up, False, 0)}  |  기대수익률 {pct(ol.expected_return)}  "
        f"|  중앙값 수익률 {pct(ol.median_return)}"
    )
    out.append(
        f"  · 만기 90% 범위 {money(ol.band90[0])} ~ {money(ol.band90[1])}  |  변동성 {ol.volatility}"
    )
    out.append(
        f"  · 기간 중 +20% 도달 확률 {pct(fc.mix_prob_hit_up, False, 0)}  |  "
        f"−20% 도달 확률 {pct(fc.mix_prob_hit_down, False, 0)}  |  최유력 시나리오: {ol.dominant}"
    )
    out.append("")

    # 2. 현재 상태 진단
    out.append("■ 현재 상태 진단")
    out.append(
        f"  추세 점수 {st.trend_score:+.2f} ({st.trend_label}, {trend_note(fc)})  |  RSI(14) {st.rsi14:.0f}  "
        f"|  52주 고점 대비 {pct(st.drawdown_now)}"
    )
    out.append(
        f"  이동평균  20일 {money(st.ma20)} · 50일 {money(st.ma50)} · 200일 {money(st.ma200)}  "
        f"(현재가는 200일선 대비 {pct(s0 / st.ma200 - 1)})"
    )
    out.append(
        f"  수익률  1년 {pct(st.return_1y)} · 12-1개월 모멘텀 {pct(st.momentum_12_1)} · "
        f"표본 CAGR {pct(st.cagr)}"
    )
    beta_txt = f"{st.beta:.2f}" if st.beta is not None else "N/A"
    out.append(
        f"  변동성  현재 {pct(st.vol_now, False)} · 1년 {pct(st.vol_1y, False)} · 장기 {pct(st.vol_long, False)}  "
        f"|  베타 {beta_txt}  |  꼬리 두께 df {st.tail_df:.1f}"
    )
    sig_names = {
        "ma200_gap": "200일선 위치",
        "ma_cross": "50/200 크로스",
        "momentum": "12-1 모멘텀",
        "slope": "6개월 추세 기울기",
        "rsi": "RSI 과열/과매도",
    }
    sig_txt = " · ".join(
        f"{sig_names[k]} {v:+.2f}" for k, v in st.signals.items() if v is not None and math.isfinite(v)
    )
    out.append(f"  신호 구성  {sig_txt}")
    out.append("")
    out.append("■ 시장 환경")
    out.extend(market_lines(fc))
    out.append(f"  기대수익률  {drift_text(fc)}  |  시나리오 폭 ±{pct(cb.spread, False)}")
    out.append("")

    # 3. 시나리오별 결과
    out.append(f"■ 시나리오별 결과 (만기 {fc.dates[-1]:%Y-%m-%d})")
    headers = ["시나리오", "확률", "연 기대수익률", "기대가격", "중앙값", "5%~95% 범위", "상승확률", "중앙 MDD"]
    rows = []
    for r in fc.results:
        sc = r.scenario
        rows.append(
            [
                sc.name,
                pct(sc.probability, False, 0),
                pct(sc.avg_drift + sc.jump_drag),
                f"{money(r.terminal_mean)} ({pct(r.terminal_mean / s0 - 1, digits=0)})",
                money(r.terminal_quantiles[0.5]),
                f"{money(r.terminal_quantiles[0.05])} ~ {money(r.terminal_quantiles[0.95])}",
                pct(r.prob_up, False, 0),
                pct(r.median_max_drawdown, digits=0),
            ]
        )
    if fc.regime is not None:
        rg = fc.regime
        rows.append(
            [
                f"국면 모델(HMM){' *' if rg.weight == 0 else ''}",
                pct(rg.weight, False, 0) if rg.weight > 0 else "참고",
                pct(float(rg.state.current @ rg.state.drift)),
                f"{money(rg.terminal_mean)} ({pct(rg.terminal_mean / s0 - 1, digits=0)})",
                money(rg.terminal_quantiles[0.5]),
                f"{money(rg.terminal_quantiles[0.05])} ~ {money(rg.terminal_quantiles[0.95])}",
                pct(rg.prob_up, False, 0),
                pct(rg.median_max_drawdown, digits=0),
            ]
        )
    rows.append(
        [
            "▶ 확률가중 전체",
            "100%",
            "",
            f"{money(fc.mix_expected_price)} ({pct(fc.mix_expected_price / s0 - 1, digits=0)})",
            money(fc.mix_terminal_quantiles[0.5]),
            f"{money(fc.mix_terminal_quantiles[0.05])} ~ {money(fc.mix_terminal_quantiles[0.95])}",
            pct(fc.mix_prob_up, False, 0),
            "",
        ]
    )
    out.append(_indent(table(headers, rows)))
    if fc.regime is not None and fc.regime_weight > 0 and fc.regime_weight < 1:
        out.append(f"  ※ 시나리오 확률은 시나리오 모델 내부 비중입니다. 전체 = 시나리오 모델 {1 - fc.regime_weight:.0%} + 국면 모델 {fc.regime_weight:.0%}")
    elif fc.regime is not None and fc.regime_weight == 0:
        out.append("  * 국면 모델은 참고용으로만 표시 (전체 분포에 미포함)")
    out.append("")
    out.append("  시나리오 확률 분포")
    for sc in fc.scenarios:
        out.append(f"   {pad(sc.name, 14)} {bar(sc.probability)} {pct(sc.probability, False, 0):>4}  {sc.description}")
    out.append("")

    # 4. 국면 분석
    if fc.regime is not None:
        st = fc.regime.state
        out.append("■ 국면 분석 (은닉 마르코프 모델, 데이터 기반)")
        out.append(regime_summary_line(fc))
        rows = [
            [
                st.names[i],
                pct(st.drift[i]),
                pct(st.vol[i], False, 0),
                f"{st.duration[i]:.0f}일",
                pct(st.stationary[i], False, 0),
                pct(st.current[i], False, 0),
            ]
            for i in range(st.params.k)
        ]
        out.append(_indent(table(["국면", "연 기대수익률*", "연 변동성", "평균 지속", "장기 비중", "현재 확률"], rows)))
        flow = fc.regime_flow()
        headers = ["기간"] + st.names
        rows = [[r["기간"]] + [pct(r[n], False, 0) for n in st.names] for r in flow]
        out.append("")
        out.append("  기간별 국면 확률 (국면 흐름)")
        out.append(_indent(table(headers, rows), 4))
        out.append("  * 국면별 기대수익률은 표본 차이를 정밀도만큼 축소한 뒤 기본 기대수익률에 맞춘 값")
        out.append("")

    # 5. 기간별 흐름 (중앙값 경로)
    out.append("■ 기간별 흐름 — 시나리오별 중앙값 가격 경로")
    tbl = fc.checkpoint_table()
    extra = ["국면 모델"] if fc.regime is not None else []
    headers = ["기간", "날짜"] + [r.scenario.name for r in fc.results] + extra + ["전체 중앙값", "전체 25~75%"]
    rows = []
    for _, row in tbl.iterrows():
        cells = [row["기간"], row["날짜"]]
        for name in [r.scenario.name for r in fc.results] + extra:
            cells.append(money(row[name]))
        cells.append(f"{money(row['전체 p50'])} ({pct(row['전체 p50'] / s0 - 1, digits=0)})")
        cells.append(f"{money(row['전체 p25'])}~{money(row['전체 p75'])}")
        rows.append(cells)
    out.append(_indent(table(headers, rows, ["left", "left"] + ["right"] * (len(headers) - 2))))
    out.append("")
    out.append(DISCLAIMER)
    return "\n".join(out)


def render_comparison(forecasts: list[Forecast]) -> str:
    """여러 종목 요약 비교표."""
    headers = ["종목", "현재가", "추세", "현재 국면", "전체 흐름", "상승확률", "기대수익률", "90% 범위", "최유력 시나리오"]
    rows = []
    for fc in forecasts:
        ol = fc.outlook
        regime = "-"
        if fc.regime is not None:
            st = fc.regime.state
            regime = f"{st.names[st.current_regime]} {st.current[st.current_regime]:.0%}"
        rows.append(
            [
                fc.symbol,
                money(fc.s0),
                f"{fc.stats.trend_score:+.2f}",
                regime,
                ol.label,
                pct(ol.prob_up, False, 0),
                pct(ol.expected_return),
                f"{money(ol.band90[0])}~{money(ol.band90[1])}",
                ol.dominant,
            ]
        )
    horizon = forecasts[0].horizon if forecasts else 0
    return f"■ 종목 비교 ({horizon}거래일 후)\n" + _indent(
        table(headers, rows, ["left", "right", "right", "left", "left", "right", "right", "right", "left"])
    )


def _indent(text: str, n: int = 2) -> str:
    return "\n".join(" " * n + line for line in text.splitlines())


# ---------------------------------------------------------------- 마크다운
def render_markdown(fc: Forecast, chart_name: str | None = None) -> str:
    st, s0, ol = fc.stats, fc.s0, fc.outlook
    md: list[str] = []
    md.append(f"# {fc.symbol} 시나리오별 주가 흐름 예측")
    md.append("")
    md.append(
        f"- 기준일: **{st.last_date:%Y-%m-%d}** / 현재가: **{money(s0)}**\n"
        f"- 예측기간: {fc.horizon}거래일 (~{fc.dates[-1]:%Y-%m-%d}) / 시뮬레이션 {fc.n_paths:,}회 × {len(fc.scenarios)}개 시나리오\n"
        f"- 모델: {model_text(fc)}"
    )
    if fc.events:
        md.append("- 이벤트: " + ", ".join(f"{d:%Y-%m-%d} (하루 충격 ±{s:.1%})" for d, s in fc.events))
    md.append("")
    md.append("## 전체 흐름 전망")
    md.append("")
    md.append(f"> **{ol.label}** — {ol.shape}")
    md.append("")
    md.append("| 항목 | 값 |\n|---|---|")
    md.append(f"| 상승 마감 확률 | {pct(ol.prob_up, False, 0)} |")
    md.append(f"| 기대수익률 / 중앙값 수익률 | {pct(ol.expected_return)} / {pct(ol.median_return)} |")
    md.append(f"| 만기 90% 범위 | {money(ol.band90[0])} ~ {money(ol.band90[1])} |")
    md.append(f"| 변동성 | {ol.volatility} |")
    md.append(f"| 기간 중 +20% / −20% 도달 확률 | {pct(fc.mix_prob_hit_up, False, 0)} / {pct(fc.mix_prob_hit_down, False, 0)} |")
    md.append(f"| 최유력 시나리오 | {ol.dominant} |")
    md.append("")
    if chart_name:
        md.append(f"![{fc.symbol} forecast]({chart_name})")
        md.append("")
    md.append("## 현재 상태 진단")
    md.append("")
    md.append("| 지표 | 값 |\n|---|---|")
    md.append(f"| 추세 점수 | {st.trend_score:+.2f} ({st.trend_label}, {trend_note(fc)}) |")
    md.append(f"| 이동평균 20/50/200일 | {money(st.ma20)} / {money(st.ma50)} / {money(st.ma200)} |")
    md.append(f"| RSI(14) | {st.rsi14:.0f} |")
    md.append(f"| 1년 수익률 / 12-1 모멘텀 | {pct(st.return_1y)} / {pct(st.momentum_12_1)} |")
    md.append(f"| 52주 고점 대비 | {pct(st.drawdown_now)} |")
    md.append(f"| 변동성 현재/1년/장기 | {pct(st.vol_now, False)} / {pct(st.vol_1y, False)} / {pct(st.vol_long, False)} |")
    md.append(f"| 베타 (vs 벤치마크) | {st.beta:.2f} |" if st.beta is not None else "| 베타 | N/A |")
    cb = fc.calib
    md.append(
        f"| 기본 기대수익률 | {drift_text(fc)} |"
    )
    md.append("")
    md.append("## 시장 환경")
    md.append("")
    md.extend(line.strip() and f"- {line.strip()}" for line in market_lines(fc))
    md.append("")
    md.append("## 시나리오별 결과")
    md.append("")
    md.append("| 시나리오 | 설명 | 확률 | 연 기대수익률 | 기대가격 | 중앙값 | 5%~95% | 상승확률 | 중앙 MDD |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in fc.results:
        sc = r.scenario
        md.append(
            f"| {sc.name} | {sc.description} | {pct(sc.probability, False, 0)} | {pct(sc.avg_drift + sc.jump_drag)} | "
            f"{money(r.terminal_mean)} | {money(r.terminal_quantiles[0.5])} | "
            f"{money(r.terminal_quantiles[0.05])}~{money(r.terminal_quantiles[0.95])} | {pct(r.prob_up, False, 0)} | "
            f"{pct(r.median_max_drawdown, digits=0)} |"
        )
    if fc.regime is not None:
        rg = fc.regime
        md.append(
            f"| 국면 모델(HMM) | 데이터 기반 국면 전환 시뮬레이션 | {pct(rg.weight, False, 0) if rg.weight > 0 else '참고'} | "
            f"{pct(float(rg.state.current @ rg.state.drift))} | {money(rg.terminal_mean)} | {money(rg.terminal_quantiles[0.5])} | "
            f"{money(rg.terminal_quantiles[0.05])}~{money(rg.terminal_quantiles[0.95])} | {pct(rg.prob_up, False, 0)} | "
            f"{pct(rg.median_max_drawdown, digits=0)} |"
        )
    md.append(
        f"| **확률가중 전체** | | 100% | | {money(fc.mix_expected_price)} | {money(fc.mix_terminal_quantiles[0.5])} | "
        f"{money(fc.mix_terminal_quantiles[0.05])}~{money(fc.mix_terminal_quantiles[0.95])} | {pct(fc.mix_prob_up, False, 0)} | |"
    )
    md.append("")
    if fc.regime is not None:
        st = fc.regime.state
        md.append("## 국면 분석 (HMM)")
        md.append("")
        md.append(regime_summary_line(fc).strip())
        md.append("")
        md.append("| 국면 | 연 기대수익률 | 연 변동성 | 평균 지속 | 장기 비중 | 현재 확률 |\n|---|---:|---:|---:|---:|---:|")
        for i in range(st.params.k):
            md.append(
                f"| {st.names[i]} | {pct(st.drift[i])} | {pct(st.vol[i], False, 0)} | {st.duration[i]:.0f}일 | "
                f"{pct(st.stationary[i], False, 0)} | {pct(st.current[i], False, 0)} |"
            )
        md.append("")
        md.append("| 기간 | " + " | ".join(st.names) + " |\n|---|" + "---:|" * st.params.k)
        for r in fc.regime_flow():
            md.append(f"| {r['기간']} | " + " | ".join(pct(r[n], False, 0) for n in st.names) + " |")
        md.append("")
    md.append("## 기간별 흐름 (중앙값 경로)")
    md.append("")
    tbl = fc.checkpoint_table()
    names = [r.scenario.name for r in fc.results] + (["국면 모델"] if fc.regime is not None else [])
    md.append("| 기간 | 날짜 | " + " | ".join(names) + " | 전체 중앙값 | 전체 25~75% |")
    md.append("|---|---|" + "---:|" * (len(names) + 2))
    for _, row in tbl.iterrows():
        cells = [row["기간"], row["날짜"]] + [money(row[n]) for n in names]
        cells.append(money(row["전체 p50"]))
        cells.append(f"{money(row['전체 p25'])}~{money(row['전체 p75'])}")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append(f"_{DISCLAIMER}_")
    md.append("")
    return "\n".join(md)


# ---------------------------------------------------------------- JSON
def write_json(fc: Forecast, path: Path) -> None:
    path.write_text(json.dumps(fc.to_dict(), ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"JSON 직렬화 불가: {type(obj)}")


__all__ = ["render_text", "render_comparison", "render_markdown", "write_json"]
