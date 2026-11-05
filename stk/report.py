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
        f"  추세 점수 {st.trend_score:+.2f} ({st.trend_label})  |  RSI(14) {st.rsi14:.0f}  "
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
    out.append(
        f"  기대수익률 보정  CAPM {pct(cb.capm_drift)} (rf {pct(cb.risk_free, False)}, β {cb.beta:.2f}, "
        f"ERP {pct(cb.erp, False)}) × {1 - cb.trend_weight:.0%} + 추세 {pct(cb.trend_drift)} × "
        f"{cb.trend_weight:.0%} = 기본 {pct(cb.base_drift)}  |  시나리오 폭 ±{pct(cb.spread, False)}"
    )
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
    out.append("")
    out.append("  시나리오 확률 분포")
    for sc in fc.scenarios:
        out.append(f"   {pad(sc.name, 14)} {bar(sc.probability)} {pct(sc.probability, False, 0):>4}  {sc.description}")
    out.append("")

    # 4. 기간별 흐름 (중앙값 경로)
    out.append("■ 기간별 흐름 — 시나리오별 중앙값 가격 경로")
    tbl = fc.checkpoint_table()
    headers = ["기간", "날짜"] + [r.scenario.name for r in fc.results] + ["전체 중앙값", "전체 25~75%"]
    rows = []
    for _, row in tbl.iterrows():
        cells = [row["기간"], row["날짜"]]
        for r in fc.results:
            cells.append(money(row[r.scenario.name]))
        cells.append(f"{money(row['전체 p50'])} ({pct(row['전체 p50'] / s0 - 1, digits=0)})")
        cells.append(f"{money(row['전체 p25'])}~{money(row['전체 p75'])}")
        rows.append(cells)
    out.append(_indent(table(headers, rows, ["left", "left"] + ["right"] * (len(headers) - 2))))
    out.append("")
    out.append(DISCLAIMER)
    return "\n".join(out)


def render_comparison(forecasts: list[Forecast]) -> str:
    """여러 종목 요약 비교표."""
    headers = ["종목", "현재가", "추세", "전체 흐름", "상승확률", "기대수익률", "90% 범위", "최유력 시나리오"]
    rows = []
    for fc in forecasts:
        ol = fc.outlook
        rows.append(
            [
                fc.symbol,
                money(fc.s0),
                f"{fc.stats.trend_score:+.2f}",
                ol.label,
                pct(ol.prob_up, False, 0),
                pct(ol.expected_return),
                f"{money(ol.band90[0])}~{money(ol.band90[1])}",
                ol.dominant,
            ]
        )
    horizon = forecasts[0].horizon if forecasts else 0
    return f"■ 종목 비교 ({horizon}거래일 후)\n" + _indent(
        table(headers, rows, ["left", "right", "right", "left", "right", "right", "right", "left"])
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
        f"- 예측기간: {fc.horizon}거래일 (~{fc.dates[-1]:%Y-%m-%d}) / 시뮬레이션 {fc.n_paths:,}회 × {len(fc.scenarios)}개 시나리오"
    )
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
    md.append(f"| 추세 점수 | {st.trend_score:+.2f} ({st.trend_label}) |")
    md.append(f"| 이동평균 20/50/200일 | {money(st.ma20)} / {money(st.ma50)} / {money(st.ma200)} |")
    md.append(f"| RSI(14) | {st.rsi14:.0f} |")
    md.append(f"| 1년 수익률 / 12-1 모멘텀 | {pct(st.return_1y)} / {pct(st.momentum_12_1)} |")
    md.append(f"| 52주 고점 대비 | {pct(st.drawdown_now)} |")
    md.append(f"| 변동성 현재/1년/장기 | {pct(st.vol_now, False)} / {pct(st.vol_1y, False)} / {pct(st.vol_long, False)} |")
    md.append(f"| 베타 (vs 벤치마크) | {st.beta:.2f} |" if st.beta is not None else "| 베타 | N/A |")
    cb = fc.calib
    md.append(
        f"| 기본 기대수익률 | {pct(cb.base_drift)} = CAPM {pct(cb.capm_drift)}×{1 - cb.trend_weight:.0%} + 추세 {pct(cb.trend_drift)}×{cb.trend_weight:.0%} |"
    )
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
    md.append(
        f"| **확률가중 전체** | | 100% | | {money(fc.mix_expected_price)} | {money(fc.mix_terminal_quantiles[0.5])} | "
        f"{money(fc.mix_terminal_quantiles[0.05])}~{money(fc.mix_terminal_quantiles[0.95])} | {pct(fc.mix_prob_up, False, 0)} | |"
    )
    md.append("")
    md.append("## 기간별 흐름 (중앙값 경로)")
    md.append("")
    tbl = fc.checkpoint_table()
    names = [r.scenario.name for r in fc.results]
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
