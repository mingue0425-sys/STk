"""워크포워드(walk-forward) 백테스트: 과거 시점마다 그때까지의 데이터로만 예측하고
실제 결과와 비교해 예측 구간의 보정 상태(calibration)와 방향 적중률을 평가합니다.

비교 기준(naive)은 '기대수익률 0, 장기 변동성'의 단순 랜덤워크 정규분포입니다.
CRPS(연속 순위 확률 점수)는 낮을수록 좋으며, skill = 1 − CRPS_모델 / CRPS_기준 입니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import TRADING_DAYS
from .data import risk_free_at
from .forecast import run_forecast
from .market import market_context
from .report import bar, pct, table


@dataclass
class BacktestResult:
    symbol: str
    horizon: int
    step: int
    records: pd.DataFrame
    summary: dict

    def to_dict(self) -> dict:
        recs = self.records.copy()
        recs["date"] = recs["date"].dt.strftime("%Y-%m-%d")
        return {"symbol": self.symbol, "horizon": self.horizon, "step": self.step, "summary": self.summary,
                "records": recs.to_dict(orient="records")}


def crps_weighted_samples(samples: np.ndarray, weights: np.ndarray, y: float) -> float:
    """가중 표본의 CRPS = E|X − y| − ½·E|X − X'|."""
    order = np.argsort(samples)
    x = samples[order]
    w = weights[order] / weights.sum()
    term1 = float(w @ np.abs(x - y))
    cdf = np.cumsum(w)
    term2 = float(2.0 * np.sum(w * x * (2.0 * cdf - w - 1.0)))
    return term1 - 0.5 * term2


def crps_normal(mu: float, sigma: float, y: float) -> float:
    z = (y - mu) / sigma
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def walk_forward(
    prices: pd.DataFrame,
    symbol: str,
    horizon: int = 63,
    step: int = 21,
    years: float = 5.0,
    n_paths: int = 2000,
    benchmark: pd.DataFrame | None = None,
    risk_free_series: pd.Series | None = None,
    vix: pd.DataFrame | None = None,
    vix3m: pd.DataFrame | None = None,
    use_market: bool = True,
    min_history: int = 2 * TRADING_DAYS,
    progress: bool = False,
    **forecast_kwargs,
) -> BacktestResult:
    """각 예측 시점에는 그 시점까지의 가격·시장 데이터만 사용합니다 (미래 정보 차단)."""
    p = prices["adj_close"].astype(float).dropna() if "adj_close" in prices else prices["close"].astype(float)
    prices = prices.loc[p.index]
    last_origin = len(p) - 1 - horizon
    if last_origin < min_history:
        raise ValueError(f"{symbol}: 백테스트에 필요한 데이터가 부족합니다 (최소 {min_history + horizon}거래일)")
    start_date = p.index[-1] - pd.Timedelta(days=int(years * 365.25))
    origins = [i for i in range(last_origin, min_history - 1, -step) if p.index[i] >= start_date][::-1]
    if not origins:
        raise ValueError(f"{symbol}: 평가 구간에 해당하는 시점이 없습니다")

    model = forecast_kwargs.get("model", "ensemble")
    forecast_kwargs.setdefault("regime_info", model != "scenario")
    forecast_kwargs.setdefault("earnings_info", False)  # 백테스트에서는 표시용 실적일 추정 생략 (속도)
    regime_init = None
    rows = []
    for k, i in enumerate(origins):
        date = p.index[i]
        hist = prices.iloc[: i + 1]
        bench = benchmark.loc[:date] if benchmark is not None else None
        mkt = market_context(bench, vix, vix3m, as_of=date) if use_market and bench is not None else None
        fc = run_forecast(
            hist,
            symbol,
            horizon=horizon,
            n_paths=n_paths,
            benchmark=bench,
            risk_free=risk_free_at(risk_free_series, date),
            seed=i,
            market=mkt,
            regime_init=regime_init,
            **forecast_kwargs,
        )
        if fc.regime is not None:
            regime_init = fc.regime.state.params  # 다음 시점 적합의 시작값 (빠른 수렴)
        s0 = fc.s0
        realized = float(p.iloc[i + horizon])
        y = math.log(realized / s0)

        terminal, weights = fc.mix_samples
        samples = np.log(terminal / s0)
        pit = float(weights[samples <= y].sum())
        q = fc.mix_terminal_quantiles
        sigma_h = fc.stats.vol_long * math.sqrt(horizon / TRADING_DAYS)
        rows.append(
            {
                "date": date,
                "price": s0,
                "realized": realized,
                "realized_return": realized / s0 - 1,
                "expected_return": fc.mix_expected_price / s0 - 1,
                "median_return": q[0.5] / s0 - 1,
                "prob_up": fc.mix_prob_up,
                "trend_score": fc.stats.trend_score,
                "outlook": fc.outlook.label,
                "pit": pit,
                "in50": q[0.25] <= realized <= q[0.75],
                "in90": q[0.05] <= realized <= q[0.95],
                "crps": crps_weighted_samples(samples, weights, y),
                "crps_naive": crps_normal(-0.5 * sigma_h**2, sigma_h, y),
            }
        )
        if progress:
            print(f"\r  백테스트 진행 {k + 1}/{len(origins)}", end="", flush=True)
    if progress:
        print()

    df = pd.DataFrame(rows)
    up = df["realized_return"] > 0
    pred_up = df["prob_up"] > 0.5
    decided = df["prob_up"] != 0.5
    mean_crps, mean_naive = float(df["crps"].mean()), float(df["crps_naive"].mean())
    summary = {
        "n": int(len(df)),
        "period": f"{df['date'].iloc[0]:%Y-%m-%d} ~ {df['date'].iloc[-1]:%Y-%m-%d}",
        "coverage50": float(df["in50"].mean()),
        "coverage90": float(df["in90"].mean()),
        "hit_rate": float((pred_up[decided] == up[decided]).mean()) if decided.any() else float("nan"),
        "up_base_rate": float(up.mean()),
        "brier": float(((df["prob_up"] - up.astype(float)) ** 2).mean()),
        "mean_pit": float(df["pit"].mean()),
        "crps": mean_crps,
        "crps_naive": mean_naive,
        "crps_skill": 1.0 - mean_crps / mean_naive if mean_naive > 0 else float("nan"),
        "corr_expected_realized": float(df["expected_return"].corr(df["realized_return"])) if len(df) > 2 else float("nan"),
        "overlapping": step < horizon,
    }
    return BacktestResult(symbol=symbol, horizon=horizon, step=step, records=df, summary=summary)


def render_backtest(bt: BacktestResult) -> str:
    s = bt.summary
    out = []
    out.append("═" * 78)
    out.append(f"  {bt.symbol} 워크포워드 백테스트  ({bt.horizon}거래일 예측, {bt.step}거래일 간격, {s['n']}회)")
    out.append("═" * 78)
    out.append(f"평가 기간: {s['period']}")
    out.append("")
    rows = [
        ["50% 구간 적중률", pct(s["coverage50"], False), "50%", _verdict(s["coverage50"], 0.5, 0.08)],
        ["90% 구간 적중률", pct(s["coverage90"], False), "90%", _verdict(s["coverage90"], 0.9, 0.06)],
        ["방향 적중률", pct(s["hit_rate"], False), f"> {pct(max(s['up_base_rate'], 1 - s['up_base_rate']), False)}*", ""],
        ["Brier 점수 (상승확률)", f"{s['brier']:.3f}", "< 0.250", "양호" if s["brier"] < 0.25 else "개선 필요"],
        ["평균 PIT", f"{s['mean_pit']:.2f}", "0.50", _pit_bias(s["mean_pit"])],
        ["CRPS 모델 / 기준", f"{s['crps']:.4f} / {s['crps_naive']:.4f}", "낮을수록 좋음", ""],
        ["CRPS skill", pct(s["crps_skill"]), "> 0%", "기준보다 나음" if s["crps_skill"] > 0 else "기준보다 못함"],
        ["기대수익률-실현수익률 상관", f"{s['corr_expected_realized']:+.2f}", "> 0", ""],
    ]
    out.append(_indent(table(["지표", "결과", "이상적 값", "평가"], rows, ["left", "right", "right", "left"])))
    out.append("  * 방향 적중률 비교 기준: 항상 다수 방향(상승 또는 하락)만 예측했을 때의 적중률")
    out.append("")
    out.append("  PIT 분포 (실제 결과가 예측 분포의 어느 분위에 떨어졌는지; 고르게 퍼질수록 보정이 잘 된 것)")
    counts, edges = np.histogram(bt.records["pit"], bins=10, range=(0, 1))
    peak = max(counts.max(), 1)
    for c, lo in zip(counts, edges[:-1]):
        out.append(f"   {lo:.1f}~{lo + 0.1:.1f} {bar(c / peak, 30)} {c}")
    out.append("")
    if s["overlapping"]:
        out.append("  ※ 예측 구간이 서로 겹치므로(간격 < 예측기간) 관측치가 독립이 아닙니다. 통계적 유의성은 과대평가될 수 있습니다.")
    return "\n".join(out)


def _verdict(value: float, target: float, tol: float) -> str:
    if abs(value - target) <= tol:
        return "적정"
    return "구간이 너무 좁음(과신)" if value < target else "구간이 너무 넓음(보수적)"


def _pit_bias(mean_pit: float) -> str:
    if mean_pit > 0.58:
        return "실제가 예측보다 높았음(비관 편향)"
    if mean_pit < 0.42:
        return "실제가 예측보다 낮았음(낙관 편향)"
    return "편향 작음"


def _indent(text: str, n: int = 2) -> str:
    return "\n".join(" " * n + line for line in text.splitlines())


# ----------------------------------------------------------------- 설정 비교
_TREND = dict(tilt=True, trend_weight=0.20)
_NO_TREND = dict(tilt=False, trend_weight=0.0)
COMPARE_CONFIGS = {
    "앙상블": dict(model="ensemble", **_NO_TREND),
    "앙상블 + 추세": dict(model="ensemble", **_TREND),
    "시나리오": dict(model="scenario", **_NO_TREND),
    "시나리오 + 추세": dict(model="scenario", **_TREND),
    "국면(HMM)": dict(model="regime", **_NO_TREND),
}


def compare_configs(prices: pd.DataFrame, symbol: str, regime_weight: float = 0.7, **common) -> pd.DataFrame:
    """여러 모델 설정을 같은 조건으로 백테스트해 요약표를 반환합니다."""
    progress = common.pop("progress", False)
    rows = []
    for name, cfg in COMPARE_CONFIGS.items():
        if progress:
            print(f"  [{name}]", flush=True)
        bt = walk_forward(prices, symbol, progress=progress, regime_weight=regime_weight, **cfg, **common)
        rows.append({"설정": name, **bt.summary})
    return pd.DataFrame(rows)


def render_comparison_table(symbol: str, horizon: int, summary: pd.DataFrame) -> str:
    best = summary["crps_skill"].idxmax()
    rows = []
    for i, r in summary.iterrows():
        rows.append(
            [
                ("▶ " if i == best else "  ") + r["설정"],
                pct(r["coverage50"], False),
                pct(r["coverage90"], False),
                pct(r["hit_rate"], False),
                f"{r['brier']:.3f}",
                pct(r["crps_skill"]),
                f"{r['corr_expected_realized']:+.2f}",
            ]
        )
    out = [
        "═" * 78,
        f"  {symbol} 모델 설정 비교 ({horizon}거래일 예측, {int(summary['n'].iloc[0])}회, {summary['period'].iloc[0]})",
        "═" * 78,
        _indent(table(["설정", "50% 적중", "90% 적중", "방향 적중", "Brier", "CRPS skill", "기대-실현 상관"], rows)),
        "",
        f"  ▶ 이 기간 CRPS skill 최고: {summary.loc[best, '설정']}  (차이가 1~2%p 이내면 통계적으로 의미 없음)",
        "  ※ 과거 성적이 가장 좋은 설정을 종목별로 골라 쓰는 것은 권장하지 않습니다. 10개 종목 실험에서",
        "    '과거 최고 설정 선택'은 고정 기본값보다 오히려 예측이 1.3~1.4% 나빠졌습니다 (과거 우위가 지속되지 않음).",
        "    이 비교는 종목 특성을 이해하는 용도로 쓰세요.",
    ]
    return "\n".join(out)
