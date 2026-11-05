"""예측 파이프라인.

    분석 → (시장 환경) → 보정 → 시나리오 확률 → 시나리오 몬테카를로
                                      └→ 국면(HMM) 모델 몬테카를로
    → 앙상블(시나리오 모델 × (1−w) + 국면 모델 × w) → 요약 / 전체 흐름 판정
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

from .analysis import TRADING_DAYS, MarketStats, _price_series, analyze, estimate_event_std, log_returns
from .market import MarketContext
from .regime import HMMParams, RegimeState, build_regime_state, simulate_regime_paths
from .scenarios import (
    DEFAULT_ERP,
    DEFAULT_TREND_WEIGHT,
    DEFAULT_VOL_HALFLIFE,
    Calibration,
    Scenario,
    assign_probabilities,
    build_scenarios,
    calibrate,
    load_spec,
    scenario_spec_with_fields,
)
from .simulation import path_max_drawdown, simulate_paths

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
LOG_QUANTILES = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 기록·추적용 (5%~95%)
CHECKPOINTS = (("1주", 5), ("1개월", 21), ("3개월", 63), ("6개월", 126), ("1년", 252), ("2년", 504))
MOVE_THRESHOLD = 0.20  # ±20% 도달 확률 계산 기준
MODELS = ("ensemble", "scenario", "regime")
DEFAULT_MODEL = "ensemble"
DEFAULT_REGIME_WEIGHT = 0.7   # 백테스트(10종목, 3·6개월)에서 정확도·구간 보정이 가장 균형 잡힌 값
DEFAULT_TREND_ON = False      # 추세 점수로 시나리오 확률 조정 (백테스트상 평균적으로 도움이 안 돼 기본 끔)


class NYSECalendar(AbstractHolidayCalendar):
    """NYSE 정규 휴장일 근사 (임시 휴장 제외)."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@lru_cache(maxsize=1)
def _nyse_offset() -> CustomBusinessDay:
    holidays = NYSECalendar().holidays(start="1990-01-01", end="2045-12-31")
    return CustomBusinessDay(holidays=holidays)


def future_trading_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    offset = _nyse_offset()
    return pd.date_range(last_date + offset, periods=horizon, freq=offset)


@dataclass
class PathSummary:
    """한 묶음의 시뮬레이션 경로 요약."""

    quantiles: dict[float, np.ndarray]   # 날짜별 분위수 경로 (길이 horizon+1)
    mean_path: np.ndarray
    terminal_mean: float
    terminal_quantiles: dict[float, float]
    prob_up: float                       # 만기 가격 > 현재가 확률
    prob_hit_up: float                   # 기간 중 +20% 이상 도달 확률
    prob_hit_down: float                 # 기간 중 −20% 이하 도달 확률
    median_max_drawdown: float
    terminal: np.ndarray = field(repr=False)  # 만기 가격 표본

    def summary_dict(self, s0: float) -> dict:
        return {
            "expected_price": self.terminal_mean,
            "expected_return": self.terminal_mean / s0 - 1.0,
            "terminal_quantiles": {f"p{int(q * 100)}": v for q, v in self.terminal_quantiles.items()},
            "prob_up": self.prob_up,
            "prob_hit_up_20": self.prob_hit_up,
            "prob_hit_down_20": self.prob_hit_down,
            "median_max_drawdown": self.median_max_drawdown,
        }


def summarize_paths(paths: np.ndarray, s0: float) -> PathSummary:
    q = np.quantile(paths, QUANTILES, axis=0)
    terminal = paths[:, -1]
    return PathSummary(
        quantiles={qq: q[i] for i, qq in enumerate(QUANTILES)},
        mean_path=paths.mean(axis=0),
        terminal_mean=float(terminal.mean()),
        terminal_quantiles={qq: float(q[i, -1]) for i, qq in enumerate(QUANTILES)},
        prob_up=float((terminal > s0).mean()),
        prob_hit_up=float((paths.max(axis=1) >= s0 * (1 + MOVE_THRESHOLD)).mean()),
        prob_hit_down=float((paths.min(axis=1) <= s0 * (1 - MOVE_THRESHOLD)).mean()),
        median_max_drawdown=float(np.median(path_max_drawdown(paths))),
        terminal=terminal.copy(),
    )


@dataclass
class ScenarioResult(PathSummary):
    scenario: Scenario = None  # type: ignore[assignment]

    def summary_dict(self, s0: float) -> dict:
        out = {"key": self.scenario.key, "name": self.scenario.name, "probability": self.scenario.probability}
        out.update(super().summary_dict(s0))
        return out


@dataclass
class RegimeResult(PathSummary):
    state: RegimeState = None  # type: ignore[assignment]
    weight: float = 0.0        # 앙상블에서의 비중

    def summary_dict(self, s0: float) -> dict:
        out = {"weight": self.weight, **self.state.to_dict()}
        out.update(super().summary_dict(s0))
        return out


@dataclass
class LevelStat:
    """목표가(위)·손절가(아래) 한 개에 대한 도달 분석 (전체 혼합 분포 기준)."""

    kind: str                  # "target" | "stop"
    level: float
    rel: float                 # 현재가 대비
    prob_touch: float          # 기간 중 한 번이라도 도달할 확률
    prob_end_beyond: float     # 만기에 그 가격을 넘어서(목표가) / 밑돌아(손절가) 있을 확률
    median_days: float | None  # 도달한 경로 기준 도달까지 걸린 거래일 중앙값
    prob_by: dict[str, float] = field(default_factory=dict)  # 체크포인트별 누적 도달 확률

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskStats:
    """만기 손익 분포의 위험 지표 (전체 혼합 분포 기준)."""

    var95: float               # 95% VaR: 5% 확률로 이보다 더 손실 (양수 = 손실률)
    cvar95: float              # 95% CVaR: 최악 5% 경우의 평균 손실률
    prob_loss_10: float        # 만기 손실 10% 이상 확률
    prob_loss_20: float
    mdd_median: float          # 경로 중 최대낙폭 중앙값 (음수)
    mdd_p95: float             # 최대낙폭 하위 5% (나쁜 경우)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Outlook:
    """확률가중(혼합) 분포로 본 전체 흐름 판정."""

    label: str            # 예: "완만한 상승 흐름"
    direction: int        # +2, +1, 0, -1, -2
    shape: str            # 경로 모양 설명
    volatility: str       # 변동성 수준 설명
    dominant: str         # 가장 확률이 높은 시나리오 이름
    expected_return: float
    median_return: float
    prob_up: float
    band90: tuple[float, float]  # 만기 90% 구간 (가격)


@dataclass
class Forecast:
    symbol: str
    stats: MarketStats
    calib: Calibration
    scenarios: list[Scenario]
    horizon: int
    n_paths: int
    dates: pd.DatetimeIndex                 # 현재 날짜 + 미래 거래일 (길이 horizon+1)
    results: list[ScenarioResult]
    mix_quantiles: dict[float, np.ndarray]  # 혼합 분포 날짜별 분위수
    mix_mean_path: np.ndarray
    mix_terminal_quantiles: dict[float, float]
    mix_expected_price: float
    mix_prob_up: float
    mix_prob_hit_up: float
    mix_prob_hit_down: float
    outlook: Outlook
    checkpoints: list[tuple[str, int]]
    model: str = DEFAULT_MODEL
    trend_tilt: bool = DEFAULT_TREND_ON
    regime: RegimeResult | None = None
    market: MarketContext | None = None
    events: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    mix_samples: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)  # (만기가격, 가중치)
    levels: list[LevelStat] = field(default_factory=list)
    cp_quantiles: dict[int, list[float]] = field(default_factory=dict)  # 체크포인트별 5~95% 분위 가격
    race: dict | None = None             # 목표가·손절가 중 먼저 도달할 확률
    risk: RiskStats | None = None

    @property
    def s0(self) -> float:
        return self.stats.last_price

    @property
    def regime_weight(self) -> float:
        return self.regime.weight if self.regime is not None else 0.0

    def checkpoint_table(self) -> pd.DataFrame:
        """체크포인트별 시나리오 중앙값과 혼합 분포 분위수."""
        rows = []
        for label, day in self.checkpoints:
            row = {"기간": label, "날짜": self.dates[day].strftime("%Y-%m-%d")}
            for res in self.results:
                row[res.scenario.name] = float(res.quantiles[0.5][day])
            if self.regime is not None:
                row["국면 모델"] = float(self.regime.quantiles[0.5][day])
            for q in QUANTILES:
                row[f"전체 p{int(q * 100)}"] = float(self.mix_quantiles[q][day])
            rows.append(row)
        return pd.DataFrame(rows)

    def regime_flow(self) -> list[dict]:
        """체크포인트별 국면 확률 (국면 흐름)."""
        if self.regime is None:
            return []
        st = self.regime.state
        rows = [{"기간": "현재", **{n: float(p) for n, p in zip(st.names, st.current)}}]
        for label, day in self.checkpoints:
            rows.append({"기간": label, **{n: float(p) for n, p in zip(st.names, st.occupancy(day))}})
        return rows

    def to_dict(self) -> dict:
        s0 = self.s0
        return {
            "symbol": self.symbol,
            "as_of": self.stats.last_date.strftime("%Y-%m-%d"),
            "last_price": s0,
            "horizon_days": self.horizon,
            "horizon_end": self.dates[-1].strftime("%Y-%m-%d"),
            "n_paths_per_scenario": self.n_paths,
            "model": self.model,
            "regime_weight": self.regime_weight,
            "trend_tilt": self.trend_tilt,
            "stats": self.stats.to_dict(),
            "calibration": self.calib.to_dict(),
            "market": self.market.to_dict() if self.market is not None else None,
            "events": [{"date": d.strftime("%Y-%m-%d"), "std": s} for d, s in self.events],
            "scenarios": [sc.to_dict() for sc in self.scenarios],
            "scenario_results": [r.summary_dict(s0) for r in self.results],
            "regime": self.regime.summary_dict(s0) if self.regime is not None else None,
            "regime_flow": self.regime_flow(),
            "overall": {
                "label": self.outlook.label,
                "direction": self.outlook.direction,
                "shape": self.outlook.shape,
                "volatility": self.outlook.volatility,
                "dominant_scenario": self.outlook.dominant,
                "expected_price": self.mix_expected_price,
                "expected_return": self.mix_expected_price / s0 - 1.0,
                "terminal_quantiles": {f"p{int(q * 100)}": v for q, v in self.mix_terminal_quantiles.items()},
                "prob_up": self.mix_prob_up,
                "prob_hit_up_20": self.mix_prob_hit_up,
                "prob_hit_down_20": self.mix_prob_hit_down,
            },
            "risk": self.risk.to_dict() if self.risk is not None else None,
            "levels": [lv.to_dict() for lv in self.levels],
            "race": self.race,
            "checkpoints": self.checkpoint_table().to_dict(orient="records"),
            "path": {
                "dates": [d.strftime("%Y-%m-%d") for d in self.dates],
                "mixture": {f"p{int(q * 100)}": self.mix_quantiles[q].round(4).tolist() for q in QUANTILES},
                "scenario_median": {
                    r.scenario.key: r.quantiles[0.5].round(4).tolist() for r in self.results
                },
                "regime_median": self.regime.quantiles[0.5].round(4).tolist() if self.regime else None,
            },
        }


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs: tuple[float, ...]) -> np.ndarray:
    """열(column)별 가중 분위수. values: (n,) 또는 (n, T) → 반환 (len(qs),) 또는 (len(qs), T)."""
    vals = np.asarray(values, dtype=float)
    squeeze = vals.ndim == 1
    if squeeze:
        vals = vals[:, None]
    w = np.asarray(weights, dtype=float)
    order = np.argsort(vals, axis=0)
    sorted_vals = np.take_along_axis(vals, order, axis=0)
    cw = np.cumsum(w[order], axis=0)
    cw /= cw[-1:, :]
    out = np.empty((len(qs), vals.shape[1]))
    for i, q in enumerate(qs):
        idx = (cw < q).sum(axis=0)
        idx = np.minimum(idx, vals.shape[0] - 1)
        out[i] = sorted_vals[idx, np.arange(vals.shape[1])]
    return out[:, 0] if squeeze else out


def resolve_events(
    events: list | None, dates: pd.DatetimeIndex, default_std: float
) -> tuple[dict[int, float], list[tuple[pd.Timestamp, float]]]:
    """[(날짜, 표준편차 또는 None)] → {경로 인덱스: 표준편차}. 기간 밖의 이벤트는 무시."""
    day_map: dict[int, float] = {}
    used: list[tuple[pd.Timestamp, float]] = []
    for item in events or []:
        date, std = item if isinstance(item, (tuple, list)) else (item, None)
        date = pd.Timestamp(date)
        pos = int(dates.searchsorted(date))  # dates[0] = 기준일
        if pos < 1 or pos >= len(dates):
            continue
        s = float(std) if std is not None else default_std
        day_map[pos - 1] = day_map.get(pos - 1, 0.0) + s  # 같은 날 여러 이벤트는 합산(보수적)
        used.append((dates[pos], s))
    return day_map, used


def run_forecast(
    prices: pd.DataFrame | pd.Series,
    symbol: str,
    horizon: int = 126,
    n_paths: int = 10_000,
    benchmark: pd.DataFrame | pd.Series | None = None,
    risk_free: float = 0.04,
    erp: float = DEFAULT_ERP,
    trend_weight: float = DEFAULT_TREND_WEIGHT,
    vol_halflife: float = DEFAULT_VOL_HALFLIFE,
    spec: dict | None = None,
    tilt: bool = DEFAULT_TREND_ON,
    stress_tilt: bool = True,
    seed: int | None = 42,
    market: MarketContext | None = None,
    model: str = DEFAULT_MODEL,
    regime_weight: float = DEFAULT_REGIME_WEIGHT,
    regime_init: HMMParams | None = None,
    regime_info: bool = True,
    events: list | None = None,
    targets: list | None = None,
    stops: list | None = None,
    keep_paths: bool = False,
) -> Forecast:
    """한 종목의 예측을 실행합니다.

    model:
      - "scenario" : 시나리오 모델만
      - "regime"   : 국면(HMM) 모델만 (시나리오 표는 참고용으로 계산)
      - "ensemble" : 시나리오 모델 × (1 − regime_weight) + 국면 모델 × regime_weight
    events: [(날짜, 충격 표준편차 또는 None)] — None 이면 과거 데이터로 추정
    tilt: 추세 점수로 시나리오 확률 조정 / stress_tilt: 시장 스트레스로 시나리오 확률 조정
    """
    if model not in MODELS:
        raise ValueError(f"model 은 {MODELS} 중 하나여야 합니다")
    spec = scenario_spec_with_fields(spec or load_spec(None))
    stats = analyze(prices, symbol=symbol, benchmark=benchmark)
    calib = calibrate(
        stats, risk_free=risk_free, erp=erp, trend_weight=trend_weight, vol_halflife=vol_halflife,
        market=market, horizon=horizon,
    )
    scenarios = build_scenarios(spec, calib)
    assign_probabilities(
        scenarios,
        stats.trend_score if tilt else 0.0,
        tilt_strength=float(spec["tilt_strength"]),
        fixed=bool(spec["fixed_probabilities"]),
        stress=calib.market_stress if stress_tilt else 0.0,
        stress_strength=float(spec["stress_strength"]),
    )

    rng = np.random.default_rng(seed)
    s0 = stats.last_price
    dates = pd.DatetimeIndex([stats.last_date]).append(future_trading_days(stats.last_date, horizon))
    event_days, used_events = resolve_events(events, dates, estimate_event_std(prices) if events else 0.0)

    # ---- 시나리오 모델
    results: list[ScenarioResult] = []
    all_paths: list[np.ndarray] = []
    for sc in scenarios:
        paths = simulate_paths(s0, sc, calib, horizon, n_paths, rng, event_days=event_days)
        results.append(ScenarioResult(**vars(summarize_paths(paths, s0)), scenario=sc))
        all_paths.append(paths)

    # ---- 국면 모델
    w_r = {"scenario": 0.0, "regime": 1.0, "ensemble": min(max(regime_weight, 0.0), 1.0)}[model]
    regime: RegimeResult | None = None
    regime_paths = None
    if w_r > 0 or regime_info:
        rets = log_returns(_price_series(prices)).to_numpy()
        try:
            state = build_regime_state(rets, calib.base_drift, init=regime_init)
        except ValueError:
            state = None  # 데이터 부족 → 시나리오 모델만 사용
        if state is not None:
            regime_paths = simulate_regime_paths(s0, state, horizon, n_paths, rng, event_days=event_days)
            regime = RegimeResult(**vars(summarize_paths(regime_paths, s0)), state=state, weight=w_r)
    if regime is None:
        w_r = 0.0
    elif w_r == 0:
        regime_paths = None  # 참고 정보로만 사용

    # ---- 혼합(앙상블) 분포
    blocks = list(all_paths)
    weights = [np.full(n_paths, (1 - w_r) * sc.probability / n_paths) for sc in scenarios]
    if regime_paths is not None and w_r > 0:
        blocks.append(regime_paths)
        weights.append(np.full(n_paths, w_r / n_paths))
    stacked = np.vstack(blocks)
    w = np.concatenate(weights)
    keep = w > 0
    stacked, w = stacked[keep], w[keep]
    mixq = weighted_quantiles(stacked, w, QUANTILES)
    mix_quantiles = {q: mixq[i] for i, q in enumerate(QUANTILES)}
    mix_mean_path = (w[:, None] * stacked).sum(axis=0) / w.sum()
    terminal_all = stacked[:, -1]
    mix_prob_up = float(w @ (terminal_all > s0) / w.sum())
    mix_hit_up = float(w @ (stacked.max(axis=1) >= s0 * (1 + MOVE_THRESHOLD)) / w.sum())
    mix_hit_down = float(w @ (stacked.min(axis=1) <= s0 * (1 - MOVE_THRESHOLD)) / w.sum())
    mix_samples = (terminal_all.copy(), w / w.sum())
    wn = w / w.sum()
    checkpoints = [(lbl, d) for lbl, d in CHECKPOINTS if d < horizon] + [(horizon_label(horizon), horizon)]
    risk = risk_stats(stacked, wn, s0)
    cp_days = [d for _, d in checkpoints]
    cpq = weighted_quantiles(stacked[:, cp_days], wn, LOG_QUANTILES)
    cp_quantiles = {d: [float(v) for v in cpq[:, i]] for i, d in enumerate(cp_days)}
    levels, race = level_analysis(stacked, wn, s0, resolve_levels(targets, s0), resolve_levels(stops, s0), checkpoints)
    if not keep_paths:
        del stacked, all_paths, blocks

    outlook = judge_outlook(
        s0=s0,
        horizon=horizon,
        calib=calib,
        scenarios=scenarios,
        mix_quantiles=mix_quantiles,
        expected_price=float(mix_mean_path[-1]),
        prob_up=mix_prob_up,
    )

    return Forecast(
        symbol=symbol,
        stats=stats,
        calib=calib,
        scenarios=scenarios,
        horizon=horizon,
        n_paths=n_paths,
        dates=dates,
        results=results,
        mix_quantiles=mix_quantiles,
        mix_mean_path=mix_mean_path,
        mix_terminal_quantiles={q: float(mixq[i, -1]) for i, q in enumerate(QUANTILES)},
        mix_expected_price=float(mix_mean_path[-1]),
        mix_prob_up=mix_prob_up,
        mix_prob_hit_up=mix_hit_up,
        mix_prob_hit_down=mix_hit_down,
        outlook=outlook,
        checkpoints=checkpoints,
        model=model,
        trend_tilt=tilt,
        regime=regime,
        market=market,
        events=used_events,
        mix_samples=mix_samples,
        levels=levels,
        race=race,
        risk=risk,
        cp_quantiles=cp_quantiles,
    )


def resolve_levels(items: list | None, s0: float) -> list[float]:
    """[400, "+15%", "-10%", 0.9(배수 아님: 가격)] → 가격 목록. 문자열 % 는 현재가 대비."""
    out = []
    for it in items or []:
        if isinstance(it, str):
            t = it.strip()
            if t.endswith("%"):
                out.append(s0 * (1 + float(t[:-1]) / 100))
            else:
                out.append(float(t))
        else:
            out.append(float(it))
    return [v for v in out if v > 0]


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][min(int(np.searchsorted(cw, q * cw[-1])), len(x) - 1)])


def risk_stats(paths: np.ndarray, w: np.ndarray, s0: float) -> RiskStats:
    ret = paths[:, -1] / s0 - 1
    q05 = _wquantile(ret, w, 0.05)
    tail = ret <= q05
    cvar = -float((w[tail] * ret[tail]).sum() / w[tail].sum()) if tail.any() else -q05
    mdd = path_max_drawdown(paths)
    return RiskStats(
        var95=-q05,
        cvar95=cvar,
        prob_loss_10=float(w @ (ret <= -0.10)),
        prob_loss_20=float(w @ (ret <= -0.20)),
        mdd_median=_wquantile(mdd, w, 0.5),
        mdd_p95=_wquantile(mdd, w, 0.05),
    )


def _first_touch(paths: np.ndarray, level: float, upward: bool) -> np.ndarray:
    """경로별 처음 도달한 일자(1부터). 도달하지 않으면 inf."""
    hit = paths[:, 1:] >= level if upward else paths[:, 1:] <= level
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1).astype(float) + 1
    first[~any_hit] = np.inf
    return first


def level_analysis(
    paths: np.ndarray,
    w: np.ndarray,
    s0: float,
    targets: list[float],
    stops: list[float],
    checkpoints: list[tuple[str, int]],
) -> tuple[list[LevelStat], dict | None]:
    stats: list[LevelStat] = []
    firsts: dict[str, np.ndarray] = {}
    for kind, levels in (("target", targets), ("stop", stops)):
        for lv in levels:
            upward = lv >= s0  # 현재가보다 위면 상향 돌파, 아래면 하향 이탈로 판정
            first = _first_touch(paths, lv, upward)
            touched = np.isfinite(first)
            p_touch = float(w @ touched)
            med = _wquantile(first[touched], w[touched], 0.5) if touched.any() else None
            end = paths[:, -1] >= lv if upward else paths[:, -1] <= lv
            stats.append(
                LevelStat(
                    kind=kind, level=lv, rel=lv / s0 - 1, prob_touch=p_touch, prob_end_beyond=float(w @ end),
                    median_days=med, prob_by={lbl: float(w @ (first <= d)) for lbl, d in checkpoints},
                )
            )
            firsts.setdefault(kind, first)  # 경쟁 분석은 첫 번째 목표가·손절가로
    race = None
    if "target" in firsts and "stop" in firsts:
        t, st = firsts["target"], firsts["stop"]
        race = {
            "target": stats[0].level,
            "stop": next(x.level for x in stats if x.kind == "stop"),
            "target_first": float(w @ (t < st)),
            "stop_first": float(w @ (st < t)),
            "neither": float(w @ (~np.isfinite(t) & ~np.isfinite(st))),
        }
    return stats, race


def horizon_label(days: int) -> str:
    for label, d in CHECKPOINTS:
        if d == days:
            return label
    if days % 21 == 0:
        return f"{days // 21}개월"
    return f"{days}거래일"


def judge_outlook(
    s0: float,
    horizon: int,
    calib: Calibration,
    scenarios: list[Scenario],
    mix_quantiles: dict[float, np.ndarray],
    expected_price: float,
    prob_up: float,
) -> Outlook:
    """혼합 분포로 전체 흐름(방향·경로 모양·변동성)을 판정합니다."""
    median = mix_quantiles[0.5]
    exp_ret = expected_price / s0 - 1.0
    med_ret = float(median[-1] / s0 - 1.0)

    if prob_up >= 0.65 and exp_ret > 0:
        label, direction = "뚜렷한 상승 흐름", 2
    elif prob_up >= 0.55:
        label, direction = "완만한 상승 흐름", 1
    elif prob_up > 0.45:
        label, direction = "방향성 약한 횡보 흐름", 0
    elif prob_up > 0.35:
        label, direction = "완만한 하락 흐름", -1
    else:
        label, direction = "뚜렷한 하락 흐름", -2

    # 경로 모양: 중앙값 경로의 저점/고점 위치
    rel = median / s0 - 1.0
    i_min, i_max = int(np.argmin(rel)), int(np.argmax(rel))
    end = rel[-1]
    dip_tol = max(0.02, 0.15 * calib.vol_long * math.sqrt(horizon / TRADING_DAYS))
    if rel[i_min] < -dip_tol and 0 < i_min < 0.8 * horizon and end - rel[i_min] > dip_tol:
        shape = f"초반 조정(저점 약 {rel[i_min]:+.1%}, {i_min}거래일째) 후 회복하는 경로"
    elif rel[i_max] > dip_tol and 0 < i_max < 0.8 * horizon and rel[i_max] - end > dip_tol:
        shape = f"초반 상승(고점 약 {rel[i_max]:+.1%}) 후 되돌림이 나오는 경로"
    elif end > dip_tol:
        shape = "꾸준히 우상향하는 경로"
    elif end < -dip_tol:
        shape = "꾸준히 우하향하는 경로"
    else:
        shape = "현재가 부근에서 등락하는 경로"

    lo, hi = float(mix_quantiles[0.05][-1]), float(mix_quantiles[0.95][-1])
    width = (hi - lo) / s0
    ann_width = width / math.sqrt(max(horizon, 1) / TRADING_DAYS)
    if ann_width > 1.2:
        vol_txt = "매우 높음"
    elif ann_width > 0.8:
        vol_txt = "높음"
    elif ann_width > 0.5:
        vol_txt = "보통"
    else:
        vol_txt = "낮음"
    volatility = f"{vol_txt} (90% 구간 폭 {width:.0%})"

    dominant = max(scenarios, key=lambda s: s.probability).name
    return Outlook(
        label=label,
        direction=direction,
        shape=shape,
        volatility=volatility,
        dominant=dominant,
        expected_return=exp_ret,
        median_return=med_ret,
        prob_up=prob_up,
        band90=(lo, hi),
    )
