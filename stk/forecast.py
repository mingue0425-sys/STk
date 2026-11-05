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
from .data import check_prices
from .earnings import EarningsModel, fit_earnings
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
from .sector import SectorInfo, analyze_sector
from .simulation import path_max_drawdown, simulate_paths

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
LOG_QUANTILES = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 기록·추적용 (5%~95%)
CHECKPOINTS = (("1주", 5), ("1개월", 21), ("3개월", 63), ("6개월", 126), ("1년", 252), ("2년", 504))
MOVE_THRESHOLD = 0.20  # ±20% 도달 확률 계산 기준
MODELS = ("ensemble", "scenario", "regime")
DEFAULT_MODEL = "ensemble"
DEFAULT_REGIME_WEIGHT = 0.7   # 백테스트(10종목, 3·6개월)에서 정확도·구간 보정이 가장 균형 잡힌 값
DEFAULT_AUTO_EVENTS = False   # 추정 실적일을 시뮬레이션에 반영할지 (백테스트상 개선 없음 → 기본 끔, 표시만 함)
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
class EventSpec:
    """시뮬레이션에 넣는 이벤트(실적 발표 등) 하루 충격."""

    date: pd.Timestamp      # (예상) 거래일
    std: float              # 하루 충격 표준편차 (로그수익률)
    spread: int = 0         # 날짜 불확실성 ±거래일 (0 = 정확한 날짜)
    source: str = "user"    # "user" = 직접 지정, "auto" = 실적 발표일 자동 추정

    def __iter__(self):     # (date, std) 로 풀어 쓰는 기존 코드와 호환
        yield self.date
        yield self.std

    def label(self) -> str:
        kind = "실적 발표 추정" if self.source == "auto" else "이벤트"
        unc = f" ±{self.spread}거래일" if self.spread else ""
        return f"{self.date:%Y-%m-%d}{unc} {kind} (하루 충격 ±{self.std:.1%})"


@dataclass
class LevelStat:
    """목표가(위)·손절가(아래) 한 개에 대한 도달 분석 (전체 혼합 분포 기준)."""

    kind: str                  # "target" | "stop"
    level: float
    rel: float                 # 현재가 대비
    prob_touch: float          # 기간 중 한 번이라도 도달할 확률 (basis 기준)
    prob_end_beyond: float     # 만기 종가가 그 가격을 넘어서(목표가) / 밑돌아(손절가) 있을 확률
    median_days: float | None  # 도달한다면 걸리는 거래일 중앙값 (basis 기준)
    prob_by: dict[str, float] = field(default_factory=dict)  # 체크포인트별 누적 도달 확률 (basis 기준)
    prob_touch_close: float | None = None  # 종가 기준 도달 확률 (비교용)
    basis: str = "intraday"    # "intraday": 장중(브라운 브리지 보정) / "close": 종가만

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
    events: list[EventSpec] = field(default_factory=list)
    earnings: EarningsModel | None = None
    next_earnings: list[pd.Timestamp] = field(default_factory=list)  # 예측 기간 안의 추정 실적일 (표시용)
    auto_events: bool = False                                          # 추정 실적일을 시뮬레이션에 반영했는지
    sector: SectorInfo | None = None
    mix_samples: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)  # (만기가격, 가중치)
    levels: list[LevelStat] = field(default_factory=list)
    cp_quantiles: dict[int, list[float]] = field(default_factory=dict)  # 체크포인트별 5~95% 분위 가격
    data_warnings: list[str] = field(default_factory=list)  # 데이터 품질 경고
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
            "events": [{"date": e.date.strftime("%Y-%m-%d"), "std": e.std, "spread": e.spread, "source": e.source}
                       for e in self.events],
            "earnings": self.earnings.to_dict() if self.earnings is not None else None,
            "next_earnings": [d.strftime("%Y-%m-%d") for d in self.next_earnings],
            "auto_events": self.auto_events,
            "sector": self.sector.to_dict() if self.sector is not None else None,
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
            "data_warnings": self.data_warnings,
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
) -> tuple[list[tuple[int, int, float]], list[EventSpec]]:
    """[(날짜, 표준편차 또는 None) | EventSpec] → 시뮬레이션용 [(경로 인덱스, ±거래일, 표준편차)]. 기간 밖은 무시."""
    sim: list[tuple[int, int, float]] = []
    used: list[EventSpec] = []
    for item in events or []:
        if isinstance(item, EventSpec):
            spec = item
        else:
            date, std = item if isinstance(item, (tuple, list)) else (item, None)
            spec = EventSpec(pd.Timestamp(date), float(std) if std is not None else default_std)
        pos = int(dates.searchsorted(pd.Timestamp(spec.date)))  # dates[0] = 기준일
        if pos < 1 or pos >= len(dates):
            continue
        spec = EventSpec(dates[pos], spec.std, spec.spread, spec.source)
        sim.append((pos - 1, spec.spread, spec.std))
        used.append(spec)
    return sim, used


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
    today: pd.Timestamp | None = None,
    auto_events: bool = DEFAULT_AUTO_EVENTS,
    earnings_info: bool = True,
    sectors: dict | None = None,
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
    earnings = None
    if (auto_events or earnings_info) and isinstance(prices, pd.DataFrame):
        earnings = fit_earnings(prices, benchmark if isinstance(benchmark, pd.DataFrame) else None)
    use_earn = auto_events and earnings is not None   # 시뮬레이션 반영 여부 (아니면 표시만)
    exclude = earnings.past_events if use_earn else None
    stats = analyze(prices, symbol=symbol, benchmark=benchmark, exclude_dates=exclude)
    data_warnings = check_prices(prices, today=today)
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
    all_events: list = list(events or [])
    if use_earn:
        user_dates = {pd.Timestamp(e.date if isinstance(e, EventSpec) else (e[0] if isinstance(e, (tuple, list)) else e))
                      for e in all_events}
        for d in earnings.next_dates(stats.last_date, dates[-1]):
            if all(abs((d - u).days) > 7 for u in user_dates):  # 직접 지정한 이벤트가 있으면 그쪽 우선
                all_events.append(EventSpec(d, earnings.event_std, earnings.spread_days, "auto"))
    default_std = earnings.event_std if use_earn else (estimate_event_std(prices) if events else 0.0)
    event_days, used_events = resolve_events(all_events, dates, default_std)

    # ---- 시나리오 모델
    results: list[ScenarioResult] = []
    all_paths: list[np.ndarray] = []
    for sc in scenarios:
        paths = simulate_paths(s0, sc, calib, horizon, n_paths, rng, events=event_days)
        results.append(ScenarioResult(**vars(summarize_paths(paths, s0)), scenario=sc))
        all_paths.append(paths)

    # ---- 국면 모델
    w_r = {"scenario": 0.0, "regime": 1.0, "ensemble": min(max(regime_weight, 0.0), 1.0)}[model]
    regime: RegimeResult | None = None
    regime_paths = None
    if w_r > 0 or regime_info:
        rets_s = log_returns(_price_series(prices))
        if exclude:  # 실적 반응일은 이벤트로 따로 넣으므로 국면 학습에서 제외 (이중 계산 방지)
            rets_s = rets_s.drop(index=pd.DatetimeIndex(exclude).intersection(rets_s.index))
        rets = rets_s.to_numpy()
        try:
            state = build_regime_state(rets, calib.base_drift, init=regime_init)
        except ValueError:
            state = None  # 데이터 부족 → 시나리오 모델만 사용
        if state is not None:
            regime_paths = simulate_regime_paths(s0, state, horizon, n_paths, rng, events=event_days)
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
        earnings=earnings,
        next_earnings=(earnings.next_dates(stats.last_date, dates[-1]) if earnings is not None else []),
        auto_events=use_earn,
        sector=(analyze_sector(prices, benchmark, sectors, as_of=stats.last_date)
                if sectors and benchmark is not None else None),
        mix_samples=mix_samples,
        levels=levels,
        race=race,
        risk=risk,
        cp_quantiles=cp_quantiles,
        data_warnings=data_warnings,
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
    """경로별 처음 종가가 도달한 일자(1부터). 도달하지 않으면 inf."""
    hit = paths[:, 1:] >= level if upward else paths[:, 1:] <= level
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1).astype(float) + 1
    first[~any_hit] = np.inf
    return first


def crossing_probs(logp: np.ndarray, log_level: float, upward: bool, var: np.ndarray) -> np.ndarray:
    """하루 단위 구간마다 가격이 수준을 건드렸을 확률 (브라운 브리지).

    두 종가가 모두 수준 안쪽이면 P(장중 도달) = exp(−2·a·b / σ²), a·b 는 양 끝 종가와 수준 사이의 로그 거리.
    종가가 이미 넘었으면 1.
    """
    if upward:
        a, b = log_level - logp[:, :-1], log_level - logp[:, 1:]
    else:
        a, b = logp[:, :-1] - log_level, logp[:, 1:] - log_level
    with np.errstate(over="ignore", under="ignore"):
        p = np.exp(-2.0 * np.clip(a, 0, None) * np.clip(b, 0, None) / var)
    p[(a <= 0) | (b <= 0)] = 1.0
    return p


def _touch_curves(paths: np.ndarray, level: float, upward: bool, intraday: bool, chunk: int = 4000) -> np.ndarray:
    """경로별 '해당 일자까지 도달했을 확률' 누적 곡선 (n, horizon). 종가 기준이면 0/1."""
    n, t1 = paths.shape
    out = np.empty((n, t1 - 1))
    log_level = math.log(level)
    for st in range(0, n, chunk):
        blk = paths[st:st + chunk]
        if intraday:
            logp = np.log(blk)
            var = np.maximum(np.var(np.diff(logp, axis=1), axis=1, keepdims=True), 1e-10)
            p = crossing_probs(logp, log_level, upward, var)
            out[st:st + chunk] = 1.0 - np.exp(np.cumsum(np.log1p(-np.minimum(p, 1 - 1e-12)), axis=1))
            out[st:st + chunk][np.cumsum(p >= 1.0, axis=1) > 0] = 1.0
        else:
            hit = blk[:, 1:] >= level if upward else blk[:, 1:] <= level
            out[st:st + chunk] = np.maximum.accumulate(hit, axis=1).astype(float)
    return out


def _race(paths: np.ndarray, w: np.ndarray, target: float, stop: float, intraday: bool, chunk: int = 4000) -> dict:
    """목표가·손절가 중 어느 쪽에 먼저 닿는지 (같은 날 둘 다 닿을 가능성은 반반으로 나눔)."""
    tf = sf = nei = 0.0
    for st in range(0, len(paths), chunk):
        blk, wb = paths[st:st + chunk], w[st:st + chunk]
        logp = np.log(blk)
        if intraday:
            var = np.maximum(np.var(np.diff(logp, axis=1), axis=1, keepdims=True), 1e-10)
            pu = crossing_probs(logp, math.log(target), True, var)
            pd_ = crossing_probs(logp, math.log(stop), False, var)
        else:
            pu = (blk[:, 1:] >= target).astype(float)
            pd_ = (blk[:, 1:] <= stop).astype(float)
        surv_step = (1 - pu) * (1 - pd_)
        surv_before = np.concatenate([np.ones((len(blk), 1)), np.cumprod(surv_step, axis=1)[:, :-1]], axis=1)
        tf += float(wb @ (surv_before * (pu * (1 - pd_) + 0.5 * pu * pd_)).sum(axis=1))
        sf += float(wb @ (surv_before * (pd_ * (1 - pu) + 0.5 * pu * pd_)).sum(axis=1))
        nei += float(wb @ np.prod(surv_step, axis=1))
    return {"target": target, "stop": stop, "target_first": tf, "stop_first": sf, "neither": nei,
            "basis": "intraday" if intraday else "close"}


def level_analysis(
    paths: np.ndarray,
    w: np.ndarray,
    s0: float,
    targets: list[float],
    stops: list[float],
    checkpoints: list[tuple[str, int]],
    intraday: bool = True,
) -> tuple[list[LevelStat], dict | None]:
    """목표가·손절가 도달 분석. intraday=True 면 종가 사이 장중 도달 가능성(브라운 브리지)까지 반영."""
    stats: list[LevelStat] = []
    for kind, levels in (("target", targets), ("stop", stops)):
        for lv in levels:
            upward = lv >= s0  # 현재가보다 위면 상향 돌파, 아래면 하향 이탈로 판정
            curve = _touch_curves(paths, lv, upward, intraday)          # (n, horizon) 누적 도달 확률
            touch_by_day = w @ curve                                     # 일자별 누적 (전체)
            p_touch = float(touch_by_day[-1])
            med = None
            if p_touch > 1e-9:
                med = float(np.searchsorted(touch_by_day, 0.5 * p_touch) + 1)
            close_first = _first_touch(paths, lv, upward)
            end = paths[:, -1] >= lv if upward else paths[:, -1] <= lv
            stats.append(
                LevelStat(
                    kind=kind, level=lv, rel=lv / s0 - 1, prob_touch=p_touch, prob_end_beyond=float(w @ end),
                    median_days=med, prob_by={lbl: float(touch_by_day[d - 1]) for lbl, d in checkpoints},
                    prob_touch_close=float(w @ np.isfinite(close_first)), basis="intraday" if intraday else "close",
                )
            )
    race = None
    if targets and stops:
        race = _race(paths, w, targets[0], stops[0], intraday)
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
