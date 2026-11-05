"""예측 파이프라인: 분석 → 보정 → 시나리오 → 시뮬레이션 → 요약/흐름 판정."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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

from .analysis import TRADING_DAYS, MarketStats, analyze
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
CHECKPOINTS = (("1주", 5), ("1개월", 21), ("3개월", 63), ("6개월", 126), ("1년", 252), ("2년", 504))
MOVE_THRESHOLD = 0.20  # ±20% 도달 확률 계산 기준


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


def future_trading_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    offset = CustomBusinessDay(calendar=NYSECalendar())
    return pd.date_range(last_date + offset, periods=horizon, freq=offset)


@dataclass
class ScenarioResult:
    scenario: Scenario
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
            "key": self.scenario.key,
            "name": self.scenario.name,
            "probability": self.scenario.probability,
            "expected_price": self.terminal_mean,
            "expected_return": self.terminal_mean / s0 - 1.0,
            "terminal_quantiles": {f"p{int(q * 100)}": v for q, v in self.terminal_quantiles.items()},
            "prob_up": self.prob_up,
            "prob_hit_up_20": self.prob_hit_up,
            "prob_hit_down_20": self.prob_hit_down,
            "median_max_drawdown": self.median_max_drawdown,
        }


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

    @property
    def s0(self) -> float:
        return self.stats.last_price

    def checkpoint_table(self) -> pd.DataFrame:
        """체크포인트별 시나리오 중앙값과 혼합 분포 분위수."""
        rows = []
        for label, day in self.checkpoints:
            row = {"기간": label, "날짜": self.dates[day].strftime("%Y-%m-%d")}
            for res in self.results:
                row[res.scenario.name] = float(res.quantiles[0.5][day])
            for q in QUANTILES:
                row[f"전체 p{int(q * 100)}"] = float(self.mix_quantiles[q][day])
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        s0 = self.s0
        return {
            "symbol": self.symbol,
            "as_of": self.stats.last_date.strftime("%Y-%m-%d"),
            "last_price": s0,
            "horizon_days": self.horizon,
            "horizon_end": self.dates[-1].strftime("%Y-%m-%d"),
            "n_paths_per_scenario": self.n_paths,
            "stats": self.stats.to_dict(),
            "calibration": self.calib.to_dict(),
            "scenarios": [sc.to_dict() for sc in self.scenarios],
            "scenario_results": [r.summary_dict(s0) for r in self.results],
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
            "checkpoints": self.checkpoint_table().to_dict(orient="records"),
            "path": {
                "dates": [d.strftime("%Y-%m-%d") for d in self.dates],
                "mixture": {f"p{int(q * 100)}": self.mix_quantiles[q].round(4).tolist() for q in QUANTILES},
                "scenario_median": {
                    r.scenario.key: r.quantiles[0.5].round(4).tolist() for r in self.results
                },
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
    tilt: bool = True,
    seed: int | None = 42,
    keep_paths: bool = False,
) -> Forecast:
    """한 종목의 시나리오별 예측을 실행합니다."""
    spec = scenario_spec_with_fields(spec or load_spec(None))
    stats = analyze(prices, symbol=symbol, benchmark=benchmark)
    calib = calibrate(stats, risk_free=risk_free, erp=erp, trend_weight=trend_weight, vol_halflife=vol_halflife)
    scenarios = build_scenarios(spec, calib)
    assign_probabilities(
        scenarios,
        stats.trend_score,
        tilt_strength=float(spec["tilt_strength"]),
        fixed=bool(spec["fixed_probabilities"]) or not tilt,
    )

    rng = np.random.default_rng(seed)
    s0 = stats.last_price
    up_level, down_level = s0 * (1 + MOVE_THRESHOLD), s0 * (1 - MOVE_THRESHOLD)

    results: list[ScenarioResult] = []
    all_paths: list[np.ndarray] = []
    for sc in scenarios:
        paths = simulate_paths(s0, sc, calib, horizon, n_paths, rng)
        q = np.quantile(paths, QUANTILES, axis=0)
        terminal = paths[:, -1]
        results.append(
            ScenarioResult(
                scenario=sc,
                quantiles={qq: q[i] for i, qq in enumerate(QUANTILES)},
                mean_path=paths.mean(axis=0),
                terminal_mean=float(terminal.mean()),
                terminal_quantiles={qq: float(q[i, -1]) for i, qq in enumerate(QUANTILES)},
                prob_up=float((terminal > s0).mean()),
                prob_hit_up=float((paths.max(axis=1) >= up_level).mean()),
                prob_hit_down=float((paths.min(axis=1) <= down_level).mean()),
                median_max_drawdown=float(np.median(path_max_drawdown(paths))),
                terminal=terminal.copy(),
            )
        )
        all_paths.append(paths)

    # 혼합(확률가중) 분포
    stacked = np.vstack(all_paths)
    weights = np.concatenate([np.full(n_paths, sc.probability / n_paths) for sc in scenarios])
    mixq = weighted_quantiles(stacked, weights, QUANTILES)
    mix_quantiles = {q: mixq[i] for i, q in enumerate(QUANTILES)}
    mix_mean_path = (weights[:, None] * stacked).sum(axis=0)
    terminal_all = stacked[:, -1]
    mix_terminal_q = {q: float(mixq[i, -1]) for i, q in enumerate(QUANTILES)}
    mix_prob_up = float(weights @ (terminal_all > s0))
    mix_hit_up = float(weights @ (stacked.max(axis=1) >= up_level))
    mix_hit_down = float(weights @ (stacked.min(axis=1) <= down_level))
    if not keep_paths:
        del stacked, all_paths

    dates = pd.DatetimeIndex([stats.last_date]).append(future_trading_days(stats.last_date, horizon))
    checkpoints = [(lbl, d) for lbl, d in CHECKPOINTS if d < horizon] + [(horizon_label(horizon), horizon)]

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
        mix_terminal_quantiles=mix_terminal_q,
        mix_expected_price=float(mix_mean_path[-1]),
        mix_prob_up=mix_prob_up,
        mix_prob_hit_up=mix_hit_up,
        mix_prob_hit_down=mix_hit_down,
        outlook=outlook,
        checkpoints=checkpoints,
    )


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
