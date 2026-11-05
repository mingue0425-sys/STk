"""몬테카를로 경로 시뮬레이션.

일간 로그수익률 모델 (국면별 파라미터가 날짜마다 바뀜):

    r_t = (μ_t − ½σ_t²)·Δt + σ_t·√Δt·ε_t + J_t

- μ_t  : 시나리오 국면의 연 기대수익률
- σ_t  : 현재 변동성에서 장기 변동성으로 지수적으로 회귀하는 경로 × 국면 변동성 배수
- ε_t  : 분산 1로 정규화한 Student-t 난수 (두꺼운 꼬리)
- J_t  : 복합 포아송 점프 (급등락). 기대수익률에 보정하지 않으므로 점프는 추가 충격입니다.
- 이벤트일(실적 발표 등): 지정한 날짜에 평균 0(가격 기준)인 추가 충격 N(−½s², s²)
"""

from __future__ import annotations

import math

import numpy as np

from .analysis import TRADING_DAYS
from .scenarios import Calibration, Scenario

DT = 1.0 / TRADING_DAYS


def phase_lengths(fracs: list[float], horizon: int) -> list[int]:
    """국면 비중을 거래일 수로 나눕니다 (합 = horizon, 반올림 오차는 마지막 국면에)."""
    lengths = [int(round(f * horizon)) for f in fracs[:-1]]
    lengths = [max(0, n) for n in lengths]
    used = sum(lengths)
    if used > horizon:
        # 앞 국면이 너무 길면 잘라냄
        excess = used - horizon
        for i in range(len(lengths) - 1, -1, -1):
            cut = min(lengths[i], excess)
            lengths[i] -= cut
            excess -= cut
        used = horizon
    lengths.append(horizon - used)
    return lengths


def daily_parameters(
    scenario: Scenario, calib: Calibration, horizon: int
) -> dict[str, np.ndarray]:
    """시나리오를 날짜별 파라미터 배열(길이 horizon)로 펼칩니다."""
    lengths = phase_lengths([ph.frac for ph in scenario.phases], horizon)
    mu = np.empty(horizon)
    vmult = np.empty(horizon)
    jrate = np.empty(horizon)
    jmean = np.empty(horizon)
    jstd = np.empty(horizon)
    start = 0
    for ph, n in zip(scenario.phases, lengths):
        sl = slice(start, start + n)
        mu[sl] = ph.drift
        vmult[sl] = ph.vol_mult
        jrate[sl] = ph.jump_rate
        jmean[sl] = ph.jump_mean
        jstd[sl] = ph.jump_std
        start += n

    # 변동성 기간구조: σ²(t) = σL² + (σ0² − σL²)·exp(−κt)
    kappa = math.log(2.0) / max(calib.vol_halflife, 1e-6)
    t = np.arange(horizon, dtype=float)
    var = calib.vol_long**2 + (calib.vol_now**2 - calib.vol_long**2) * np.exp(-kappa * t)
    sigma = np.sqrt(np.maximum(var, 1e-8)) * vmult
    return {"mu": mu, "sigma": sigma, "jump_rate": jrate, "jump_mean": jmean, "jump_std": jstd}


def simulate_paths(
    s0: float,
    scenario: Scenario,
    calib: Calibration,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
    event_days: dict[int, float] | None = None,
) -> np.ndarray:
    """가격 경로 배열 (n_paths, horizon + 1)을 반환합니다. 0열은 현재가.

    event_days: {경로상 일자 인덱스(0 = 첫 미래 거래일): 충격 표준편차(로그)}
    """
    if horizon < 1:
        raise ValueError("horizon 은 1 거래일 이상이어야 합니다")
    prm = daily_parameters(scenario, calib, horizon)
    mu, sigma = prm["mu"], prm["sigma"]

    df = calib.tail_df
    if math.isfinite(df) and df < 30:
        df = max(df, 2.5)
        eps = rng.standard_t(df, size=(n_paths, horizon)) * math.sqrt((df - 2.0) / df)
    else:
        eps = rng.standard_normal(size=(n_paths, horizon))

    log_ret = (mu - 0.5 * sigma**2) * DT + sigma * math.sqrt(DT) * eps

    if np.any(prm["jump_rate"] > 0):
        counts = rng.poisson(prm["jump_rate"] * DT, size=(n_paths, horizon))
        has = counts > 0
        if has.any():
            z = rng.standard_normal(size=int(has.sum()))
            n = counts[has]
            log_ret[has] += n * prm["jump_mean"][np.nonzero(has)[1]] + np.sqrt(n) * prm["jump_std"][
                np.nonzero(has)[1]
            ] * z

    if event_days:
        for day, std in event_days.items():
            if 0 <= day < horizon and std > 0:
                log_ret[:, day] += std * rng.standard_normal(n_paths) - 0.5 * std**2

    paths = np.empty((n_paths, horizon + 1))
    paths[:, 0] = s0
    np.cumsum(log_ret, axis=1, out=paths[:, 1:])
    paths[:, 1:] = s0 * np.exp(paths[:, 1:])
    return paths


def path_max_drawdown(paths: np.ndarray) -> np.ndarray:
    """각 경로의 최대 낙폭 (음수)."""
    peak = np.maximum.accumulate(paths, axis=1)
    return (paths / peak - 1.0).min(axis=1)
