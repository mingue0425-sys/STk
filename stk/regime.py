"""데이터 기반 국면(regime) 모델: 가우시안 은닉 마르코프 모델(HMM).

일간 로그수익률이 관측되지 않는 K개 국면(저변동·중간 변동·고변동) 중 하나에서
생성되고, 국면은 마르코프 연쇄로 전환된다고 가정합니다.

- Baum-Welch(EM) 알고리즘으로 국면별 평균·분산과 전환확률을 추정합니다.
- 마지막 날의 필터링 확률 = '현재 어떤 국면에 있는가'
- 전환행렬의 거듭제곱 = '앞으로 각 국면에 있을 확률' (국면 흐름)
- 국면 전환 몬테카를로 = 시나리오를 사람이 정하지 않는 데이터 기반 예측 분포

국면별 평균수익률은 표본 기간의 운(예: 특정 기간의 급등)에 크게 좌우되므로,
국면 간 '차이'를 추정 정밀도만큼만 반영하고(경험적 베이즈 축소: 표본이 적고 변동성이 큰
국면일수록 0 쪽으로), 전체 수준은 보정된 기본 기대수익률에 맞춥니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .analysis import TRADING_DAYS

REGIME_NAMES = {
    2: ["저변동(안정)", "고변동(불안)"],
    3: ["저변동(안정)", "중간 변동", "고변동(불안)"],
}
MIN_DAILY_VAR = 0.002**2
DRIFT_PRIOR_SD = 0.10  # 국면 간 기대수익률 차이의 사전 표준편차 (연)


@dataclass
class HMMParams:
    means: np.ndarray    # (K,) 일간 로그수익률 평균
    vars: np.ndarray     # (K,) 일간 분산
    trans: np.ndarray    # (K, K) 전환확률 (행: 오늘, 열: 내일)
    init: np.ndarray     # (K,)
    loglik: float = float("nan")
    n_iter: int = 0

    @property
    def k(self) -> int:
        return len(self.means)

    def stationary(self) -> np.ndarray:
        vals, vecs = np.linalg.eig(self.trans.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
        v = np.abs(v)
        return v / v.sum()


@dataclass
class RegimeState:
    """적합 결과 + 현재 국면 + 예측용 파라미터."""

    params: HMMParams
    names: list[str]
    current: np.ndarray        # (K,) 현재 국면 확률 (필터링)
    drift: np.ndarray          # (K,) 보정 후 연 기대수익률
    vol: np.ndarray            # (K,) 연 변동성
    raw_drift: np.ndarray      # (K,) 보정 전(표본) 연 기대수익률
    duration: np.ndarray       # (K,) 평균 지속기간 (거래일)
    stationary: np.ndarray     # (K,) 장기 비중
    n_obs: int

    @property
    def current_regime(self) -> int:
        return int(np.argmax(self.current))

    def occupancy(self, days: int) -> np.ndarray:
        """days 거래일 뒤 각 국면에 있을 확률."""
        return self.current @ np.linalg.matrix_power(self.params.trans, int(days))

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "current": self.current.tolist(),
            "current_regime": self.names[self.current_regime],
            "drift": self.drift.tolist(),
            "raw_drift": self.raw_drift.tolist(),
            "vol": self.vol.tolist(),
            "duration_days": self.duration.tolist(),
            "stationary": self.stationary.tolist(),
            "transition": self.params.trans.tolist(),
            "loglik": self.params.loglik,
            "n_obs": self.n_obs,
        }


# ----------------------------------------------------------------- HMM 핵심
def _emissions(r: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    d = r[:, None] - means[None, :]
    b = np.exp(-0.5 * d * d / vars_[None, :]) / np.sqrt(2 * np.pi * vars_[None, :])
    return np.maximum(b, 1e-300)


def forward_backward(r: np.ndarray, p: HMMParams):
    """스케일링한 전향-후향 알고리즘 → (gamma, xi 합, 로그우도, 마지막 필터링 확률)."""
    t_len, k = len(r), p.k
    b = _emissions(r, p.means, p.vars)
    a = p.trans
    alpha = np.empty((t_len, k))
    c = np.empty(t_len)
    prev = p.init * b[0]
    c[0] = prev.sum()
    alpha[0] = prev / c[0]
    for t in range(1, t_len):
        prev = (alpha[t - 1] @ a) * b[t]
        c[t] = prev.sum()
        alpha[t] = prev / c[t]
    beta = np.empty((t_len, k))
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = a @ (b[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    scaled = b[1:] * beta[1:] / c[1:, None]
    xi_sum = (alpha[:-1].T @ scaled) * a
    return gamma, xi_sum, float(np.log(c).sum()), alpha[-1]


def filter_last(r: np.ndarray, p: HMMParams) -> np.ndarray:
    """마지막 시점의 필터링 확률 (전향 알고리즘만)."""
    b = _emissions(r, p.means, p.vars)
    prev = p.init * b[0]
    prev /= prev.sum()
    for t in range(1, len(r)):
        prev = (prev @ p.trans) * b[t]
        prev /= prev.sum()
    return prev


def _initial_params(r: np.ndarray, k: int) -> HMMParams:
    win = 21
    rv = np.sqrt(np.convolve(r * r, np.ones(win) / win, mode="same"))
    edges = np.quantile(rv, np.linspace(0, 1, k + 1))
    labels = np.clip(np.searchsorted(edges, rv, side="right") - 1, 0, k - 1)
    means = np.array([r[labels == i].mean() if np.any(labels == i) else r.mean() for i in range(k)])
    vars_ = np.array([r[labels == i].var() if np.sum(labels == i) > 2 else r.var() for i in range(k)])
    trans = np.full((k, k), 0.05 / (k - 1))
    np.fill_diagonal(trans, 0.95)
    return HMMParams(means=means, vars=np.maximum(vars_, MIN_DAILY_VAR), trans=trans, init=np.full(k, 1.0 / k))


def _sorted_by_vol(p: HMMParams) -> HMMParams:
    order = np.argsort(p.vars)
    return HMMParams(
        means=p.means[order],
        vars=p.vars[order],
        trans=p.trans[np.ix_(order, order)],
        init=p.init[order],
        loglik=p.loglik,
        n_iter=p.n_iter,
    )


def fit_hmm(
    returns: np.ndarray,
    k: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    init: HMMParams | None = None,
) -> HMMParams:
    """Baum-Welch 로 HMM 을 적합합니다. init 을 주면 그 값에서 시작(warm start)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 50 * k:
        raise ValueError(f"국면 모델 적합에 데이터가 부족합니다 ({len(r)}개)")
    p = init if init is not None and init.k == k else _initial_params(r, k)
    p = HMMParams(p.means.copy(), p.vars.copy(), p.trans.copy(), p.init.copy())
    prev_ll = -np.inf
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        gamma, xi_sum, ll, _ = forward_backward(r, p)
        w = gamma.sum(axis=0) + 1e-12
        means = (gamma * r[:, None]).sum(axis=0) / w
        vars_ = (gamma * (r[:, None] - means[None, :]) ** 2).sum(axis=0) / w
        trans = xi_sum + 1e-3  # 작은 사전 카운트로 퇴화 방지
        trans /= trans.sum(axis=1, keepdims=True)
        p = HMMParams(means, np.maximum(vars_, MIN_DAILY_VAR), trans, gamma[0] * 0.9 + 0.1 / k)
        if abs(ll - prev_ll) < tol * max(1.0, abs(ll)):
            prev_ll = ll
            break
        prev_ll = ll
    p.loglik, p.n_iter = prev_ll, n_iter
    return _sorted_by_vol(p)


# ----------------------------------------------------------------- 예측용 구성
def build_regime_state(
    returns: np.ndarray,
    base_drift: float,
    k: int = 3,
    init: HMMParams | None = None,
    prior_sd: float = DRIFT_PRIOR_SD,
    max_obs: int = 10 * TRADING_DAYS,
) -> RegimeState:
    """HMM 적합 → 현재 국면, 국면별 (보정) 기대수익률·변동성."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)][-max_obs:]
    params = fit_hmm(r, k=k, init=init, max_iter=200 if init is None else 50)
    current = filter_last(r, params)
    vol = np.sqrt(params.vars * TRADING_DAYS)
    raw = params.means * TRADING_DAYS + 0.5 * vol**2  # 산술 연 기대수익률
    pi = params.stationary()
    drift = base_drift + shrink_regime_drifts(raw, vol, pi, len(r), prior_sd)
    duration = 1.0 / np.maximum(1.0 - np.diag(params.trans), 1e-6)
    return RegimeState(
        params=params,
        names=list(REGIME_NAMES.get(k, [f"국면{i + 1}" for i in range(k)])),
        current=current,
        drift=drift,
        vol=vol,
        raw_drift=raw,
        duration=duration,
        stationary=pi,
        n_obs=len(r),
    )


def shrink_regime_drifts(
    raw: np.ndarray, vol: np.ndarray, pi: np.ndarray, n_obs: int, prior_sd: float = DRIFT_PRIOR_SD
) -> np.ndarray:
    """국면별 (표본 평균 − 전체 평균)을 정밀도 가중으로 축소하고, 장기 평균이 0 이 되게 재중심화."""
    diff = raw - float(pi @ raw)
    n_k = np.maximum(pi * n_obs, 1.0)
    se2 = vol**2 * TRADING_DAYS / n_k          # 연율 평균 추정치의 분산
    weight = 1.0 / (1.0 + se2 / prior_sd**2)
    shrunk = diff * weight
    return shrunk - float(pi @ shrunk)


def simulate_regime_paths(
    s0: float,
    state: RegimeState,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
    events: list[tuple[int, int, float]] | None = None,
) -> np.ndarray:
    """국면 전환 몬테카를로. 반환: (n_paths, horizon + 1) 가격 경로."""
    a = state.params.trans
    cum = np.cumsum(a, axis=1)
    cum[:, -1] = 1.0
    cur = rng.choice(state.params.k, size=n_paths, p=state.current / state.current.sum())
    states = np.empty((n_paths, horizon), dtype=np.int8)
    for t in range(horizon):
        u = rng.random(n_paths)
        cur = (u[:, None] > cum[cur]).sum(axis=1)
        states[:, t] = cur
    dt = 1.0 / TRADING_DAYS
    mu_d = (state.drift - 0.5 * state.vol**2) * dt
    sd_d = state.vol * math.sqrt(dt)
    log_ret = mu_d[states] + sd_d[states] * rng.standard_normal((n_paths, horizon))
    from .simulation import apply_events

    apply_events(log_ret, events, rng)
    paths = np.empty((n_paths, horizon + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(np.cumsum(log_ret, axis=1))
    return paths
