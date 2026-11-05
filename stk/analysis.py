"""가격 이력 진단: 변동성, 추세, 모멘텀, 베타, 꼬리 두께, 추세 점수."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_OBSERVATIONS = 60

# 추세 점수 구성 요소 가중치 (합 = 1)
SIGNAL_WEIGHTS = {
    "ma200_gap": 0.25,   # 200일 이동평균 대비 위치
    "ma_cross": 0.20,    # 50일선 vs 200일선 (골든/데드 크로스)
    "momentum": 0.25,    # 12-1개월 모멘텀
    "slope": 0.20,       # 6개월 로그가격 회귀 기울기의 t-통계량
    "rsi": 0.10,         # RSI 과열/과매도 (역추세)
}


@dataclass
class MarketStats:
    """한 종목의 현재 상태 요약."""

    symbol: str
    last_date: pd.Timestamp
    last_price: float
    n_obs: int
    vol_long: float          # 장기(최대 5년) 연율 변동성
    vol_1y: float            # 최근 1년 연율 변동성
    vol_now: float           # EWMA(λ=0.94) 현재 연율 변동성
    cagr: float              # 표본 기간 연복리 수익률
    return_1y: float         # 최근 1년 수익률
    slope_6m: float          # 6개월 로그가격 회귀 기울기(연율)
    slope_1y: float          # 1년 로그가격 회귀 기울기(연율)
    r2_6m: float             # 6개월 추세선 결정계수 (추세의 깨끗함)
    ma20: float
    ma50: float
    ma200: float
    rsi14: float
    momentum_12_1: float     # t-252 ~ t-21 수익률
    drawdown_now: float      # 52주 고점 대비 현재 낙폭
    max_drawdown_1y: float
    high_52w: float
    low_52w: float
    beta: float | None
    tail_df: float           # Student-t 자유도 (작을수록 꼬리 두꺼움)
    trend_score: float       # -1(강한 하락) ~ +1(강한 상승)
    trend_label: str
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["last_date"] = self.last_date.strftime("%Y-%m-%d")
        return out


def log_returns(prices: pd.Series) -> pd.Series:
    prices = prices.astype(float)
    return np.log(prices / prices.shift(1)).dropna()


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics 방식 EWMA 변동성(연율)."""
    r = returns.to_numpy(dtype=float)
    if r.size == 0:
        return float("nan")
    var = float(np.var(r[: min(r.size, 60)]))  # 초기값
    for x in r:
        var = lam * var + (1.0 - lam) * x * x
    return math.sqrt(var * TRADING_DAYS)


def ewma_vol_series(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """각 시점까지의 EWMA 일간 변동성 (해당 시점 수익률 반영 전)."""
    r = returns.to_numpy(dtype=float)
    out = np.empty_like(r)
    var = float(np.var(r[: min(r.size, 60)])) if r.size else 0.0
    for i, x in enumerate(r):
        out[i] = math.sqrt(var)
        var = lam * var + (1.0 - lam) * x * x
    return pd.Series(out, index=returns.index)


def annualized_vol(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def trend_regression(log_prices: np.ndarray) -> tuple[float, float]:
    """로그가격 OLS 회귀 → (연율 기울기, R²)."""
    y = np.asarray(log_prices, dtype=float)
    n = y.size
    if n < 3:
        return float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    x -= x.mean()
    yc = y - y.mean()
    sxx = float(x @ x)
    slope = float(x @ yc) / sxx
    resid = yc - slope * x
    sst = float(yc @ yc)
    r2 = 1.0 - float(resid @ resid) / sst if sst > 0 else 0.0
    return slope * TRADING_DAYS, r2


def rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder RSI."""
    delta = prices.astype(float).diff().dropna()
    if len(delta) < period:
        return float("nan")
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    rs = gain / loss
    return float(100.0 - 100.0 / (1.0 + rs))


def max_drawdown(prices: pd.Series) -> float:
    p = prices.astype(float).to_numpy()
    if p.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(p)
    return float((p / peak - 1.0).min())


def estimate_beta(returns: pd.Series, bench_returns: pd.Series, window: int = 2 * TRADING_DAYS) -> float | None:
    joined = pd.concat([returns, bench_returns], axis=1, join="inner").dropna().iloc[-window:]
    if len(joined) < MIN_OBSERVATIONS:
        return None
    a, b = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    var_b = float(np.var(b, ddof=1))
    if var_b <= 0:
        return None
    return float(np.cov(a, b, ddof=1)[0, 1] / var_b)


def estimate_tail_df(returns: pd.Series) -> float:
    """EWMA 변동성으로 표준화한 수익률의 초과첨도로 Student-t 자유도를 추정.

    초과첨도 k 인 t분포의 자유도는 df = 4 + 6/k 입니다.
    """
    if len(returns) < 120:
        return 6.0
    sigma = ewma_vol_series(returns)
    z = (returns / sigma.replace(0, np.nan)).iloc[60:].dropna()
    z = z[np.isfinite(z)]
    if len(z) < 60:
        return 6.0
    k = float(z.kurt())  # pandas: 초과첨도
    if not np.isfinite(k) or k <= 0.05:
        return 30.0
    return float(np.clip(4.0 + 6.0 / k, 3.0, 30.0))


def estimate_event_std(prices: pd.DataFrame | pd.Series, quarters: int = 12) -> float:
    """실적 발표일 같은 이벤트 하루 충격의 표준편차(로그) 근사.

    분기(63거래일)마다 가장 큰 일간 변동은 대개 실적 발표일이라는 점을 이용해,
    최근 N분기 최대 변동의 제곱평균에서 평상시 일간 분산을 뺀 값을 씁니다.
    """
    r = log_returns(_price_series(prices)).to_numpy()
    q = 63
    n = min(quarters, len(r) // q)
    if n < 2:
        return float(np.std(r) * 3) if len(r) else 0.05
    blocks = r[-n * q:].reshape(n, q)
    peak2 = float(np.mean(np.max(np.abs(blocks), axis=1) ** 2))
    daily_var = float(np.median(blocks.var(axis=1)))
    return float(np.sqrt(max(peak2 - daily_var, daily_var)))


def _tanh(x: float) -> float:
    return float(np.tanh(x)) if np.isfinite(x) else float("nan")


def compute_signals(prices: pd.Series, vol: float) -> dict[str, float]:
    """각 추세 신호를 대략 [-1, 1] 범위로 정규화해 반환합니다 (데이터 부족 시 NaN)."""
    p = prices.astype(float)
    last = float(p.iloc[-1])
    vol = max(vol, 0.05)
    sig: dict[str, float] = {k: float("nan") for k in SIGNAL_WEIGHTS}

    if len(p) >= 200:
        ma200 = float(p.iloc[-200:].mean())
        ma50 = float(p.iloc[-50:].mean())
        sig["ma200_gap"] = _tanh(math.log(last / ma200) / (vol * math.sqrt(0.5)))
        sig["ma_cross"] = _tanh(math.log(ma50 / ma200) / (vol * 0.3))
    if len(p) >= TRADING_DAYS + 1:
        mom = math.log(float(p.iloc[-22]) / float(p.iloc[-TRADING_DAYS - 1]))
        sig["momentum"] = _tanh(mom / (vol * math.sqrt(11 / 12)))
    window = min(len(p), TRADING_DAYS // 2)
    if window >= 40:
        slope, _ = trend_regression(np.log(p.iloc[-window:].to_numpy()))
        years = window / TRADING_DAYS
        tstat = slope / vol * math.sqrt(years)
        sig["slope"] = _tanh(tstat / 2.0)
    r = rsi(p)
    if np.isfinite(r):
        if r > 70:
            sig["rsi"] = -(r - 70) / 30
        elif r < 30:
            sig["rsi"] = (30 - r) / 30
        else:
            sig["rsi"] = 0.0
    return sig


def combine_signals(signals: dict[str, float]) -> float:
    """유효한 신호만으로 가중평균 → [-1, 1] 추세 점수."""
    num = den = 0.0
    for key, w in SIGNAL_WEIGHTS.items():
        v = signals.get(key, float("nan"))
        if np.isfinite(v):
            num += w * v
            den += w
    if den == 0:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))


def trend_label(score: float) -> str:
    if score >= 0.5:
        return "강한 상승 추세"
    if score >= 0.15:
        return "상승 추세"
    if score > -0.15:
        return "중립/횡보"
    if score > -0.5:
        return "하락 추세"
    return "강한 하락 추세"


def analyze(
    prices: pd.DataFrame | pd.Series,
    symbol: str = "",
    benchmark: pd.DataFrame | pd.Series | None = None,
    exclude_dates: list | None = None,
) -> MarketStats:
    """가격 이력으로 MarketStats 를 계산합니다. (수정종가 기준)

    exclude_dates: 변동성·꼬리 추정에서 뺄 날짜 (예: 실적 반응일 — 이벤트로 따로 시뮬레이션할 때 이중 계산 방지)
    """
    p = _price_series(prices)
    if len(p) < MIN_OBSERVATIONS:
        raise ValueError(f"{symbol}: 분석에 최소 {MIN_OBSERVATIONS}거래일 데이터가 필요합니다 (현재 {len(p)}일)")

    rets = log_returns(p)
    vol_rets = rets.drop(index=pd.DatetimeIndex(exclude_dates).intersection(rets.index)) if exclude_dates else rets
    last = float(p.iloc[-1])
    long_rets = vol_rets.loc[vol_rets.index >= rets.index[max(0, len(rets) - 5 * TRADING_DAYS)]]
    year = p.iloc[-(TRADING_DAYS + 1):]
    year_rets = vol_rets.loc[vol_rets.index >= rets.index[max(0, len(rets) - TRADING_DAYS)]]

    vol_long = annualized_vol(long_rets)
    vol_1y = annualized_vol(year_rets)
    vol_now = ewma_vol(vol_rets.loc[vol_rets.index >= rets.index[max(0, len(rets) - 2 * TRADING_DAYS)]])

    span_years = max((p.index[-1] - p.index[0]).days / 365.25, 1e-9)
    cagr = (last / float(p.iloc[0])) ** (1.0 / span_years) - 1.0

    six = p.iloc[-(TRADING_DAYS // 2):]
    slope_6m, r2_6m = trend_regression(np.log(six.to_numpy()))
    slope_1y, _ = trend_regression(np.log(year.to_numpy()))

    beta = None
    if benchmark is not None:
        beta = estimate_beta(rets, log_returns(_price_series(benchmark)))

    signals = compute_signals(p, vol_long)
    score = combine_signals(signals)

    high_52w = float(year.max())
    mom_12_1 = float(p.iloc[-22] / p.iloc[-TRADING_DAYS - 1] - 1.0) if len(p) > TRADING_DAYS + 1 else float("nan")

    return MarketStats(
        symbol=symbol,
        last_date=p.index[-1],
        last_price=last,
        n_obs=len(p),
        vol_long=vol_long,
        vol_1y=vol_1y,
        vol_now=vol_now,
        cagr=float(cagr),
        return_1y=float(last / float(year.iloc[0]) - 1.0),
        slope_6m=slope_6m,
        slope_1y=slope_1y,
        r2_6m=r2_6m,
        ma20=float(p.iloc[-20:].mean()),
        ma50=float(p.iloc[-50:].mean()),
        ma200=float(p.iloc[-200:].mean()),
        rsi14=rsi(p),
        momentum_12_1=mom_12_1,
        drawdown_now=last / high_52w - 1.0,
        max_drawdown_1y=max_drawdown(year),
        high_52w=high_52w,
        low_52w=float(year.min()),
        beta=beta,
        tail_df=estimate_tail_df(long_rets),
        trend_score=score,
        trend_label=trend_label(score),
        signals=signals,
    )


def _price_series(prices: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        col = "adj_close" if "adj_close" in prices.columns else "close"
        series = prices[col]
    else:
        series = prices
    series = series.astype(float).dropna()
    return series[series > 0]
