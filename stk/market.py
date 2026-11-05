"""시장 환경(market context): VIX, VIX 기간구조, S&P500(SPY) 추세.

개별주의 위험은 '시장 전체 위험(베타 부분)' + '종목 고유 위험'으로 나눌 수 있습니다.
VIX는 옵션 시장이 예상하는 S&P500의 향후 30일 변동성이므로, 시장 부분의 변동성을
과거 데이터보다 앞서 반영합니다.

    σ_fwd² = β² · (VIX × 위험프리미엄 보정)² + σ_고유²

또한 VIX 수준(5년 백분위), 기간구조(VIX/VIX3M > 1 이면 단기 공포), SPY 낙폭으로
'시장 스트레스 점수'(−1 안정 ~ +1 공포)를 만들어, 스트레스가 높을수록
급락·조정 후 반등 같은 고변동 시나리오의 확률을 높입니다.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .analysis import TRADING_DAYS, analyze, annualized_vol, log_returns

# VIX(내재변동성)는 평균적으로 실현변동성보다 높게 형성됨 (변동성 위험 프리미엄)
VRP_RATIO_30D = 0.85
VRP_RATIO_3M = 0.88

STRESS_WEIGHTS = {"vix_rank": 0.5, "term": 0.3, "drawdown": 0.2}


@dataclass
class MarketContext:
    as_of: pd.Timestamp
    vix: float | None
    vix3m: float | None
    vix_rank: float | None        # 최근 5년 중 현재 VIX 백분위 (0~1)
    term_ratio: float | None      # VIX / VIX3M (1 초과 = 역전, 단기 공포)
    spy_vol_1y: float | None      # SPY 1년 실현변동성
    spy_trend_score: float | None
    spy_drawdown: float | None    # SPY 52주 고점 대비
    stress: float                 # −1(안정) ~ +1(공포)
    label: str
    components: dict

    def implied_market_vol(self, horizon_days: int) -> float | None:
        """기간에 맞는 시장 내재변동성(실현 기준으로 보정, 연율)."""
        if self.vix is None:
            return None
        short = self.vix / 100 * VRP_RATIO_30D
        if self.vix3m is None or horizon_days <= 21:
            return short
        long = self.vix3m / 100 * VRP_RATIO_3M
        w = min(1.0, (horizon_days - 21) / 42)
        return math.sqrt((1 - w) * short**2 + w * long**2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.strftime("%Y-%m-%d")
        return d


def _close(frame: pd.DataFrame | pd.Series | None, as_of: pd.Timestamp | None) -> pd.Series | None:
    if frame is None:
        return None
    s = frame["close"] if isinstance(frame, pd.DataFrame) else frame
    s = s.astype(float).dropna()
    if as_of is not None:
        s = s.loc[:as_of]
    return s if len(s) else None


def stress_label(stress: float) -> str:
    if stress >= 0.5:
        return "공포/위기"
    if stress >= 0.15:
        return "불안"
    if stress > -0.4:
        return "보통"
    return "안정"


def market_context(
    spy: pd.DataFrame | None,
    vix: pd.DataFrame | None = None,
    vix3m: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> MarketContext | None:
    """주어진 시점(기본: 최신)까지의 데이터만으로 시장 환경을 계산합니다."""
    spy_px = None
    if spy is not None:
        col = "adj_close" if "adj_close" in spy else "close"
        spy_px = spy[col].astype(float).dropna()
        if as_of is not None:
            spy_px = spy_px.loc[:as_of]
    vix_s = _close(vix, as_of)
    vix3m_s = _close(vix3m, as_of)
    if (spy_px is None or len(spy_px) < 60) and vix_s is None:
        return None

    comps: dict[str, float] = {}
    vix_now = vix3m_now = rank = ratio = None
    if vix_s is not None:
        vix_now = float(vix_s.iloc[-1])
        hist = vix_s.iloc[-5 * TRADING_DAYS:]
        rank = float((hist <= vix_now).mean())
        comps["vix_rank"] = 2 * rank - 1
    if vix3m_s is not None and vix_now is not None:
        vix3m_now = float(vix3m_s.iloc[-1])
        if vix3m_now > 0:
            ratio = vix_now / vix3m_now
            comps["term"] = float(np.tanh((ratio - 0.92) / 0.06))

    spy_vol = trend = dd = None
    if spy_px is not None and len(spy_px) >= 60:
        year = spy_px.iloc[-TRADING_DAYS:]
        dd = float(year.iloc[-1] / year.max() - 1)
        comps["drawdown"] = float(2 * np.tanh(-dd / 0.08) - 1)
        spy_vol = annualized_vol(log_returns(year))
        trend = analyze(spy_px, "SPY").trend_score

    num = sum(STRESS_WEIGHTS[k] * v for k, v in comps.items())
    den = sum(STRESS_WEIGHTS[k] for k in comps)
    stress = float(np.clip(num / den, -1, 1)) if den else 0.0
    as_of_date = (vix_s.index[-1] if vix_s is not None else spy_px.index[-1])
    return MarketContext(
        as_of=as_of_date,
        vix=vix_now,
        vix3m=vix3m_now,
        vix_rank=rank,
        term_ratio=ratio,
        spy_vol_1y=spy_vol,
        spy_trend_score=trend,
        spy_drawdown=dd,
        stress=stress,
        label=stress_label(stress),
        components=comps,
    )


def forward_vol(
    stock_vol_1y: float,
    beta: float | None,
    market: MarketContext | None,
    horizon_days: int,
) -> float | None:
    """VIX 기반 종목 선행 변동성: β²·σ_시장(내재)² + σ_고유²."""
    if market is None or market.spy_vol_1y is None or not math.isfinite(stock_vol_1y):
        return None
    implied = market.implied_market_vol(horizon_days)
    if implied is None:
        return None
    b = 1.0 if beta is None or not math.isfinite(beta) else min(max(beta, 0.0), 3.5)
    idio_var = max(stock_vol_1y**2 - b**2 * market.spy_vol_1y**2, (0.3 * stock_vol_1y) ** 2)
    return math.sqrt(b**2 * implied**2 + idio_var)
