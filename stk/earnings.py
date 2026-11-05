"""실적 발표일 자동 추정 (가격·거래량만 사용).

실적 발표 다음 거래일에는 대개 '시장 대비 비정상적인 거래량'과 '종목 고유의 큰 움직임'이 함께 나타납니다.
회사는 해마다 거의 같은 시기에 실적을 발표하므로, 과거 여러 해의 패턴으로 다음 발표일을 추정합니다.

1. 비정상 점수 = (거래량 / 50일 중앙값) ÷ (SPY 같은 비율) × √(|베타 제거 수익률| / 60일 변동성)
2. 1차: 분기마다 '분기말 이후 6~84일' 구간의 최고 점수일 → 점수 가중 커널 밀도로 회사의 대표 발표 시점(분기말 +N일)
3. 2차: 대표 시점 ±12일 안에서만 다시 고름 → 시장 전체 충격일 같은 엉뚱한 날을 걸러냄
   최근 3년(오늘 기준) 강한 반응일이 6개 미만이거나, 최근 약 13개월 안에 2개 미만이면 '패턴 없음' (예전 패턴만 남은 종목 제외)
4. 계절(1~4분기)별로 발표 시점이 일관되면(중앙값 편차 ≤ 5일) 그 값을, 아니면 대표 시점을 사용
5. 다음 발표일 = 분기말 + 추정 일수, 날짜 불확실성 ±N거래일 (경로마다 그 범위에서 무작위로 배치)

검증(16개 종목, 과거 데이터로만 예측한 다음 분기 발표일): 대부분 중앙 오차 1~4일, ±7일 이내 75~100%.
NVDA·AMD 처럼 발표 시점이 들쭉날쭉한 종목은 정확도가 낮아 불확실성 범위가 넓게 잡힙니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_OFFSET, MAX_OFFSET = 6, 84     # 분기말 이후 실적 발표가 나올 수 있는 구간 (일)
WINDOW = 12                        # 2차 선택 폭 (±일)
STRONG_SCORE = 2.5                 # '실적 반응일'로 볼 최소 점수
MIN_EVENTS = 6                     # 최근 기간 강한 이벤트가 이보다 적으면 패턴 없음으로 판단
MIN_CONSISTENCY = 0.4              # 분기별 선택일 중 강한 반응일 비율이 이보다 낮으면 패턴 없음
FRESH_DAYS, MIN_FRESH = 400, 2     # 최근 약 13개월 안에 강한 반응일이 이보다 적으면 '지금은 안 맞는 패턴'으로 판단


@dataclass
class EarningsModel:
    offset_mode: int                       # 대표 발표 시점 (분기말 + 일)
    season_offsets: dict[int, float]       # 발표 분기(1~4)별 분기말 이후 일수
    past_events: list[pd.Timestamp]        # 과거 실적 반응일로 판단된 날 (변동성 추정에서 제외)
    event_std: float                       # 실적 반응일 하루 충격 표준편차 (로그)
    spread_days: int                       # 날짜 불확실성 (±거래일)
    consistency: float                     # 2차 선택일 중 강한 반응일 비율 (0~1)
    last_event: pd.Timestamp | None = None
    history: list[dict] = field(default_factory=list, repr=False)

    def next_dates(self, as_of: pd.Timestamp, until: pd.Timestamp) -> list[pd.Timestamp]:
        """as_of 이후 until 까지 예상되는 실적 반응일."""
        out = []
        q = pd.Period(as_of, freq="Q")
        for k in range(0, 10):
            period = q + k
            q_end = period.start_time - pd.Timedelta(days=1)
            date = q_end + pd.Timedelta(days=round(self.season_offsets.get(period.quarter, self.offset_mode)))
            date = _next_business_day(date)
            if date > until:
                break
            if date <= as_of:
                continue
            # 이번 분기 실적이 이미 나왔으면(창 안에서 강한 반응일 확인) 건너뜀
            if any(abs((e - date).days) <= WINDOW and e <= as_of for e in self.past_events):
                continue
            out.append(date)
        return out

    def to_dict(self) -> dict:
        return {
            "offset_mode": self.offset_mode,
            "season_offsets": self.season_offsets,
            "past_events": [d.strftime("%Y-%m-%d") for d in self.past_events[-8:]],
            "event_std": self.event_std,
            "spread_days": self.spread_days,
            "consistency": self.consistency,
        }


def _next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def abnormal_scores(prices: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """일별 비정상 점수 표. 거래량 데이터가 없으면 None."""
    if "volume" not in prices or prices["volume"].isna().mean() > 0.2:
        return None
    s = prices["adj_close"].astype(float)
    v = prices["volume"].astype(float).replace(0, np.nan)
    r = np.log(s / s.shift(1))
    if benchmark is not None and "volume" in benchmark:
        b = benchmark["adj_close"].astype(float).reindex(s.index).ffill()
        bv = benchmark["volume"].astype(float).reindex(s.index).replace(0, np.nan)
        rm = np.log(b / b.shift(1))
        beta = (r.rolling(250, min_periods=60).cov(rm) / rm.rolling(250, min_periods=60).var()).shift(1).fillna(1.0)
        resid = r - beta * rm
        mvr = (bv / bv.rolling(50, min_periods=20).median().shift(1)).clip(lower=1.0).fillna(1.0)
    else:
        resid = r
        mvr = pd.Series(1.0, index=s.index)
    sig = resid.rolling(60, min_periods=30).std().shift(1)
    vr = v / v.rolling(50, min_periods=20).median().shift(1)
    d = pd.DataFrame({"r": r, "z": resid.abs() / sig, "av": vr / mvr}).dropna()
    if d.empty:
        return None
    d["score"] = d["av"].clip(upper=8) * np.sqrt(d["z"].clip(upper=8))
    q_end = d.index.to_period("Q").start_time - pd.Timedelta(days=1)
    d["off"] = (d.index - q_end).days
    d["qtr"] = d.index.to_period("Q")
    return d


def _kde_mode(offs: np.ndarray, w: np.ndarray, bw: float = 4.0) -> int:
    grid = np.arange(MIN_OFFSET, MAX_OFFSET + 1)
    dens = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - offs[None, :]) / bw) ** 2)).sum(axis=1)
    return int(grid[int(np.argmax(dens))])


def fit_earnings(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    years: int = 8,
) -> EarningsModel | None:
    """과거 가격·거래량으로 실적 발표 패턴을 추정합니다. 뚜렷한 패턴이 없으면 None."""
    d = abnormal_scores(prices, benchmark)
    if d is None or len(d) < 400:
        return None
    end = d.index[-1]  # 데이터 마지막 날 (분기 구간 필터 전)
    d = d[d.index >= end - pd.DateOffset(years=years)]
    d = d[(d["off"] >= MIN_OFFSET) & (d["off"] <= MAX_OFFSET)]
    if d["qtr"].nunique() < 8:
        return None
    first = d.loc[d.groupby("qtr")["score"].idxmax()]
    mode = _kde_mode(first["off"].to_numpy(float), first["score"].to_numpy(float))
    win = d[(d["off"] - mode).abs() <= WINDOW]
    if win.empty:
        return None
    picks = win.loc[win.groupby("qtr")["score"].idxmax()]
    strong = picks[picks["score"] >= STRONG_SCORE]
    recent = strong[strong.index >= end - pd.DateOffset(years=3)]   # 최근 3년 = 오늘 기준 (마지막 반응일 기준 아님)
    fresh = strong[strong.index >= end - pd.Timedelta(days=FRESH_DAYS)]
    if len(recent) < MIN_EVENTS or len(strong) / len(picks) < MIN_CONSISTENCY or len(fresh) < MIN_FRESH:
        return None

    season: dict[int, float] = {}
    mads = []
    for k in range(1, 5):
        g = picks[picks.index.quarter == k]["off"].iloc[-4:]  # 최근 4년 — 발표 일정이 바뀌면 따라감
        if len(g) >= 3:
            med = float(np.median(g))
            mad = float(np.median(np.abs(g - med)))
            mads.append(mad)
            season[k] = med if mad <= 5 else float(mode)
        else:
            season[k] = float(mode)
    # 날짜 불확실성: 계절별 편차의 중앙값 (달력일 → 거래일 근사), 2~7거래일
    spread = int(np.clip(round(np.median(mads) * 5 / 7) if mads else 4, 2, 7))

    # 실적 반응일 하루 충격: 최근 3년 반응일 수익률 제곱평균 − 평상시 일간 분산
    normal = d.drop(index=strong.index, errors="ignore")["r"]
    normal_var = float(np.nanvar(normal.iloc[-500:])) if len(normal) else 0.0
    ev_var = float(np.mean(recent["r"].to_numpy() ** 2))
    event_std = float(np.sqrt(max(ev_var - normal_var, normal_var)))

    return EarningsModel(
        offset_mode=mode,
        season_offsets=season,
        past_events=list(strong.index),
        event_std=event_std,
        spread_days=spread,
        consistency=float(len(strong) / len(picks)),
        last_event=strong.index.max() if len(strong) else None,
        history=[{"date": t.strftime("%Y-%m-%d"), "ret": float(r), "score": float(sc)}
                 for t, r, sc in zip(strong.index[-8:], strong["r"][-8:], strong["score"][-8:])],
    )
