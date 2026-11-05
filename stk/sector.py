"""섹터 분석: 가장 잘 맞는 섹터 ETF 자동 판별, 위험 분해(시장·섹터·종목 고유), 섹터 대비 상대 강도.

1. 종목과 각 섹터 ETF(SPDR 11개)의 일간 수익률에서 S&P500(SPY) 영향을 빼고(잔차),
   두 잔차의 상관이 가장 높은 ETF 를 그 종목의 섹터로 봅니다 (최근 2년).
2. r = α + β_시장·r_SPY + β_섹터·(섹터 잔차) + ε  → 분산을 시장·섹터·고유 몫으로 나눕니다.
3. 최근 3개월·6개월 수익률을 섹터 ETF 와 비교합니다 (상대 강도).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .analysis import TRADING_DAYS, analyze

SECTOR_ETFS = {
    "XLK": "기술",
    "XLC": "커뮤니케이션",
    "XLY": "임의소비재",
    "XLP": "필수소비재",
    "XLF": "금융",
    "XLV": "헬스케어",
    "XLE": "에너지",
    "XLI": "산업재",
    "XLB": "소재",
    "XLU": "유틸리티",
    "XLRE": "부동산",
}
MIN_OVERLAP = 120
WEAK_FIT = 0.30   # 이보다 낮으면 (시장 비중이 큰 초대형주 등) 단순 상관으로 판별하고 '연관 약함'으로 표시


@dataclass
class SectorInfo:
    etf: str
    name: str
    fit: float                 # 시장 영향을 뺀 뒤 섹터와의 상관 (0~1)
    beta_market: float
    beta_sector: float
    share_market: float        # 분산 중 시장 몫
    share_sector: float        # 섹터 몫
    share_idio: float          # 종목 고유 몫
    rel_3m: float              # 최근 3개월 수익률 − 섹터 ETF 수익률
    rel_6m: float
    sector_ret_3m: float
    sector_trend: float        # 섹터 ETF 추세 점수
    sector_label: str
    runner_up: str | None = None
    method: str = "residual"   # "residual": 시장 영향 제거 후 상관 / "raw": 초대형주라 단순 상관으로 판별(연관 약함)

    def to_dict(self) -> dict:
        return asdict(self)


def _rets(frame: pd.DataFrame | pd.Series) -> pd.Series:
    s = frame["adj_close"] if isinstance(frame, pd.DataFrame) else frame
    s = s.astype(float).dropna()
    return np.log(s / s.shift(1)).dropna()


def _resid(y: pd.Series, x: pd.Series) -> tuple[pd.Series, float]:
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(df) < MIN_OVERLAP:
        return pd.Series(dtype=float), float("nan")
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    beta = float(np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    return a - beta * b, beta


def _period_return(frame, days: int) -> float:
    s = (frame["adj_close"] if isinstance(frame, pd.DataFrame) else frame).astype(float).dropna()
    if len(s) <= days:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1)


def analyze_sector(
    prices: pd.DataFrame | pd.Series,
    benchmark: pd.DataFrame | pd.Series,
    sectors: dict[str, pd.DataFrame],
    as_of: pd.Timestamp | None = None,
    window: int = 2 * TRADING_DAYS,
) -> SectorInfo | None:
    """섹터 ETF 시세(dict: 티커 → DataFrame)를 받아 종목의 섹터 정보를 계산합니다."""
    if benchmark is None or not sectors:
        return None

    def cut(f):
        return f.loc[:as_of] if as_of is not None else f

    r = _rets(cut(prices)).iloc[-window:]
    m = _rets(cut(benchmark))
    e_stock, beta_m = _resid(r, m)
    if e_stock.empty:
        return None
    scores, raw_scores = [], []
    resid_by_etf = {}
    for etf, frame in sectors.items():
        if frame is None or len(frame) < MIN_OVERLAP:
            continue
        etf_r = _rets(cut(frame))
        e_etf, _ = _resid(etf_r, m)
        df = pd.concat([e_stock, e_etf], axis=1, join="inner").dropna()
        if len(df) < MIN_OVERLAP:
            continue
        scores.append((float(df.corr().iloc[0, 1]), etf))
        raw = pd.concat([r, etf_r], axis=1, join="inner").dropna()
        raw_scores.append((float(raw.corr().iloc[0, 1]), etf))
        resid_by_etf[etf] = e_etf
    if not scores:
        return None
    scores.sort(reverse=True)
    raw_scores.sort(reverse=True)
    best_c, best = scores[0]
    method = "residual"
    if best_c < WEAK_FIT:  # 시장 영향을 빼면 섹터 신호가 거의 안 남는 경우 (예: 지수 비중이 큰 초대형주)
        best = raw_scores[0][1]
        best_c = max(dict((e, c) for c, e in scores).get(best, 0.0), 0.0)
        method = "raw"

    df = pd.concat([r, m, resid_by_etf[best]], axis=1, join="inner").dropna()
    df.columns = ["r", "m", "s"]
    x = np.column_stack([np.ones(len(df)), df["m"], df["s"]])
    coef, *_ = np.linalg.lstsq(x, df["r"].to_numpy(), rcond=None)
    _, bm, bs = coef
    eps = df["r"].to_numpy() - x @ coef
    var_m, var_s, var_e = bm**2 * df["m"].var(), bs**2 * df["s"].var(), eps.var(ddof=1)
    total = var_m + var_s + var_e

    etf_frame = cut(sectors[best])
    st = analyze(etf_frame, best) if len(etf_frame) >= 60 else None
    p = cut(prices)
    return SectorInfo(
        etf=best,
        name=SECTOR_ETFS.get(best, best),
        fit=best_c,
        beta_market=float(bm),
        beta_sector=float(bs),
        share_market=float(var_m / total),
        share_sector=float(var_s / total),
        share_idio=float(var_e / total),
        rel_3m=_period_return(p, 63) - _period_return(etf_frame, 63),
        rel_6m=_period_return(p, 126) - _period_return(etf_frame, 126),
        sector_ret_3m=_period_return(etf_frame, 63),
        sector_trend=st.trend_score if st else float("nan"),
        sector_label=st.trend_label if st else "-",
        runner_up=(scores[1][1] if method == "residual" and len(scores) > 1 and scores[1][0] > 0.6 * best_c else None),
        method=method,
    )


def load_sector_etfs(loader) -> dict[str, pd.DataFrame]:
    """loader(티커) → DataFrame. 실패한 ETF 는 건너뜁니다."""
    out = {}
    for etf in SECTOR_ETFS:
        try:
            out[etf] = loader(etf)
        except Exception:  # noqa: BLE001 - 섹터 데이터는 보조 정보
            continue
    return out
