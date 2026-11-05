"""시세 데이터 수집.

- Yahoo Finance chart API(v8)에서 일봉(수정주가 포함)을 받아옵니다.
- 받아온 데이터는 로컬 CSV로 캐시해 반복 실행 시 네트워크 호출을 줄입니다.
- 네트워크를 쓸 수 없을 때는 ``load_csv`` 로 직접 준비한 CSV를 읽을 수 있습니다.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stk-forecaster/0.1)"}
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "stk"
DEFAULT_RISK_FREE = 0.04
RISK_FREE_SYMBOL = "^IRX"  # 13주 미국 국채(T-bill) 수익률, 단위: %

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class DataError(RuntimeError):
    """데이터를 가져오거나 해석할 수 없을 때 발생합니다."""


def normalize_symbol(symbol: str) -> str:
    """Yahoo 표기법으로 변환합니다 (예: BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def fetch_yahoo(
    symbol: str,
    years: float = 10,
    session: requests.Session | None = None,
    timeout: float = 20.0,
    retries: int = 3,
) -> pd.DataFrame:
    """Yahoo Finance에서 일봉 데이터를 받아 DataFrame으로 반환합니다."""
    symbol = normalize_symbol(symbol)
    now = int(time.time())
    params = {
        "period1": str(int(now - years * 365.25 * 86400)),
        "period2": str(now),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    http = session or requests.Session()
    url = YAHOO_CHART_URL.format(symbol=requests.utils.quote(symbol, safe=""))

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = http.get(url, params=params, headers=HTTP_HEADERS, timeout=timeout)
            if resp.status_code == 404:
                raise DataError(f"{symbol}: 종목을 찾을 수 없습니다 (HTTP 404)")
            resp.raise_for_status()
            return parse_yahoo_chart(resp.json(), symbol)
        except DataError:
            raise
        except (requests.RequestException, ValueError) as err:
            last_err = err
            time.sleep(1.5 * (attempt + 1))
    raise DataError(f"{symbol}: 데이터 다운로드 실패 ({last_err})")


def parse_yahoo_chart(payload: dict, symbol: str = "") -> pd.DataFrame:
    """Yahoo chart API 응답(JSON)을 가격 DataFrame으로 변환합니다."""
    chart = payload.get("chart") or {}
    if chart.get("error"):
        desc = chart["error"].get("description", chart["error"])
        raise DataError(f"{symbol}: Yahoo 오류 - {desc}")
    results = chart.get("result") or []
    if not results:
        raise DataError(f"{symbol}: 응답에 데이터가 없습니다")
    res = results[0]
    timestamps = res.get("timestamp")
    if not timestamps:
        raise DataError(f"{symbol}: 가격 데이터가 비어 있습니다")

    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "America/New_York"
    index = (
        pd.to_datetime(np.asarray(timestamps, dtype="int64"), unit="s", utc=True)
        .tz_convert(tz)
        .normalize()
        .tz_localize(None)
    )
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]
    adj = (res.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")

    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adj if adj is not None else quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=index,
    )
    frame.index.name = "date"
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame["adj_close"] = frame["adj_close"].fillna(frame["close"])
    # 같은 날짜가 두 번 들어오는 경우(장중 데이터 등) 마지막 값을 사용
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise DataError(f"{symbol}: 유효한 종가가 없습니다")
    return frame


def load_csv(path: str | Path) -> pd.DataFrame:
    """사용자 CSV를 읽습니다. 'Date'와 'Close'(또는 'Adj Close') 열이 필요합니다."""
    raw = pd.read_csv(path)
    cols = {c.lower().strip().replace(" ", "_"): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("datetime") or cols.get("timestamp")
    if date_col is None:
        raise DataError(f"{path}: 날짜 열(Date)이 없습니다")
    close_col = cols.get("close")
    adj_col = cols.get("adj_close") or cols.get("adjclose") or close_col
    if adj_col is None:
        raise DataError(f"{path}: 종가 열(Close 또는 Adj Close)이 없습니다")

    frame = pd.DataFrame(index=pd.to_datetime(raw[date_col]).dt.tz_localize(None))
    frame.index.name = "date"
    for key in ("open", "high", "low", "volume"):
        frame[key] = pd.to_numeric(raw[cols[key]], errors="coerce").to_numpy() if key in cols else np.nan
    frame["close"] = pd.to_numeric(raw[close_col or adj_col], errors="coerce").to_numpy()
    frame["adj_close"] = pd.to_numeric(raw[adj_col], errors="coerce").to_numpy()
    frame = frame[PRICE_COLUMNS].dropna(subset=["adj_close"]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.empty:
        raise DataError(f"{path}: 유효한 가격 행이 없습니다")
    return frame


def get_history(
    symbol: str,
    years: float = 10,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    max_age_hours: float = 12.0,
    refresh: bool = False,
) -> pd.DataFrame:
    """캐시를 우선 사용해 가격 이력을 반환합니다."""
    symbol = normalize_symbol(symbol)
    cache_file = None
    if cache_dir is not None:
        safe = symbol.replace("^", "_idx_")
        cache_file = Path(cache_dir) / f"{safe}_{years:g}y.csv"
        if cache_file.exists() and not refresh:
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_h <= max_age_hours:
                cached = pd.read_csv(cache_file, index_col="date", parse_dates=["date"])
                if not cached.empty:
                    return cached

    frame = fetch_yahoo(symbol, years=years)
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache_file)
        except OSError:
            pass  # 캐시 실패는 치명적이지 않음
    return frame


def get_risk_free_series(
    years: float = 10,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.Series | None:
    """13주 T-bill 수익률(연율, 소수) 시계열. 실패하면 None."""
    try:
        frame = get_history(RISK_FREE_SYMBOL, years=years, cache_dir=cache_dir, refresh=refresh)
    except DataError:
        return None
    series = (frame["close"] / 100.0).clip(lower=0.0, upper=0.25)
    series.name = "risk_free"
    return series.dropna()


def risk_free_at(series: pd.Series | None, date: pd.Timestamp | None = None) -> float:
    """주어진 날짜(기본: 최신) 기준 무위험이자율. 데이터가 없으면 기본값."""
    if series is None or series.empty:
        return DEFAULT_RISK_FREE
    if date is not None:
        series = series.loc[:date]
        if series.empty:
            return DEFAULT_RISK_FREE
    return float(series.iloc[-1])
