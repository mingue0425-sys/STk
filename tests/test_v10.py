import pandas as pd

from stk import nyse
from stk.data import parse_yahoo_chart


def _ny(x):
    return int(pd.Timestamp(x, tz="America/New_York").timestamp())


def _payload(bars, meta_time, period=None):
    ts = [_ny(d + " 09:30") for d in bars]
    n = len(bars)
    meta = {"exchangeTimezoneName": "America/New_York", "regularMarketTime": meta_time, "regularMarketPrice": 101.5,
            "regularMarketDayHigh": 102.0, "regularMarketDayLow": 100.5, "regularMarketVolume": 1_000_000}
    if period:
        meta["currentTradingPeriod"] = {"regular": {"start": _ny(period[0]), "end": _ny(period[1])}}
    close = [100.0 + i for i in range(n)]
    return {"chart": {"result": [{
        "meta": meta, "timestamp": ts,
        "indicators": {"quote": [{"open": close, "high": close, "low": close, "close": close, "volume": [5] * n}],
                       "adjclose": [{"adjclose": close}]}}]}}


def test_nyse_calendar_new_year_saturday_and_thanksgiving():
    # 신정이 토요일이면 전날 12/31 은 개장 (NYSE 규칙), 일요일이면 다음 날 월요일 휴장
    assert nyse.is_trading_day("2021-12-31") and nyse.is_trading_day("2027-12-31")
    assert not nyse.is_trading_day("2023-01-02")          # 1/1 일요일 → 1/2 월요일 휴장
    assert nyse.is_trading_day("2028-01-03")              # 1/1 토요일 → 대체 휴장 없음
    assert not nyse.is_trading_day("2026-11-26") and nyse.is_trading_day("2026-11-27")
    assert not nyse.is_trading_day("2027-07-05")          # 7/4 일요일 → 월요일 휴장
    assert not nyse.is_trading_day("2026-07-03")          # 7/4 토요일 → 금요일 휴장
    days = nyse.future_trading_days(pd.Timestamp("2027-12-30"), 3)
    assert [d.strftime("%Y-%m-%d") for d in days] == ["2027-12-31", "2028-01-03", "2028-01-04"]
    assert nyse.sessions_between("2026-11-25", "2026-11-27") == 1
    assert nyse.sessions_between("2026-11-24", "2026-11-27") == 2
    assert nyse.sessions_between("2026-11-13", "2026-11-16") == 1


def test_forecast_dates_skip_holidays():
    from stk.forecast import future_trading_days
    days = future_trading_days(pd.Timestamp("2026-11-25"), 3)
    assert [d.strftime("%Y-%m-%d") for d in days] == ["2026-11-27", "2026-11-30", "2026-12-01"]


def test_meta_patch_across_holiday_but_not_across_missing_session():
    period = ("2026-11-27 09:30", "2026-11-27 13:00")
    # 11-25 다음 거래일은 11-27 (11-26 휴장) → 조기 폐장 후 meta 로 채움
    f = parse_yahoo_chart(_payload(["2026-11-24", "2026-11-25"], _ny("2026-11-27 13:00"), period), "SPY")
    assert f.index[-1] == pd.Timestamp("2026-11-27") and f["close"].iloc[-1] == 101.5
    # 11-25 일봉까지 빠져 있으면(11-24 다음이 11-27 이 아님) 채우지 않음
    f = parse_yahoo_chart(_payload(["2026-11-23", "2026-11-24"], _ny("2026-11-27 13:00"), period), "SPY")
    assert f.index[-1] == pd.Timestamp("2026-11-24") and len(f) == 2
    # 금요일 → 월요일은 그대로 채움
    f = parse_yahoo_chart(_payload(["2026-11-12", "2026-11-13"], _ny("2026-11-16 16:00")), "SPY")
    assert f.index[-1] == pd.Timestamp("2026-11-16")
    # 휴장일(11-26) 당일에는 meta 시각이 11-25 그대로 → 아무것도 추가하지 않음
    f = parse_yahoo_chart(_payload(["2026-11-24", "2026-11-25"], _ny("2026-11-25 16:00")), "SPY")
    assert f.index[-1] == pd.Timestamp("2026-11-25") and len(f) == 2
