"""NYSE 거래일 달력 — 정규 휴장일 (임시 휴장은 포함하지 않음).

예측 날짜(체크포인트), Yahoo 일봉 보정(바로 다음 거래일인지), 데이터 지연 경고에서 함께 씁니다.
data.py 와 forecast.py 가 모두 가져다 쓰므로 다른 stk 모듈에 의존하지 않습니다.
"""

from __future__ import annotations

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
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NYSECalendar(AbstractHolidayCalendar):
    """NYSE 정규 휴장일 근사 (임시 휴장 제외).

    토요일 휴일은 전날(금) 휴장, 일요일 휴일은 다음 날(월) 휴장. 단, 신정(1/1)이 토요일이면
    NYSE 는 12/31 에 문을 엽니다 (회계연도 마감일이라 대체 휴장 없음 — 예: 2021-12-31, 2027-12-31 개장).
    """

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
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
def holidays() -> np.ndarray:
    """평일에 걸린 휴장일 (numpy datetime64[D] 배열, 1990~2045)."""
    days = NYSECalendar().holidays(start="1990-01-01", end="2045-12-31")
    days = days[days.dayofweek < 5]
    return days.values.astype("datetime64[D]")


@lru_cache(maxsize=1)
def offset() -> CustomBusinessDay:
    return CustomBusinessDay(holidays=pd.DatetimeIndex(holidays()))


def future_trading_days(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """last_date 다음 거래일부터 horizon 개의 거래일."""
    off = offset()
    return pd.date_range(pd.Timestamp(last_date) + off, periods=horizon, freq=off)


def is_trading_day(day) -> bool:
    return bool(np.is_busday(np.datetime64(pd.Timestamp(day).date(), "D"), holidays=holidays()))


def sessions_between(start, end) -> int:
    """start 일부터 end 전날까지의 거래일 수 (np.busday_count 와 같은 반열린 구간, 휴장일 반영).

    start 가 거래일이면 'end 가 start 보다 몇 거래일 뒤인가'와 같습니다 (예: 11-25 → 11-27 = 1, 11-26 휴장)."""
    a = np.datetime64(pd.Timestamp(start).date(), "D")
    b = np.datetime64(pd.Timestamp(end).date(), "D")
    return int(np.busday_count(a, b, holidays=holidays()))
