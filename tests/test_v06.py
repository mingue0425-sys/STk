import pandas as pd

from stk.data import parse_yahoo_chart
from stk.forecast import LOG_QUANTILES
from stk.track_page import render_track_page
from stk.tracking import breakdown, by_symbol, evaluate


def _record(sym, as_of, horizon, price, cp_date, width=0.05):
    lv = list(LOG_QUANTILES)
    q = [price * (1 + width * (x - 0.5) * 4) for x in lv]   # 단조 증가 분위
    return {"id": f"{sym}|{as_of}|{horizon}|e", "symbol": sym, "as_of": as_of, "price": price, "horizon": horizon,
            "model": "e", "prob_up": 0.55, "quantile_levels": lv,
            "checkpoints": [{"label": "1주", "day": 5, "date": cp_date, "q": q},
                            {"label": "만기", "day": horizon, "date": "2099-01-01", "q": q}]}


def _prices():
    idx = pd.bdate_range("2026-11-02", "2026-11-13")
    return pd.Series([100.0] * len(idx), index=idx).mul(pd.Series(range(len(idx)), index=idx) * 0.002 + 1)


def test_breakdown_splits_horizons_and_counts_dates():
    recs = [_record(s, d, h, 100.0, c) for s in ("AAA", "BBB") for h in (21, 126)
            for d, c in (("2026-11-04", "2026-11-11"), ("2026-11-05", "2026-11-12"))]
    tr = evaluate(recs, lambda s: _prices())
    assert tr.summary["evaluated"] == 8 and tr.summary["target_dates"] == 2
    bd = breakdown(tr.evaluations)
    assert list(bd["horizon"]) == [21, 126] and list(bd["n"]) == [4, 4] and list(bd["dates"]) == [2, 2]
    assert set(by_symbol(tr.evaluations)["symbol"]) == {"AAA", "BBB"}
    assert {"AAA", "BBB"} <= set(tr.prices)


def test_track_page_renders_charts_and_empty_state():
    recs = [_record("AAA", "2026-11-04", 21, 100.0, "2026-11-11"),
            _record("AAA", "2026-11-04", 126, 100.0, "2026-11-11"),
            _record("BBB", "2026-11-05", 21, 100.0, "2026-11-12", width=0.001)]   # 좁은 범위 → 90% 밖
    page = render_track_page(evaluate(recs, lambda s: _prices()))
    assert page.count('class="sm"') == 2 and "misspt" in page and "PIT 분포" in page
    assert page.count("<polyline") == 2 and "서로 다른 평가 날짜" in page
    empty = render_track_page(evaluate([_record("AAA", "2026-11-12", 21, 100.0, "2099-01-01")], lambda s: _prices()))
    assert "아직 날짜가 지난 체크포인트가 없습니다" in empty


def _payload(meta_time):
    ts = [int(pd.Timestamp(d + " 14:30", tz="UTC").timestamp()) for d in ("2026-11-11", "2026-11-12")]
    return {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "America/New_York", "regularMarketTime": meta_time, "regularMarketPrice": 641.69,
                 "regularMarketDayHigh": 643.9, "regularMarketDayLow": 624.8, "regularMarketVolume": 15_000_000},
        "timestamp": ts,
        "indicators": {"quote": [{"open": [1, 1], "high": [2, 2], "low": [1, 1], "close": [600.0, 610.0], "volume": [5, 5]}],
                       "adjclose": [{"adjclose": [600.0, 610.0]}]}}]}}


def test_missing_last_bar_is_filled_from_meta():
    after_close = int(pd.Timestamp("2026-11-13 16:00", tz="America/New_York").timestamp())
    f = parse_yahoo_chart(_payload(after_close), "META")
    assert f.index[-1] == pd.Timestamp("2026-11-13") and f["adj_close"].iloc[-1] == 641.69
    assert f["high"].iloc[-1] == 643.9 and len(f) == 3


def test_meta_not_used_intraday_or_after_gap():
    intraday = int(pd.Timestamp("2026-11-13 11:00", tz="America/New_York").timestamp())
    assert len(parse_yahoo_chart(_payload(intraday), "META")) == 2
    two_days = int(pd.Timestamp("2026-11-16 16:00", tz="America/New_York").timestamp())   # 11-13 이 빠진 채 월요일
    assert len(parse_yahoo_chart(_payload(two_days), "META")) == 2
    same_day = int(pd.Timestamp("2026-11-12 16:00", tz="America/New_York").timestamp())
    assert parse_yahoo_chart(_payload(same_day), "META")["close"].iloc[-1] == 610.0
