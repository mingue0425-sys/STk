import math

import numpy as np
import pandas as pd
import pytest

from stk.analysis import (
    analyze,
    combine_signals,
    estimate_beta,
    estimate_tail_df,
    ewma_vol,
    log_returns,
    rsi,
    trend_regression,
)
from stk.data import DataError, load_csv, parse_yahoo_chart

from .conftest import make_prices


def test_trend_regression_recovers_slope():
    y = np.log(100) + 0.001 * np.arange(200)
    slope, r2 = trend_regression(y)
    assert slope == pytest.approx(0.001 * 252)
    assert r2 == pytest.approx(1.0)


def test_ewma_vol_close_to_true_vol():
    px = make_prices(n=3000, sigma=0.30, seed=1)
    vol = ewma_vol(log_returns(px["adj_close"]))
    assert 0.2 < vol < 0.42


def test_rsi_bounds():
    up = pd.Series(np.linspace(100, 200, 50))
    down = pd.Series(np.linspace(200, 100, 50))
    assert rsi(up) == 100.0
    assert rsi(down) < 5


def test_beta_estimate():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=600)
    b = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    s = 1.5 * b + pd.Series(rng.normal(0, 0.005, 600), index=idx)
    assert estimate_beta(s, b) == pytest.approx(1.5, abs=0.1)


def test_tail_df_detects_fat_tails():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2015-01-01", periods=2500)
    normal = pd.Series(rng.normal(0, 0.01, 2500), index=idx)
    fat = pd.Series(rng.standard_t(3.5, 2500) * 0.007, index=idx)
    assert estimate_tail_df(normal) > estimate_tail_df(fat)
    assert estimate_tail_df(fat) < 8


def test_trend_score_direction():
    up = analyze(make_prices(mu=0.6, sigma=0.15, seed=5), "UP")
    down = analyze(make_prices(mu=-0.5, sigma=0.15, seed=5), "DOWN")
    assert up.trend_score > 0.3
    assert down.trend_score < -0.3
    assert "상승" in up.trend_label
    assert "하락" in down.trend_label


def test_combine_signals_ignores_nan():
    assert combine_signals({"ma200_gap": float("nan"), "momentum": 0.5}) == pytest.approx(0.5)
    assert combine_signals({}) == 0.0


def test_analyze_requires_minimum_history():
    with pytest.raises(ValueError):
        analyze(make_prices(n=30), "SHORT")


def test_analyze_short_history_still_works():
    st = analyze(make_prices(n=120), "NEW")
    assert math.isfinite(st.trend_score)
    assert math.isnan(st.signals["momentum"])


def test_analyze_with_benchmark(prices, bench):
    st = analyze(prices, "X", benchmark=bench)
    assert st.beta is not None
    assert st.last_price == pytest.approx(float(prices["adj_close"].iloc[-1]))


def test_parse_yahoo_chart():
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"exchangeTimezoneName": "America/New_York"},
                    "timestamp": [1704205800, 1704292200, 1704378600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [1, 2, 3],
                                "high": [1, 2, 3],
                                "low": [1, 2, 3],
                                "close": [10.0, None, 12.0],
                                "volume": [5, 6, 7],
                            }
                        ],
                        "adjclose": [{"adjclose": [9.0, None, 11.0]}],
                    },
                }
            ],
            "error": None,
        }
    }
    df = parse_yahoo_chart(payload, "TEST")
    assert list(df["close"]) == [10.0, 12.0]
    assert list(df["adj_close"]) == [9.0, 11.0]
    assert str(df.index[0].date()) == "2024-01-02"


def test_parse_yahoo_error():
    with pytest.raises(DataError):
        parse_yahoo_chart({"chart": {"result": None, "error": {"description": "No data found"}}}, "BAD")


def test_load_csv(tmp_path, prices):
    path = tmp_path / "p.csv"
    out = prices.rename(columns={"adj_close": "Adj Close", "close": "Close"})
    out.index.name = "Date"
    out.to_csv(path)
    df = load_csv(path)
    assert len(df) == len(prices)
    assert df["adj_close"].iloc[-1] == pytest.approx(prices["adj_close"].iloc[-1])
