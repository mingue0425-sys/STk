import numpy as np
import pandas as pd
import pytest


def make_prices(n=1500, mu=0.10, sigma=0.25, seed=0, start="2020-01-01", s0=100.0):
    """기하 브라운 운동으로 합성 일봉 데이터 생성."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = s0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
        },
        index=pd.DatetimeIndex(idx, name="date"),
    )


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def bench():
    return make_prices(seed=99, mu=0.08, sigma=0.18)


@pytest.fixture(autouse=True)
def _isolated_forecast_log(tmp_path, monkeypatch):
    """테스트 중 예측 기록이 사용자 홈에 쌓이지 않도록 임시 파일로 돌립니다."""
    monkeypatch.setenv("STK_LOG", str(tmp_path / "forecasts.jsonl"))
