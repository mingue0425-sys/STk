import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from stk.forecast import resolve_events, run_forecast
from stk.html_report import render_html
from stk.market import forward_vol, market_context
from stk.regime import (
    build_regime_state,
    fit_hmm,
    shrink_regime_drifts,
    simulate_regime_paths,
)
from stk.scenarios import DEFAULT_SPEC, assign_probabilities, build_scenarios

from .conftest import make_prices
from .test_simulation import _calib


def _vix_frame(values, index):
    return pd.DataFrame({"close": values}, index=index)


def _market(vix_level, vix3m_level, spy_mu=0.1, seed=0):
    spy = make_prices(n=1500, mu=spy_mu, sigma=0.16, seed=seed)
    idx = spy.index
    rng = np.random.default_rng(seed)
    hist = np.clip(18 + rng.normal(0, 4, len(idx)), 10, 60)
    vix = hist.copy()
    vix[-1] = vix_level
    vix3m = hist + 1.5
    vix3m[-1] = vix3m_level
    return spy, _vix_frame(vix, idx), _vix_frame(vix3m, idx)


def test_market_stress_direction():
    calm = market_context(*_market(11.0, 14.0))
    panic = market_context(*_market(45.0, 33.0, spy_mu=-0.4))
    assert calm.stress < 0 < panic.stress
    assert panic.term_ratio > 1
    assert panic.label in ("불안", "공포/위기")


def test_market_context_is_point_in_time():
    spy, vix, vix3m = _market(30.0, 25.0)
    cutoff = spy.index[1000]
    ctx = market_context(spy, vix, vix3m, as_of=cutoff)
    assert ctx.as_of == cutoff
    assert ctx.vix == pytest.approx(float(vix["close"].iloc[1000]))


def test_implied_vol_and_forward_vol():
    ctx = market_context(*_market(40.0, 30.0))
    short, long = ctx.implied_market_vol(10), ctx.implied_market_vol(126)
    assert short == pytest.approx(0.40 * 0.85)
    assert long < short  # 역전된 기간구조 → 장기 내재변동성이 더 낮음
    fv = forward_vol(0.30, 1.2, ctx, 21)
    assert fv > 0.30  # 시장 공포 → 선행 변동성이 과거보다 높음
    assert forward_vol(0.30, 1.0, None, 21) is None


def test_stress_raises_crash_probability():
    scs = build_scenarios(DEFAULT_SPEC, _calib())
    assign_probabilities(scs, 0.0, stress=-0.8)
    calm = {s.key: s.probability for s in scs}
    assign_probabilities(scs, 0.0, stress=0.9)
    panic = {s.key: s.probability for s in scs}
    assert panic["crash"] > calm["crash"] * 2
    assert panic["dip_recovery"] > calm["dip_recovery"]
    assert sum(panic.values()) == pytest.approx(1.0)


def _two_regime_returns(seed=0):
    rng = np.random.default_rng(seed)
    out, state = [], 0
    for _ in range(3000):
        if rng.random() < (0.01 if state == 0 else 0.03):
            state = 1 - state
        out.append(rng.normal(0.0005, 0.007 if state == 0 else 0.03))
    return np.array(out)


def test_hmm_recovers_regimes():
    r = _two_regime_returns()
    p = fit_hmm(r, k=2)
    vol = np.sqrt(p.vars * 252)
    assert vol[0] == pytest.approx(0.007 * math.sqrt(252), rel=0.15)
    assert vol[1] == pytest.approx(0.03 * math.sqrt(252), rel=0.15)
    assert np.all(np.diag(p.trans) > 0.9)
    assert np.allclose(p.trans.sum(axis=1), 1.0)


def test_regime_state_detects_current_regime():
    r = np.concatenate([_two_regime_returns()[:2500], np.random.default_rng(1).normal(0, 0.035, 40)])
    st = build_regime_state(r, base_drift=0.08, k=2)
    assert st.current_regime == 1
    occ = st.occupancy(252)
    assert occ == pytest.approx(st.stationary, abs=0.05)  # 장기적으로 정상분포에 수렴


def test_shrink_regime_drifts():
    raw = np.array([0.30, 0.05, -1.5])
    vol = np.array([0.15, 0.30, 0.90])
    pi = np.array([0.5, 0.45, 0.05])
    adj = shrink_regime_drifts(raw, vol, pi, n_obs=2500)
    assert float(pi @ adj) == pytest.approx(0.0, abs=1e-12)
    assert abs(adj[2]) < abs(raw[2] - pi @ raw) * 0.1  # 표본이 적고 변동성이 큰 국면은 강하게 축소


def test_regime_simulation_shape_and_vol():
    r = _two_regime_returns()
    st = build_regime_state(r, base_drift=0.08, k=2)
    paths = simulate_regime_paths(100.0, st, 63, 5000, np.random.default_rng(0))
    assert paths.shape == (5000, 64)
    assert np.all(paths > 0)
    assert paths[:, 0].tolist() == [100.0] * 5000


def test_forecast_model_modes(prices):
    sc = run_forecast(prices, "S", horizon=63, n_paths=800, model="scenario")
    rg = run_forecast(prices, "S", horizon=63, n_paths=800, model="regime")
    en = run_forecast(prices, "S", horizon=63, n_paths=800, model="ensemble", regime_weight=0.5)
    assert sc.regime is not None and sc.regime_weight == 0.0  # 참고용 국면 정보
    assert rg.regime_weight == 1.0
    assert rg.mix_terminal_quantiles[0.5] == pytest.approx(rg.regime.terminal_quantiles[0.5], rel=0.01)
    assert en.regime_weight == 0.5
    assert len(en.regime_flow()) == len(en.checkpoints) + 1
    with pytest.raises(ValueError):
        run_forecast(prices, "S", horizon=21, n_paths=100, model="nope")


def test_resolve_events():
    dates = pd.DatetimeIndex(pd.bdate_range("2026-01-02", periods=11))
    days, used = resolve_events([("2026-01-06", 0.05), ("2026-01-10", None), ("2030-01-01", None)], dates, 0.07)
    # 01-06 은 dates[2] → 경로 인덱스 1, 01-10(토) 은 다음 거래일 01-12 = dates[6] → 인덱스 5
    assert days == {1: 0.05, 5: 0.07}
    assert [d.strftime("%m-%d") for d, _ in used] == ["01-06", "01-12"]


def test_event_widens_distribution(prices):
    last = prices.index[-1]
    ev = (last + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    base = run_forecast(prices, "S", horizon=21, n_paths=4000, model="scenario")
    shocked = run_forecast(prices, "S", horizon=21, n_paths=4000, model="scenario", events=[(ev, 0.15)])
    width = lambda f: f.mix_terminal_quantiles[0.95] - f.mix_terminal_quantiles[0.05]
    assert width(shocked) > width(base) * 1.3
    assert len(shocked.events) == 1


def test_render_html(prices, bench):
    spy, vix, vix3m = _market(25.0, 22.0)
    ctx = market_context(spy, vix, vix3m)
    fc = run_forecast(prices, "SYN", horizon=63, n_paths=500, benchmark=bench, market=ctx)
    page = render_html(fc, prices)
    assert page.startswith("<!doctype html>")
    assert "SYN" in page and fc.outlook.label in page
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    data = json.loads(m.group(1))
    assert data["symbol"] == "SYN"
    assert len(data["fut"]["dates"]) == 64
    assert data["regimeFlow"] is not None
