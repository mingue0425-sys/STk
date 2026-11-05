import json
import math

import numpy as np
import pytest

from stk.backtest import crps_normal, crps_weighted_samples, walk_forward
from stk.forecast import QUANTILES, run_forecast, weighted_quantiles
from stk.report import render_markdown, render_text
from stk.scenarios import (
    DEFAULT_SPEC,
    Calibration,
    Phase,
    Scenario,
    assign_probabilities,
    build_scenarios,
)
from stk.simulation import daily_parameters, phase_lengths, simulate_paths


def _calib(**kw):
    base = dict(
        risk_free=0.04, erp=0.05, beta=1.0, capm_drift=0.09, trend_drift=0.09, trend_weight=0.2,
        base_drift=0.09, spread=0.12, vol_now=0.25, vol_long=0.25, vol_halflife=42, tail_df=30.0,
    )
    base.update(kw)
    return Calibration(**base)


def _single(drift=0.10, **phase_kw):
    return Scenario("s", "s", "", 1.0, 0.0, [Phase(frac=1.0, drift=drift, **phase_kw)])


def test_phase_lengths_sum_to_horizon():
    for fr in ([1.0], [0.35, 0.65], [0.2, 0.3, 0.5], [0.9, 0.9, 0.1]):
        for h in (1, 5, 21, 126, 252):
            fr_n = [f / sum(fr) for f in fr]
            lengths = phase_lengths(fr_n, h)
            assert sum(lengths) == h
            assert all(n >= 0 for n in lengths)


def test_gbm_moments():
    rng = np.random.default_rng(0)
    calib = _calib()
    paths = simulate_paths(100.0, _single(0.10), calib, 252, 40_000, rng)
    terminal = paths[:, -1]
    assert terminal.mean() == pytest.approx(100 * math.exp(0.10), rel=0.01)
    assert np.log(terminal / 100).std() == pytest.approx(0.25, rel=0.03)


def test_student_t_keeps_variance():
    rng = np.random.default_rng(1)
    paths = simulate_paths(100.0, _single(0.0), _calib(tail_df=4.0), 252, 40_000, rng)
    assert np.log(paths[:, -1] / 100).std() == pytest.approx(0.25, rel=0.05)


def test_vol_mean_reversion():
    calib = _calib(vol_now=0.60, vol_long=0.20, vol_halflife=21)
    prm = daily_parameters(_single(), calib, 252)
    assert prm["sigma"][0] == pytest.approx(0.60)
    assert prm["sigma"][21] == pytest.approx(math.sqrt(0.2**2 + (0.6**2 - 0.2**2) / 2))
    assert prm["sigma"][-1] == pytest.approx(0.20, rel=0.01)


def test_jumps_lower_terminal_prices():
    calib = _calib()
    no_jump = simulate_paths(100.0, _single(0.1), calib, 252, 20_000, np.random.default_rng(2))
    jumps = simulate_paths(
        100.0, _single(0.1, jump_rate=4, jump_mean=-0.10, jump_std=0.03), calib, 252, 20_000,
        np.random.default_rng(2),
    )
    assert jumps[:, -1].mean() < no_jump[:, -1].mean() * 0.75


def test_multi_phase_path_shape():
    sc = Scenario("v", "v", "", 1, 0, [Phase(0.4, -0.8, 0.5), Phase(0.6, 0.8, 0.5)])
    paths = simulate_paths(100.0, sc, _calib(), 100, 5000, np.random.default_rng(3))
    med = np.median(paths, axis=0)
    trough = int(np.argmin(med))
    assert 30 <= trough <= 50
    assert med[-1] > med[trough]


def test_build_scenarios_resolves_drift():
    calib = _calib(base_drift=0.10, spread=0.15)
    spec = {
        "scenarios": [
            {"key": "a", "phases": [{"drift_spread": 1.0}]},
            {"key": "b", "phases": [{"drift": -0.3}]},
            {"key": "c", "drift_spread": -2.0, "vol_mult": 1.5},  # 단일 국면 축약형
        ]
    }
    scs = build_scenarios(spec, calib)
    assert scs[0].phases[0].drift == pytest.approx(0.25)
    assert scs[1].phases[0].drift == pytest.approx(-0.3)
    assert scs[2].phases[0].drift == pytest.approx(-0.2)
    assert scs[2].phases[0].vol_mult == 1.5


def test_build_scenarios_rejects_bad_input():
    calib = _calib()
    with pytest.raises(ValueError):
        build_scenarios({"scenarios": [{"key": "a", "phases": [{"drfit": 0.1}]}]}, calib)
    with pytest.raises(ValueError):
        build_scenarios({"scenarios": [{"key": "a"}, {"key": "a"}]}, calib)


def test_assign_probabilities_tilt():
    scs = build_scenarios(DEFAULT_SPEC, _calib())
    assign_probabilities(scs, trend_score=0.8)
    p_up = {s.key: s.probability for s in scs}
    assign_probabilities(scs, trend_score=-0.8)
    p_down = {s.key: s.probability for s in scs}
    assert sum(p_up.values()) == pytest.approx(1.0)
    assert p_up["bull"] > p_down["bull"]
    assert p_up["bear"] < p_down["bear"]
    assign_probabilities(scs, trend_score=0.8, fixed=True)
    assert scs[1].probability == pytest.approx(0.40)


def test_weighted_quantiles_matches_numpy():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20_001, 3))
    w = np.ones(20_001)
    wq = weighted_quantiles(x, w, (0.05, 0.5, 0.95))
    nq = np.quantile(x, (0.05, 0.5, 0.95), axis=0)
    assert np.allclose(wq, nq, atol=0.02)


def test_weighted_quantiles_respects_weights():
    x = np.array([0.0] * 100 + [10.0] * 100)
    w = np.array([0.9 / 100] * 100 + [0.1 / 100] * 100)
    assert weighted_quantiles(x, w, (0.5,))[0] == 0.0
    assert weighted_quantiles(x, w, (0.95,))[0] == 10.0


def test_crps_weighted_matches_normal():
    rng = np.random.default_rng(6)
    s = rng.normal(0.02, 0.1, 100_000)
    w = np.ones_like(s)
    assert crps_weighted_samples(s, w, 0.05) == pytest.approx(crps_normal(0.02, 0.1, 0.05), rel=0.02)


def test_run_forecast_end_to_end(prices, bench):
    fc = run_forecast(prices, "SYN", horizon=126, n_paths=2000, benchmark=bench, risk_free=0.04, seed=1)
    assert len(fc.dates) == 127
    assert sum(s.probability for s in fc.scenarios) == pytest.approx(1.0)
    for i in range(len(QUANTILES) - 1):
        assert np.all(fc.mix_quantiles[QUANTILES[i]] <= fc.mix_quantiles[QUANTILES[i + 1]] + 1e-9)
    assert fc.mix_quantiles[0.5][0] == pytest.approx(fc.s0)
    assert 0 <= fc.mix_prob_up <= 1
    assert [c[1] for c in fc.checkpoints] == [5, 21, 63, 126]
    assert fc.dates[1] > fc.dates[0] and fc.dates[1].weekday() < 5
    json.dumps(fc.to_dict(), default=float)  # 직렬화 가능
    text = render_text(fc)
    assert "전체 흐름 전망" in text and fc.outlook.label in text
    assert "## 시나리오별 결과" in render_markdown(fc, "chart.png")


def test_run_forecast_is_reproducible(prices):
    a = run_forecast(prices, "SYN", horizon=21, n_paths=500, seed=7)
    b = run_forecast(prices, "SYN", horizon=21, n_paths=500, seed=7)
    assert a.mix_expected_price == b.mix_expected_price


def test_outlook_follows_drift():
    from .conftest import make_prices

    px = make_prices(mu=0.1, sigma=0.2, seed=8)
    bull = {"scenarios": [{"key": "x", "phases": [{"drift": 0.8}]}]}
    bear = {"scenarios": [{"key": "x", "phases": [{"drift": -0.8}]}]}
    up = run_forecast(px, "S", horizon=126, n_paths=2000, spec=bull)
    down = run_forecast(px, "S", horizon=126, n_paths=2000, spec=bear)
    assert up.outlook.direction > 0 > down.outlook.direction


def test_walk_forward_small(prices):
    bt = walk_forward(prices, "SYN", horizon=21, step=63, years=2, n_paths=300)
    s = bt.summary
    assert s["n"] > 3
    assert 0 <= s["coverage90"] <= 1
    assert set(bt.records.columns) >= {"pit", "in50", "in90", "crps", "crps_naive"}
