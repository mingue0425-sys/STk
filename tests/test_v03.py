import json

import numpy as np
import pandas as pd
import pytest

from stk import watch as watch_mod
from stk.cli import _level_parser, main
from stk.forecast import _first_touch, level_analysis, resolve_levels, risk_stats, run_forecast
from stk.tracking import append_log, evaluate, forecast_record, load_log, render_track
from stk.watch import WatchItem, read_watchlist, render_dashboard, run_watchlist

from .conftest import make_prices


# ---------------------------------------------------------------- 목표가·손절가 / 위험 지표
def test_resolve_levels():
    assert resolve_levels(["+10%", "-20%", 150, "120"], 100.0) == pytest.approx([110.0, 80.0, 150.0, 120.0])
    assert resolve_levels(None, 100.0) == []


def test_first_touch():
    paths = np.array([[100, 101, 105, 111, 108], [100, 99, 95, 90, 92], [100, 100, 100, 100, 100.0]])
    up = _first_touch(paths, 110, upward=True)
    down = _first_touch(paths, 95, upward=False)
    assert up.tolist() == [3, np.inf, np.inf]
    assert down.tolist() == [np.inf, 2, np.inf]


def test_level_analysis_race_and_probabilities():
    paths = np.array([[100, 105, 111, 112], [100, 96, 89, 91], [100, 101, 99, 100.0], [100, 111, 120, 89]])
    w = np.full(4, 0.25)
    cps = [("1", 1), ("3", 3)]
    stats, race = level_analysis(paths, w, 100.0, [110.0], [90.0], cps, intraday=False)
    target, stop = stats
    assert target.prob_touch == pytest.approx(0.5)      # 경로 0, 3
    assert target.prob_end_beyond == pytest.approx(0.25)  # 경로 0 만 만기에 110 이상
    assert stop.prob_touch == pytest.approx(0.5)        # 경로 1, 3
    assert race["target_first"] == pytest.approx(0.5)   # 경로 0, 3(목표가가 먼저)
    assert race["stop_first"] == pytest.approx(0.25)
    assert race["neither"] == pytest.approx(0.25)
    assert sum(race[k] for k in ("target_first", "stop_first", "neither")) == pytest.approx(1.0)
    assert target.prob_by["1"] == pytest.approx(0.25)


def test_risk_stats_known_distribution():
    rng = np.random.default_rng(0)
    terminal = 100 * np.exp(rng.normal(0, 0.2, 200_000))
    paths = np.column_stack([np.full_like(terminal, 100.0), terminal])
    r = risk_stats(paths, np.full(len(terminal), 1 / len(terminal)), 100.0)
    assert r.var95 == pytest.approx(1 - np.exp(-1.645 * 0.2), abs=0.005)
    assert r.cvar95 > r.var95
    assert r.prob_loss_10 == pytest.approx(float(np.mean(terminal <= 90)), abs=1e-9)


def test_forecast_with_levels(prices):
    fc = run_forecast(prices, "S", horizon=63, n_paths=1000, targets=["+10%"], stops=["-10%"])
    assert [lv.kind for lv in fc.levels] == ["target", "stop"]
    for lv in fc.levels:
        assert lv.prob_touch >= lv.prob_end_beyond
        assert list(lv.prob_by.values()) == sorted(lv.prob_by.values())  # 누적 확률은 증가
    assert fc.race is not None and fc.risk is not None
    assert set(fc.cp_quantiles) == {d for _, d in fc.checkpoints}
    assert fc.cp_quantiles[63] == sorted(fc.cp_quantiles[63])


def test_level_parser():
    tgt, stp = _level_parser("목표가"), _level_parser("손절가")
    assert tgt("15%") == "+15%" and tgt("400") == 400.0
    assert stp("10%") == "-10%" and stp("+5%") == "+5%" and stp("-8%") == "-8%"
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        stp("150%")
    with pytest.raises(argparse.ArgumentTypeError):
        tgt("abc")


def test_cli_targets_in_report(tmp_path, prices, capsys):
    csv = tmp_path / "p.csv"
    prices.to_csv(csv)
    code = main(["SYN", "--csv", str(csv), "--benchmark", "none", "--rf", "0.04", "--no-market", "-H", "3m",
                 "-n", "800", "--no-save", "--no-cache", "--target", "12%", "--stop", "8%"])
    assert code == 0
    out = capsys.readouterr().out
    assert "목표가 / 손절가 분석" in out and "먼저 도달" in out and "95% VaR" in out


# ---------------------------------------------------------------- 예측 기록·추적
def test_log_roundtrip_and_dedupe(tmp_path, prices):
    path = tmp_path / "log.jsonl"
    fc = run_forecast(prices, "SYN", horizon=21, n_paths=300)
    append_log(fc, path)
    append_log(fc, path)  # 같은 id → 교체
    recs = load_log(path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["symbol"] == "SYN" and len(rec["checkpoints"]) == len(fc.checkpoints)
    assert len(rec["checkpoints"][0]["q"]) == 19


def test_cli_logs_to_env_path(tmp_path, prices, monkeypatch):
    log = tmp_path / "env_log.jsonl"
    monkeypatch.setenv("STK_LOG", str(log))
    csv = tmp_path / "p.csv"
    prices.to_csv(csv)
    main(["SYN", "--csv", str(csv), "--benchmark", "none", "--rf", "0.04", "--no-market", "-H", "21",
          "-n", "300", "--no-save", "--no-cache"])
    assert len(load_log(log)) == 1
    main(["SYN", "--csv", str(csv), "--benchmark", "none", "--rf", "0.04", "--no-market", "-H", "42",
          "-n", "300", "--no-save", "--no-cache", "--no-log"])
    assert len(load_log(log)) == 1


def test_evaluate_matured_checkpoints():
    full = make_prices(n=900, seed=3)
    cut = full.iloc[:800]
    fc = run_forecast(cut, "SYN", horizon=63, n_paths=2000)
    rec = forecast_record(fc)
    res = evaluate([rec], lambda sym: full)
    ev = res.evaluations
    assert len(ev) == len(fc.checkpoints)  # 900-800=100일 > 63일 → 모두 평가
    row = ev[ev["final"]].iloc[0]
    expected = float(full["adj_close"].loc[:pd.Timestamp(rec["checkpoints"][-1]["date"])].iloc[-1]) / float(cut["adj_close"].iloc[-1]) - 1
    assert row["realized_return"] == pytest.approx(expected)
    assert 0 <= row["pit"] <= 1
    assert res.summary["evaluated"] == len(ev)
    assert "실전 성과" in render_track(res, "x.jsonl")


def test_evaluate_pending_when_future():
    px = make_prices(n=800, seed=4)
    fc = run_forecast(px, "SYN", horizon=63, n_paths=300)
    res = evaluate([forecast_record(fc)], lambda sym: px)
    assert res.summary["evaluated"] == 0 and res.summary["pending"] == len(fc.checkpoints)


def test_cli_track(tmp_path, monkeypatch, capsys):
    full = make_prices(n=900, seed=5)
    log = tmp_path / "t.jsonl"
    append_log(run_forecast(full.iloc[:800], "SYN", horizon=21, n_paths=300), log)
    monkeypatch.setattr("stk.cli.get_history", lambda *a, **k: full)
    assert main(["track", "--log", str(log)]) == 0
    assert "평가된 체크포인트" in capsys.readouterr().out


# ---------------------------------------------------------------- 관심종목
def test_read_watchlist(tmp_path):
    f = tmp_path / "w.txt"
    f.write_text("AAPL, msft  # 메가캡\n\n# 주석만\nbrk.b NVDA\naapl\n", encoding="utf-8")
    assert read_watchlist(f) == ["AAPL", "MSFT", "BRK-B", "NVDA"]


def test_run_watchlist_and_dashboard(tmp_path, monkeypatch):
    data = {"AAA": make_prices(seed=1), "BBB": make_prices(seed=2, mu=-0.3)}

    def fake_history(sym, **kw):
        if sym not in data:
            raise watch_mod.DataError(f"{sym}: 종목을 찾을 수 없습니다")
        return data[sym]

    monkeypatch.setattr(watch_mod, "get_history", fake_history)
    items = run_watchlist(["AAA", "BBB", "ZZZ"], tmp_path, {"horizon": 42, "n_paths": 300}, jobs=1)
    assert [it.symbol for it in items] == ["AAA", "BBB", "ZZZ"]
    assert items[2].forecast is None and "찾을 수 없습니다" in items[2].error
    assert (tmp_path / items[0].report).exists()
    page = render_dashboard(items, 42)
    assert page.startswith("<!doctype html>") and "AAA" in page and "BBB" in page
    assert page.count('class="spark"') == 2 and "오류: 종목을 찾을 수 없습니다" in page


def test_cli_watch_all_fail(tmp_path, monkeypatch):
    def boom(sym, **kw):
        raise watch_mod.DataError(f"{sym}: 종목을 찾을 수 없습니다")

    monkeypatch.setattr(watch_mod, "get_history", boom)
    out = tmp_path / "dash"
    code = main(["watch", "XXX", "YYY", "--benchmark", "none", "--rf", "0.04", "--no-market",
                 "-H", "21", "-n", "100", "--jobs", "1", "-o", str(out), "--no-cache"])
    assert code == 1
    assert not (out / "index.html").exists()


def test_watch_item_error_row():
    from stk.watch import _row

    row = _row(WatchItem("XXX", None, None, error="XXX: 종목을 찾을 수 없습니다"))
    assert "오류: 종목을 찾을 수 없습니다" in row and "XXX: 종목" not in row


def test_cli_watch(tmp_path, monkeypatch, capsys):
    data = {"AAA": make_prices(seed=1), "BBB": make_prices(seed=2)}
    monkeypatch.setattr(watch_mod, "get_history", lambda sym, **kw: data[sym])
    wl = tmp_path / "wl.txt"
    wl.write_text("AAA\nBBB\n", encoding="utf-8")
    out = tmp_path / "dash"
    code = main(["watch", "--file", str(wl), "--benchmark", "none", "--rf", "0.04", "--no-market",
                 "-H", "21", "-n", "300", "--jobs", "1", "-o", str(out), "--no-cache"])
    assert code == 0
    assert (out / "index.html").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [s["symbol"] for s in summary] == ["AAA", "BBB"]
    assert "종목 비교" in capsys.readouterr().out


def test_crossing_probs_formula():
    from stk.forecast import crossing_probs

    logp = np.log(np.array([[100.0, 102.0, 111.0]]))
    var = np.array([[0.02**2]])
    p = crossing_probs(logp, np.log(110.0), True, var)
    a, b = np.log(110 / 100), np.log(110 / 102)
    assert p[0, 0] == pytest.approx(np.exp(-2 * a * b / 0.02**2))
    assert p[0, 1] == 1.0  # 종가가 이미 넘음


def test_intraday_touch_at_least_close_basis():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.02, (4000, 60))
    paths = 100 * np.exp(np.column_stack([np.zeros(4000), np.cumsum(r, axis=1)]))
    w = np.full(4000, 1 / 4000)
    cps = [("e", 60)]
    intr, race_i = level_analysis(paths, w, 100.0, [115.0], [88.0], cps, intraday=True)
    close, race_c = level_analysis(paths, w, 100.0, [115.0], [88.0], cps, intraday=False)
    for a, b in zip(intr, close):
        assert a.prob_touch > b.prob_touch          # 장중 도달 가능성이 더해짐
        assert a.prob_touch_close == pytest.approx(b.prob_touch)
        assert a.prob_end_beyond == pytest.approx(b.prob_end_beyond)
    for race in (race_i, race_c):
        assert race["target_first"] + race["stop_first"] + race["neither"] == pytest.approx(1.0)
    assert race_i["neither"] < race_c["neither"]


def test_intraday_equals_close_for_smooth_paths():
    paths = np.array([np.linspace(100, 120, 41)])  # 변동 없는 직선 경로 → 장중 보정이 거의 없음
    stats, _ = level_analysis(paths, np.array([1.0]), 100.0, [110.0], [], [("e", 40)], intraday=True)
    assert stats[0].prob_touch == pytest.approx(1.0)
    assert stats[0].median_days == pytest.approx(20, abs=1)
