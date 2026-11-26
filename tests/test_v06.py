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


def test_matured_final_checkpoint_renders_everywhere(tmp_path):
    from stk.tracking import render_track

    rec = _record("AAA", "2026-11-02", 7, 100.0, "2026-11-04")
    rec["checkpoints"][1]["date"] = "2026-11-11"          # 만기 도달
    tr = evaluate([rec, _record("BBB", "2026-11-02", 21, 100.0, "2026-11-09")], lambda s: _prices())
    assert tr.summary["final_n"] == 1 and tr.evaluations["hit"].notna().sum() == 1
    bd = breakdown(tr.evaluations)
    assert bd["hit"].notna().sum() == 1
    assert "방향 적중" in render_track(tr, tmp_path / "f.jsonl")
    page = render_track_page(tr)
    assert "만기" in page and page.count('class="sm"') == 2


# ---- v0.7: 종목별 기록 페이지 · 부가 페이지 실패 보호 · meta 보정 표시 ----

def _logrec(sym, as_of, horizon, prob, outlook="뚜렷한 방향 우위 없음", regime="중간 변동"):
    r = _record(sym, as_of, horizon, 100.0, "2099-01-01")
    r.update(prob_up=prob, expected_return=prob - 0.5, outlook=outlook, regime=regime)
    return r


def test_symbol_page_single_record_and_gaps():
    from stk.symbol_page import render_symbol_index, render_symbol_page

    one = render_symbol_page("AAA", [_logrec("AAA", "2026-11-04", 21, 0.55)], None)
    assert "AAA" in one and "아직 평가할 날짜가 지나지 않았습니다" in one and "<svg" in one
    recs = [_logrec("BBB", d, h, p, outlook=o, regime=g) for d, h, p, o, g in [
        ("2026-11-04", 63, 0.5, "횡보", "중간 변동"),           # 예전 버전 기간: 그리지 않음
        ("2026-11-05", 21, 0.52, "횡보", "중간 변동"), ("2026-11-05", 126, 0.6, "상승", "중간 변동"),
        ("2026-11-09", 21, 0.62, "상승", "저변동"), ("2026-11-09", 126, 0.61, "상승", "저변동")]]   # 11-06 빠짐
    page = render_symbol_page("BBB", recs, None)
    assert page.count('class="pl"') == 2                      # 21·126일 선만 (63일은 최근 기준일에 없음)
    assert "판정 변화" in page and "국면 변화" in page and "3개월 판정" not in page
    assert page.index("2026-11-09") < page.index("2026-11-05")   # 표는 최근 순
    idx = render_symbol_index({"BBB": recs})
    assert 'href="BBB.html"' in idx and "1개월 62%" in idx


def test_publish_survives_page_failure_and_writes_symbol_pages(tmp_path, monkeypatch):
    import json

    from stk import publish as pub
    from tests.conftest import make_prices
    from tests.test_v05 import _fake_watch

    data = {"AAA": make_prices(seed=1), "BBB": make_prices(seed=2)}
    _fake_watch(monkeypatch, data)
    site = tmp_path / "site"
    res = pub.publish_site(["AAA", "BBB"], site, [21], {"n_paths": 200}, site / "data" / "f.jsonl", lambda s: data[s], jobs=1)
    assert res.render_errors == []
    assert (site / "symbols" / "AAA.html").exists() and (site / "symbols" / "index.html").exists()
    assert 'href="../../symbols/AAA.html"' in (site / "latest/1m/index.html").read_text(encoding="utf-8")
    assert 'href="symbols/index.html"' in (site / "index.html").read_text(encoding="utf-8")

    def boom(*a, **k):
        raise RuntimeError("차트 오류")
    monkeypatch.setattr(pub, "render_track_page", boom)
    monkeypatch.setattr(pub, "write_symbol_pages", boom)
    res = pub.publish_site(["AAA", "BBB"], site, [21], {"n_paths": 200}, site / "data" / "f.jsonl", lambda s: data[s], jobs=1)
    assert len(res.render_errors) == 2 and "차트 오류" in res.render_errors[0]
    status = json.loads((site / "data" / "latest.json").read_text(encoding="utf-8"))
    assert len(status["render_errors"]) == 2 and status["meta_patched"] == []
    assert len((site / "data" / "f.jsonl").read_text(encoding="utf-8").splitlines()) == 2   # 기록은 그대로 저장
    assert (site / "index.html").exists() and (site / "README.md").exists()


def test_meta_patched_marker_survives_csv(tmp_path):
    from stk.watch import is_meta_patched

    after_close = int(pd.Timestamp("2026-11-13 16:00", tz="America/New_York").timestamp())
    f = parse_yahoo_chart(_payload(after_close), "META")
    assert is_meta_patched(f)
    f.to_csv(tmp_path / "c.csv")
    assert is_meta_patched(pd.read_csv(tmp_path / "c.csv", index_col="date", parse_dates=["date"]))
    assert not is_meta_patched(parse_yahoo_chart(_payload(after_close - 86400 * 2), "META"))


def test_cli_publish_returns_3_when_only_pages_fail(tmp_path, monkeypatch, capsys):
    from stk import publish as pub
    from stk.cli import main
    from tests.conftest import make_prices
    from tests.test_v05 import _fake_watch

    data = {"AAA": make_prices(seed=1)}
    _fake_watch(monkeypatch, data)
    monkeypatch.setattr("stk.cli.get_history", lambda sym, **kw: data[sym])

    def boom(*a, **k):
        raise RuntimeError("페이지 오류")
    monkeypatch.setattr(pub, "write_symbol_pages", boom)
    site = tmp_path / "site"
    code = main(["publish", "AAA", "--horizons", "1m", "--site", str(site), "--benchmark", "none", "--rf", "0.04",
                 "--no-market", "-n", "200", "--jobs", "1", "--no-cache", "--no-news"])
    assert code == 3 and "페이지 생성 실패" in capsys.readouterr().err
    assert (site / "data" / "forecasts.jsonl").exists() and (site / "index.html").exists()


def test_unfinished_intraday_bar_is_dropped():
    # 마지막 봉(11-12)과 같은 날 장중(11:00) 시각 → 오늘 봉은 아직 끝나지 않음
    intraday_same_day = int(pd.Timestamp("2026-11-12 11:00", tz="America/New_York").timestamp())
    f = parse_yahoo_chart(_payload(intraday_same_day), "META")
    assert f.index[-1] == pd.Timestamp("2026-11-11") and len(f) == 1
    closed = int(pd.Timestamp("2026-11-12 16:00", tz="America/New_York").timestamp())
    assert parse_yahoo_chart(_payload(closed), "META").index[-1] == pd.Timestamp("2026-11-12")


def test_early_close_day_bar_is_kept_and_patched():
    def ny(x):
        return int(pd.Timestamp(x, tz="America/New_York").timestamp())

    # 11-27(금) 13:00 조기 폐장: 마감 후 시각이 13:00 이어도 장이 끝난 것으로 봐야 함
    period = {"regular": {"start": ny("2026-11-27 09:30"), "end": ny("2026-11-27 13:00")}}
    pay = _payload(ny("2026-11-27 13:00"))
    pay["chart"]["result"][0]["timestamp"] = [ny("2026-11-25 09:30"), ny("2026-11-27 09:30")]
    pay["chart"]["result"][0]["meta"]["currentTradingPeriod"] = period
    assert parse_yahoo_chart(pay, "SPY").index[-1] == pd.Timestamp("2026-11-27")      # 완성된 봉 유지
    pay["chart"]["result"][0]["meta"]["regularMarketTime"] = ny("2026-11-27 11:00")   # 조기 폐장일 장중
    assert parse_yahoo_chart(pay, "SPY").index[-1] == pd.Timestamp("2026-11-25")
    # 일봉에 11-27 이 빠졌고 meta 에 13:00 종가가 있으면 채움
    pay2 = _payload(ny("2026-11-27 13:00"))
    pay2["chart"]["result"][0]["timestamp"] = [ny("2026-11-24 09:30"), ny("2026-11-25 09:30")]
    pay2["chart"]["result"][0]["meta"]["currentTradingPeriod"] = period
    f = parse_yahoo_chart(pay2, "SPY")
    assert f.index[-1] == pd.Timestamp("2026-11-27")   # 11-26 은 휴장 → 11-27 이 11-25 의 바로 다음 거래일이므로 채움
    assert f["close"].iloc[-1] == 641.69 and pd.isna(f["open"].iloc[-1])
