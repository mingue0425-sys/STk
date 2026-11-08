import json

import numpy as np
import pandas as pd
import pytest

from stk import watch as watch_mod
from stk.cli import main
from stk.earnings import fit_earnings
from stk.forecast import EventSpec, run_forecast
from stk.narrative import narrative, narrative_text
from stk.publish import build_alerts, horizon_code, publish_site
from stk.report import render_markdown, render_text
from stk.sector import analyze_sector
from stk.simulation import apply_events
from stk.tracking import forecast_record

from .conftest import make_prices


def _with_earnings(n=2100, offset=30, move=0.07, seed=0):
    """분기말 + offset 일(다음 영업일)마다 거래량 4배·±move 충격이 있는 합성 데이터."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    r = rng.normal(0.0003, 0.012, n)
    vol = rng.lognormal(15, 0.15, n)
    q_ends = pd.date_range("2017-12-31", idx[-1], freq="QE")
    event_days = []
    for qe in q_ends:
        d = qe + pd.Timedelta(days=offset)
        pos = idx.searchsorted(d)
        if 0 < pos < n:
            r[pos] += rng.choice([-1, 1]) * move
            vol[pos] *= 4.0
            event_days.append(idx[pos])
    close = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "adj_close": close, "volume": vol},
                      index=pd.DatetimeIndex(idx, name="date"))
    return df, event_days


def _bench(n=2100, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    close = 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame({"close": close, "adj_close": close, "volume": rng.lognormal(18, 0.1, n)},
                        index=pd.DatetimeIndex(idx, name="date"))


# ---------------------------------------------------------------- 실적일 추정
def test_fit_earnings_recovers_pattern():
    px, events = _with_earnings()
    m = fit_earnings(px, _bench())
    assert m is not None
    assert abs(m.offset_mode - 30) <= 3
    assert m.event_std == pytest.approx(0.07, abs=0.02)
    hits = sum(any(abs((e - p).days) <= 3 for p in m.past_events) for e in events[-12:])
    assert hits >= 10  # 최근 12번 중 대부분을 찾아냄
    as_of = px.index[-1]
    nxt = m.next_dates(as_of, as_of + pd.Timedelta(days=200))
    assert nxt and all(d > as_of for d in nxt)
    q_end = pd.Period(nxt[0], freq="Q").start_time - pd.Timedelta(days=1)
    assert abs((nxt[0] - q_end).days - 30) <= 4


def test_fit_earnings_none_without_pattern_or_volume():
    px = make_prices(n=2100, seed=3)
    px["volume"] = np.random.default_rng(1).lognormal(15, 0.15, len(px))
    assert fit_earnings(px, _bench()) is None             # 패턴 없음
    px2, _ = _with_earnings()
    assert fit_earnings(px2.drop(columns=["volume"]), _bench()) is None  # 거래량 없음


def test_fit_earnings_rejects_stale_pattern():
    px, events = _with_earnings(n=2100)
    stop = events[-6]  # 마지막 약 1년 반 동안은 실적 반응이 사라진 종목
    px.loc[px.index >= stop, "volume"] = px["volume"].iloc[:100].median()
    r = np.log(px["adj_close"]).diff().fillna(0.0)
    calm = px.index >= stop
    r[calm] = np.random.default_rng(5).normal(0.0003, 0.012, int(calm.sum()))
    px["adj_close"] = px["close"] = 100 * np.exp(r.cumsum())
    assert fit_earnings(px.loc[:stop], _bench()) is not None   # 그때까지는 패턴이 뚜렷
    assert fit_earnings(px, _bench()) is None                   # 지금 기준으로는 오래된 패턴 → 표시 안 함


def test_auto_events_in_forecast_and_vol_exclusion():
    px, _ = _with_earnings()
    off = run_forecast(px, "E", horizon=126, n_paths=500, benchmark=_bench(), auto_events=False)
    on = run_forecast(px, "E", horizon=126, n_paths=500, benchmark=_bench(), auto_events=True)
    assert off.events == [] and off.next_earnings and not off.auto_events   # 표시만
    assert on.auto_events and any(e.source == "auto" for e in on.events)
    assert on.stats.vol_long < off.stats.vol_long                          # 실적일 제외 → 평상시 변동성
    assert "실적 발표 추정" in render_text(on)
    assert "다음 실적 발표(추정)" in render_text(off)


def test_apply_events_spread_and_mean():
    rng = np.random.default_rng(0)
    lr = np.zeros((20000, 30))
    apply_events(lr, [(10, 3, 0.10)], rng)
    days = np.nonzero(lr)[1]
    assert days.min() >= 7 and days.max() <= 13
    assert np.exp(lr.sum(axis=1)).mean() == pytest.approx(1.0, abs=0.01)  # 가격 기준 평균 0
    lr2 = np.zeros((100, 5))
    apply_events(lr2, [(4, 3, 0.1)], rng)  # 일부는 기간 밖 → 무시
    assert np.count_nonzero(lr2) < 100


def test_event_spec_unpacks():
    e = EventSpec(pd.Timestamp("2027-01-29"), 0.05, 2, "auto")
    d, s = e
    assert s == 0.05 and "±2거래일" in e.label() and "실적 발표 추정" in e.label()


# ---------------------------------------------------------------- 섹터
def _sector_world(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    m = rng.normal(0.0004, 0.010, n)
    fa, fb = rng.normal(0, 0.008, n), rng.normal(0, 0.008, n)

    def frame(r):
        c = 100 * np.exp(np.cumsum(r))
        return pd.DataFrame({"close": c, "adj_close": c, "volume": 1e6}, index=pd.DatetimeIndex(idx, name="date"))

    stock = frame(1.1 * m + 0.9 * fa + rng.normal(0, 0.008, n))
    return stock, frame(m), {"XLK": frame(m + fa), "XLF": frame(m + fb)}


def test_sector_detection_and_decomposition():
    stock, bench, sectors = _sector_world()
    s = analyze_sector(stock, bench, sectors)
    assert s.etf == "XLK" and s.method == "residual" and s.fit > 0.5
    assert s.share_market + s.share_sector + s.share_idio == pytest.approx(1.0)
    assert s.beta_market == pytest.approx(1.1, abs=0.15)
    assert analyze_sector(stock, bench, {}) is None


def test_forecast_with_sector_reports(prices):
    stock, bench, sectors = _sector_world(n=900)
    fc = run_forecast(stock, "S", horizon=21, n_paths=300, benchmark=bench, sectors=sectors)
    assert fc.sector is not None and fc.sector.etf == "XLK"
    assert "■ 섹터 분석" in render_text(fc) and "위험 분해" in render_markdown(fc)
    assert json.loads(json.dumps(fc.to_dict(), default=float))["sector"]["etf"] == "XLK"


# ---------------------------------------------------------------- 해설
def test_narrative_mentions_key_facts(prices):
    fc = run_forecast(prices, "SYN", horizon=63, n_paths=500, targets=["+10%"], stops=["-10%"])
    sents = narrative(fc)
    text = narrative_text(fc)
    assert 3 <= len(sents) <= 10
    assert text.startswith("SYN의 향후 3개월 흐름은") and fc.outlook.label in text
    assert "목표가" in text and "90% 확률" in text
    assert "■ 요약 해설" in render_text(fc)


# ---------------------------------------------------------------- 사이트 생성
def test_horizon_code():
    assert horizon_code(21) == "1m" and horizon_code(126) == "6m" and horizon_code(30) == "30d"


def _fake_watch(monkeypatch, data):
    monkeypatch.setattr(watch_mod, "get_history", lambda sym, **kw: data[sym])


def test_publish_site_structure_and_alerts(tmp_path, monkeypatch):
    data = {"AAA": make_prices(seed=1), "BBB": make_prices(seed=2, mu=-0.2)}
    _fake_watch(monkeypatch, data)
    site, log = tmp_path / "site", tmp_path / "site" / "data" / "f.jsonl"
    res = publish_site(["AAA", "BBB"], site, [21, 63], {"n_paths": 300}, log, lambda s: data[s], jobs=1)
    for f in ["index.html", "README.md", ".nojekyll", "latest/1m/index.html", "latest/3m/index.html",
              "latest/1m/summary.md", "data/f.jsonl"]:
        assert (site / f).exists(), f
    assert len(list((site / "history").glob("*.md"))) == 1
    assert not (site / "alerts.md").exists()  # 첫 실행: 직전 예측 없음 → 알림 없음
    readme = (site / "README.md").read_text(encoding="utf-8")
    assert "](latest/1m/AAA_" in readme and "1개월 전망" in readme
    assert 'href="../3m/index.html"' in (site / "latest/1m/index.html").read_text(encoding="utf-8")
    # 직전 예측(더 이른 기준일, 다른 판정)을 기록에 넣고 다시 실행 → 흐름 변화 알림
    rec = forecast_record(res.runs[0].items[0].forecast)
    rec.update(as_of="2020-01-01", prob_up=rec["prob_up"] - 0.2, outlook="뚜렷한 하락 흐름",
               id="AAA|2020-01-01|21|ensemble")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res2 = publish_site(["AAA", "BBB"], site, [21, 63], {"n_paths": 300}, log, lambda s: data[s], jobs=1)
    assert any("AAA" in a and "흐름 판정" in a for a in res2.alerts)
    assert "AAA" in (site / "alerts.md").read_text(encoding="utf-8")


def test_build_alerts_empty_without_changes():
    assert build_alerts([]) == []


def test_build_alerts_regime_once_per_symbol():
    from types import SimpleNamespace as NS

    from stk.publish import HorizonRun
    from stk.watch import WatchItem

    state = NS(names=["저변동(안정)", "중간 변동", "고변동(불안)"], current_regime=2, current=[0.0, 0.1, 0.9])
    fc = NS(symbol="AAA", regime=NS(state=state), outlook=NS(label="횡보"), data_warnings=[])
    ch = NS(outlook_changed=False, regime_changed=True, prev_regime="중간 변동")
    runs = [HorizonRun(h, str(h), [WatchItem("AAA", fc, "a.html")], {"AAA": ch}) for h in (21, 126)]
    alerts = build_alerts(runs)
    assert alerts == ["AAA: '고변동(불안)' 국면 진입 (90%, 직전 중간 변동)"]


def test_cli_publish(tmp_path, monkeypatch, capsys):
    data = {"AAA": make_prices(seed=1)}
    _fake_watch(monkeypatch, data)
    monkeypatch.setattr("stk.cli.get_history", lambda sym, **kw: data[sym])
    site = tmp_path / "site"
    step = tmp_path / "step.md"
    code = main(["publish", "AAA", "--horizons", "1m,2m", "--site", str(site), "--benchmark", "none", "--rf", "0.04",
                 "--no-market", "-n", "200", "--jobs", "1", "--no-cache", "--summary-md", str(step)])
    assert code == 0
    assert (site / "latest" / "1m" / "index.html").exists() and (site / "latest" / "42d" / "index.html").exists()
    assert "관심종목 흐름 예측" in step.read_text(encoding="utf-8")
    assert "사이트:" in capsys.readouterr().out
    assert main(["publish", "AAA", "--horizons", "xx", "--site", str(site)]) == 2


# ---------------------------------------------------------------- 늦은 데이터 재시도 / 실행 상태
def test_stale_symbol_is_refetched(tmp_path, monkeypatch):
    full = {s: make_prices(n=600, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])}
    calls = []

    def fake(sym, refresh=False, **kw):
        calls.append((sym, refresh))
        if sym == "CCC" and not refresh:
            return full[sym].iloc[:-1]       # 처음엔 마지막 날 봉이 아직 없음
        return full[sym]

    monkeypatch.setattr(watch_mod, "get_history", fake)
    monkeypatch.setattr(watch_mod, "STALE_RETRY_WAIT", 0.0)
    items = watch_mod.run_watchlist(["AAA", "BBB", "CCC"], tmp_path, {"horizon": 21, "n_paths": 200}, jobs=1)
    assert ("CCC", True) in calls and ("AAA", True) not in calls
    last = {it.symbol: it.forecast.stats.last_date for it in items}
    assert len(set(last.values())) == 1                        # 재시도로 따라잡음
    assert not any("늦습니다" in w for it in items for w in it.forecast.data_warnings)
    assert sorted(p.name for p in tmp_path.glob("CCC_*.html")) == [items[2].report]  # 오래된 리포트 삭제


def test_stale_symbol_still_late_gets_warning(tmp_path, monkeypatch):
    full = {s: make_prices(n=600, seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])}
    monkeypatch.setattr(watch_mod, "get_history",
                        lambda sym, **kw: full[sym].iloc[:-2] if sym == "CCC" else full[sym])
    monkeypatch.setattr(watch_mod, "STALE_RETRY_WAIT", 0.0)
    items = watch_mod.run_watchlist(["AAA", "BBB", "CCC"], tmp_path, {"horizon": 21, "n_paths": 200}, jobs=1)
    ccc = items[2].forecast
    assert any("2거래일 늦습니다" in w for w in ccc.data_warnings)
    assert "늦습니다" in (tmp_path / items[2].report).read_text(encoding="utf-8")  # 개별 리포트에도 표시
    assert watch_mod.reference_date([]) is None


def test_publish_names_history_by_data_date(tmp_path, monkeypatch):
    from datetime import datetime

    data = {"AAA": make_prices(seed=1)}
    _fake_watch(monkeypatch, data)
    site = tmp_path / "site"
    publish_site(["AAA"], site, [21], {"n_paths": 200}, site / "data" / "f.jsonl", lambda s: data[s], jobs=1,
                 generated=datetime(2031, 1, 4, 2, 0))   # 데이터 마지막 날보다 한참 뒤에 실행
    last = data["AAA"].index[-1]
    assert [p.name for p in (site / "history").glob("*.md")] == [f"{last:%Y-%m-%d}.md"]
    page = (site / "history" / f"{last:%Y-%m-%d}.html").read_text(encoding="utf-8")
    assert "<table>" in page and "AAA" in page and "AAA_" not in page          # 지난 리포트 링크는 글자만
    assert f'href="{last:%Y-%m-%d}.html"' in (site / "history" / "index.html").read_text(encoding="utf-8")
    assert 'href="history/index.html"' in (site / "index.html").read_text(encoding="utf-8")
    status = json.loads((site / "data" / "latest.json").read_text(encoding="utf-8"))
    assert status["as_of"] == f"{last:%Y-%m-%d}" and status["horizons"]["1m"] == {"ok": 1, "total": 1}
    assert status["generated"].startswith("2031-01-04")


def test_md_to_html_handles_summary_markdown():
    from stk.publish import md_to_html

    md = ("# 제목\n\n기준일 **2026-11-06** · 생성\n\n| 종목 | 값 |\n|---|---:|\n| [AAA](latest/1m/AAA_x.html) | 1 <b> |\n"
          "\n## 알림\n\n- 하나\n- 둘\n\n---\n_면책_")
    h = md_to_html(md)
    assert "<h1>제목</h1>" in h and "<strong>2026-11-06</strong>" in h
    assert "<th>종목</th>" in h and "<td>AAA</td>" in h and "&lt;b&gt;" in h and "---:" not in h
    assert "<ul><li>하나</li><li>둘</li></ul>" in h and "<hr>" in h and "<em>면책</em>" in h
