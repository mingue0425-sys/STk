import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stk import data as data_mod
from stk import watch as watch_mod
from stk.cli import main
from stk.data import DataError, _backoff, check_prices, fetch_yahoo
from stk.forecast import run_forecast
from stk.report import render_text
from stk.tracking import evaluate, forecast_record
from stk.watch import WatchItem, compute_changes, render_dashboard, render_markdown_summary

from .conftest import make_prices


# ---------------------------------------------------------------- 데이터 점검
def test_check_prices_clean():
    assert check_prices(make_prices(n=600)) == []


def test_check_prices_flags_problems():
    px = make_prices(n=300)
    s = px["adj_close"].copy()
    s.iloc[150:] *= 2.0  # 하루 +100% 점프
    px["adj_close"] = s
    px = px.drop(px.index[200:215])  # 15 영업일 공백
    w = check_prices(px, today=px.index[-1] + pd.Timedelta(days=30))
    text = " ".join(w)
    assert "±35%" in text and "공백" in text and "영업일 전" in text
    assert any("짧아" in x for x in check_prices(make_prices(n=120)))


def test_data_warnings_in_report():
    px = make_prices(n=150)
    fc = run_forecast(px, "S", horizon=21, n_paths=200, today=px.index[-1] + pd.Timedelta(days=40))
    assert fc.data_warnings
    assert "⚠ 데이터 점검" in render_text(fc)


# ---------------------------------------------------------------- 재시도
class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, *a, **k):
        self.calls += 1
        return self.responses.pop(0)


def _payload():
    return {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [1704205800, 1704292200],
        "indicators": {"quote": [{"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [10.0, 11.0], "volume": [1, 1]}],
                       "adjclose": [{"adjclose": [10.0, 11.0]}]},
    }]}}


def test_backoff_respects_retry_after():
    assert _backoff(0, "7") == 7.0
    assert _backoff(0, "999") == 60.0
    assert 0.5 <= _backoff(1) <= 3.0


def test_fetch_retries_on_429(monkeypatch):
    monkeypatch.setattr(data_mod.time, "sleep", lambda s: None)
    sess = _Session([_Resp(429, headers={"Retry-After": "1"}), _Resp(503), _Resp(200, _payload())])
    df = fetch_yahoo("AAPL", session=sess)
    assert sess.calls == 3 and list(df["close"]) == [10.0, 11.0]


def test_fetch_no_retry_on_client_error(monkeypatch):
    monkeypatch.setattr(data_mod.time, "sleep", lambda s: None)
    sess = _Session([_Resp(404)])
    with pytest.raises(DataError, match="404"):
        fetch_yahoo("NOPE", session=sess)
    sess = _Session([_Resp(400)])
    with pytest.raises(DataError, match="400"):
        fetch_yahoo("BAD", session=sess)
    assert sess.calls == 1


def test_fetch_gives_up(monkeypatch):
    monkeypatch.setattr(data_mod.time, "sleep", lambda s: None)
    sess = _Session([_Resp(429)] * 4)
    with pytest.raises(DataError, match="4회 시도"):
        fetch_yahoo("AAPL", session=sess)


# ---------------------------------------------------------------- 직전 대비 변화 / 요약
def _item(sym, px, horizon=42):
    fc = run_forecast(px, sym, horizon=horizon, n_paths=400)
    return WatchItem(sym, fc, f"{sym}.html", history=px["adj_close"].iloc[-63:].tolist())


def test_compute_changes_and_flip_threshold():
    px = make_prices(seed=7)
    it = _item("AAA", px)
    fc = it.forecast
    base = forecast_record(fc)
    prev_small = dict(base, as_of="2020-01-01", prob_up=fc.mix_prob_up - 0.01, outlook="다른 판정", regime="저변동(안정)")
    ch = compute_changes([it], [prev_small])["AAA"]
    assert ch.d_prob_up == pytest.approx(0.01)
    assert not ch.outlook_changed           # 1%p 변화는 경계값 흔들림으로 간주
    prev_big = dict(prev_small, prob_up=fc.mix_prob_up - 0.08)
    ch = compute_changes([it], [prev_big])["AAA"]
    assert ch.outlook_changed and ch.d_prob_up == pytest.approx(0.08)
    assert ch.regime_changed == (fc.regime.state.names[fc.regime.state.current_regime] != "저변동(안정)")
    # 같은 기준일 기록이나 다른 기간·모델 기록은 비교 대상이 아님
    assert compute_changes([it], [dict(base)]) == {}
    assert compute_changes([it], [dict(prev_big, horizon=999)]) == {}


def test_dashboard_and_markdown_with_changes_and_track():
    full = make_prices(n=900, seed=8)
    it = _item("AAA", full.iloc[:800])
    rec = dict(forecast_record(it.forecast), as_of="2020-01-01")
    changes = compute_changes([it], [rec])
    track = evaluate([forecast_record(it.forecast)], lambda s: full)
    err = WatchItem("ZZZ", None, None, error="ZZZ: 종목을 찾을 수 없습니다")
    page = render_dashboard([it, err], 42, changes=changes, track=track)
    assert "직전 대비" in page and "실전 성과" in page and 'class="delta' in page
    md = render_markdown_summary([it, err], 42, changes=changes, track=track)
    assert "| [AAA](AAA.html)" in md and "%p" in md and "실전 성과" in md and "오류: 종목을 찾을 수 없습니다" in md


def test_forecast_record_has_regime_and_risk(prices):
    rec = forecast_record(run_forecast(prices, "S", horizon=21, n_paths=200))
    assert rec["regime"] and 0 < rec["regime_prob"] <= 1 and rec["var95"] > 0


def test_cli_watch_writes_summary_and_step_summary(tmp_path, monkeypatch):
    data = {"AAA": make_prices(seed=1), "BBB": make_prices(seed=2)}
    monkeypatch.setattr(watch_mod, "get_history", lambda sym, **kw: data[sym])
    monkeypatch.setattr("stk.cli.get_history", lambda sym, **kw: data[sym])
    out = tmp_path / "site"
    step = tmp_path / "step.md"
    log = tmp_path / "log.jsonl"
    args = ["watch", "AAA", "BBB", "--benchmark", "none", "--rf", "0.04", "--no-market", "-H", "21", "-n", "200",
            "--jobs", "1", "-o", str(out), "--no-cache", "--log", str(log), "--summary-md", str(step)]
    assert main(args) == 0
    assert main(args) == 0  # 두 번째 실행: 같은 기준일 → 기록 교체, 직전 비교 없음
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "AAA" in md and "신규" in md
    assert step.read_text(encoding="utf-8").count("# 관심종목 흐름 예측") == 2
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


# ---------------------------------------------------------------- 워크플로 파일
def test_workflows_are_valid_yaml():
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    daily = yaml.safe_load((root / "daily-forecast.yml").read_text(encoding="utf-8"))
    triggers = daily[True] if True in daily else daily["on"]  # YAML 1.1 에서 'on' 은 True 로 읽힘
    assert "schedule" in triggers and "workflow_dispatch" in triggers
    assert daily["permissions"]["contents"] == "write"
    names = [s.get("name") for s in daily["jobs"]["forecast"]["steps"]]
    assert "Publish to reports branch" in names
    ci = yaml.safe_load((root / "ci.yml").read_text(encoding="utf-8"))
    assert ci["jobs"]["test"]["strategy"]["matrix"]["python-version"] == ["3.10", "3.11", "3.12", "3.13"]


def test_summary_json_roundtrip(tmp_path):
    it = _item("AAA", make_prices(seed=3))
    data = json.loads(watch_mod.summary_json([it]))
    assert data[0]["symbol"] == "AAA" and "path" not in data[0]
    assert np.isfinite(data[0]["overall"]["prob_up"])
