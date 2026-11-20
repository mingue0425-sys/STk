import json
from datetime import datetime, timezone

import pytest

from stk import news

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item><guid>g1</guid><title>Nvidia beats &lt;estimates&gt; @someone</title>
<link>https://www.example.com/a?x=1</link><pubDate>Fri, 20 Nov 2026 17:48:02 +0000</pubDate>
<description>long body not stored</description></item>
<item><guid>g2</guid><title>Bad link</title><link>javascript:alert(1)</link><pubDate>Fri, 20 Nov 2026 16:00:00 +0000</pubDate></item>
<item><guid>g3</guid><title>NVIDIA BEATS &lt;ESTIMATES&gt; @someone</title><link>https://b.com/x</link>
<pubDate>Fri, 20 Nov 2026 15:00:00 +0000</pubDate></item>
<item><guid>g4</guid><title>Older</title><link>http://c.org/y</link><pubDate>Mon, 02 Nov 2026 15:00:00 +0000</pubDate></item>
</channel></rss>"""


def test_parse_rss_filters_and_cleans():
    items = news.parse_rss(RSS, "NVDA")
    assert [i["guid"] for i in items] == ["g1", "g4"]          # javascript: 링크 제외, 같은 제목 중복 제거
    assert items[0]["publisher"] == "example.com" and "description" not in items[0]
    assert items[0]["published"] == "2026-11-20T17:48:02+00:00"
    now = datetime(2026, 11, 20, 20, tzinfo=timezone.utc)
    assert [i["guid"] for i in news.recent(items, now=now)] == ["g1"]   # 10일 지난 기사 제외


def test_parse_rss_rejects_entities():
    evil = b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x "y">]><rss><channel></channel></rss>'
    with pytest.raises(ValueError):
        news.parse_rss(evil, "X")


def test_parse_filings_labels_and_links():
    payload = {"filings": {"recent": {
        "form": ["8-K", "4", "10-Q", "8-K"],
        "filingDate": ["2026-11-17", "2026-11-16", "2026-11-01", "2025-01-01"],
        "accessionNumber": ["0001045810-26-000088", "x", "0001045810-26-000070", "0001045810-25-000001"],
        "primaryDocument": ["nvda-20261117.htm", "a.xml", "nvda-10q.htm", "old.htm"],
        "items": ["2.02,9.01", "", "", "5.02"]}}}
    f = news.parse_filings(payload, "NVDA", 1045810, since="2026-07-01")
    assert [x["form"] for x in f] == ["8-K", "10-Q"]
    assert f[0]["label"] == "수시 공시: 실적 발표"
    assert f[0]["link"] == "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000088/nvda-20261117.htm"


def test_archive_appends_only_new(tmp_path):
    p = tmp_path / "news.jsonl"
    info = {"NVDA": {"news": news.parse_rss(RSS, "NVDA")}}
    assert news.append_archive(p, info) == 2
    assert news.append_archive(p, info) == 0
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    assert {"guid", "symbol", "title", "link", "publisher", "published", "first_seen"} == set(rows[0])


def test_collect_survives_failures_and_skips_sec_without_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(news, "fetch_news", lambda s: (_ for _ in ()).throw(TimeoutError()) if s == "B" else [])
    monkeypatch.setattr(news, "fetch_filings", lambda s, ua: calls.append(ua) or [])
    out = news.collect(["A", "B"], sec_user_agent=None)
    assert out["B"]["errors"] == ["news: TimeoutError"] and out["A"]["errors"] == [] and calls == []
    news.collect(["A"], sec_user_agent="id")
    assert calls == ["id"]


def test_publish_shows_news_and_never_fails_on_it(tmp_path, monkeypatch):
    from stk import publish as pub
    from tests.conftest import make_prices
    from tests.test_v05 import _fake_watch

    data = {"AAA": make_prices(seed=1)}
    _fake_watch(monkeypatch, data)
    now_item = dict(news.parse_rss(RSS, "AAA")[0], published=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    fake = {"AAA": {"news": [now_item], "errors": [],
                    "filings": [{"symbol": "AAA", "date": "2026-11-17", "form": "8-K", "items": ["2.02"],
                                 "label": "수시 공시: 실적 발표", "link": "https://www.sec.gov/x.htm"}]}}
    site = tmp_path / "site"
    res = pub.publish_site(["AAA"], site, [21], {"n_paths": 200}, site / "data" / "f.jsonl", lambda s: data[s],
                           jobs=1, news_fetcher=lambda syms: fake)
    page = (site / "symbols" / "AAA.html").read_text(encoding="utf-8")
    assert "Nvidia beats &lt;estimates&gt;" in page and "수시 공시: 실적 발표" in page and 'rel="noopener' in page
    assert "Nvidia" not in (site / "README.md").read_text(encoding="utf-8")       # 헤드라인은 요약·알림에 넣지 않음
    st = json.loads((site / "data" / "latest.json").read_text(encoding="utf-8"))["news"]
    assert st["headlines"] == 1 and st["filings"] == 1 and st["new_archived"] == 1 and res.render_errors == []

    def broken(syms):
        raise ConnectionError("down")
    res = pub.publish_site(["AAA"], site, [21], {"n_paths": 200}, site / "data" / "f.jsonl", lambda s: data[s],
                           jobs=1, news_fetcher=broken)
    assert res.render_errors == []
    assert json.loads((site / "data" / "latest.json").read_text(encoding="utf-8"))["news"] == {"error": "ConnectionError"}
    assert "관련 뉴스" not in (site / "symbols" / "AAA.html").read_text(encoding="utf-8")


def test_mentions_prefers_direct_headlines():
    assert news.mentions({"symbol": "NVDA", "title": "Why Nvidia Stock Is Down Today"})
    assert news.mentions({"symbol": "NVDA", "title": "NVDA slides"})
    assert not news.mentions({"symbol": "NVDA", "title": "Why Intuit Stock Crashed Today"})
    assert not news.mentions({"symbol": "META", "title": "Metallica tour"})
    assert news.mentions({"symbol": "BRK-B", "title": "Buffett trims stake"})
    assert not news.mentions({"symbol": "V", "title": "V-shaped recovery ahead"})   # 한 글자 티커는 이름으로만
    assert news.mentions({"symbol": "V", "title": "Visa earnings beat"})
