"""종목별 관련 뉴스 헤드라인(Yahoo Finance RSS)과 SEC 공시(8-K·10-Q·10-K) — 참고 표시용.

예측 계산에는 쓰지 않습니다. 뉴스가 주가 방향을 맞히는 데 도움이 되는지는 검증된 적이 없어서,
우선 헤드라인을 매일 그대로 쌓아 두고(data/news.jsonl) 충분히 모이면 사전 기준을 정해 검증합니다.

- 뉴스: 제목·링크·게시 시각·출처(도메인)만 저장/표시 (본문·요약은 쓰지 않음)
- 공시: data.sec.gov 제출 목록 API. SEC 는 요청자 식별(User-Agent)을 요구하므로 환경변수
  STK_SEC_USER_AGENT 가 있을 때만 사용합니다 (GitHub Actions 에서는 저장소 Secret 으로 전달 — 코드에 넣지 않음).
- 모든 실패는 종목 단위로 삼키고 개수만 보고합니다. 뉴스가 안 돼도 예측 실행은 실패하지 않습니다.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ENV = "STK_SEC_USER_AGENT"
MAX_BYTES = 2_000_000
MAX_ITEMS = 30
NEWS_DAYS = 10            # 이보다 오래된 헤드라인은 표시하지 않음
FILING_DAYS = 120
FILING_FORMS = ("8-K", "10-Q", "10-K")

# 관심종목 SEC 고유번호(CIK) — data.sec.gov 로 하나씩 확인함 (www.sec.gov 의 전체 목록 파일은 자동 요청을 막음)
CIK = {
    "AAPL": 320193, "MSFT": 789019, "GOOGL": 1652044, "GOOG": 1652044, "AMZN": 1018724, "NVDA": 1045810,
    "META": 1326801, "TSLA": 1318605, "AVGO": 1730168, "BRK-B": 1067983, "JPM": 19617, "V": 1403161,
    "LLY": 59478, "UNH": 731766, "XOM": 34088, "COST": 909832, "WMT": 104169, "KO": 21344, "PFE": 78003,
    "AMD": 2488, "NFLX": 1065280,
}

# 제목에 회사가 직접 언급됐는지 판단할 이름 (소문자 포함 검사)
NAMES = {
    "AAPL": ("apple",), "MSFT": ("microsoft",), "GOOGL": ("alphabet", "google"), "GOOG": ("alphabet", "google"),
    "AMZN": ("amazon",), "NVDA": ("nvidia",), "META": ("meta", "facebook", "instagram"), "TSLA": ("tesla",),
    "AVGO": ("broadcom",), "BRK-B": ("berkshire", "buffett"), "JPM": ("jpmorgan", "jp morgan", "dimon"),
    "V": ("visa",), "LLY": ("eli lilly", "lilly"), "UNH": ("unitedhealth", "optum"), "XOM": ("exxon",),
    "COST": ("costco",), "WMT": ("walmart",), "KO": ("coca-cola", "coca cola", "coke"), "PFE": ("pfizer",),
    "AMD": ("amd", "advanced micro"), "NFLX": ("netflix",),
}


def mentions(item: dict) -> bool:
    """제목에 티커나 회사 이름이 직접 나오는지 (예: 'Why Intuit Stock Crashed' 는 NVDA 피드에 있어도 False)."""
    sym = item.get("symbol", "")
    title = item.get("title", "")
    low = f" {title.lower()} "
    tick = sym.replace("-", ".")
    if re.search(rf"(?<![A-Za-z]){re.escape(tick)}(?![A-Za-z])", title) and len(tick) > 1:
        return True
    return any(re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", low) for n in NAMES.get(sym, ()))


FORM_LABELS = {"8-K": "수시 공시", "10-Q": "분기 보고서", "10-K": "연간 보고서"}
ITEM_LABELS = {
    "1.01": "중요 계약 체결", "1.02": "중요 계약 해지", "1.03": "파산·회생", "1.05": "사이버 보안 사고",
    "2.01": "자산 인수·처분 완료", "2.02": "실적 발표", "2.03": "채무 발생", "2.04": "채무 조기 상환 사유",
    "2.05": "구조조정 비용", "2.06": "자산 손상", "3.01": "상장 유지 관련 통지", "3.02": "미등록 증권 발행",
    "3.03": "주주 권리 변경", "4.01": "감사인 변경", "4.02": "과거 재무제표 신뢰 불가", "5.01": "지배권 변동",
    "5.02": "임원·이사 변동", "5.03": "정관 변경", "5.07": "주주총회 결과", "5.08": "주주 제안 일정",
    "7.01": "공정 공시(Reg FD)", "8.01": "기타 중요 사항", "9.01": "재무제표·첨부 서류",
}


def _safe_link(url: str | None) -> str | None:
    if not url or len(url) > 1000:
        return None
    u = urlparse(url.strip())
    return url.strip() if u.scheme in ("http", "https") and u.netloc else None


def _clean(text: str | None, limit: int = 300) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    return t[: limit - 1] + "…" if len(t) > limit else t


def parse_rss(content: bytes, symbol: str) -> list[dict]:
    """RSS → [{guid, symbol, title, link, publisher, published}] (최신 순). 외부 엔티티·DTD 가 있으면 거부."""
    if len(content) > MAX_BYTES:
        raise ValueError("RSS 응답이 너무 큽니다")
    head = content[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in content.lower():
        raise ValueError("RSS 에 DTD/엔티티가 있어 읽지 않습니다")
    root = ET.fromstring(content)
    out = []
    for item in root.iter("item"):
        title = _clean(item.findtext("title"))
        link = _safe_link(item.findtext("link"))
        if not title or not link:
            continue
        try:
            pub = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        guid = _clean(item.findtext("guid"), 200) or link
        host = urlparse(link).netloc.lower().removeprefix("www.")
        out.append({"guid": guid, "symbol": symbol, "title": title, "link": link, "publisher": host,
                    "published": pub.isoformat(timespec="seconds")})
        if len(out) >= MAX_ITEMS:
            break
    out.sort(key=lambda r: r["published"], reverse=True)
    seen, uniq = set(), []
    for r in out:                       # 같은 제목 중복 제거
        k = r["title"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def fetch_news(symbol: str, session=None, timeout: float = 10.0) -> list[dict]:
    s = session or requests
    resp = s.get(YAHOO_RSS, params={"s": symbol, "region": "US", "lang": "en-US"},
                 headers={"User-Agent": "Mozilla/5.0 (STk forecast site)"}, timeout=timeout)
    resp.raise_for_status()
    return parse_rss(resp.content, symbol)


def parse_filings(payload: dict, symbol: str, cik: int, since: str) -> list[dict]:
    rec = (payload.get("filings") or {}).get("recent") or {}
    forms = rec.get("form") or []
    out = []
    for i, form in enumerate(forms):
        if form not in FILING_FORMS:
            continue
        date = (rec.get("filingDate") or [""] * len(forms))[i]
        if date < since:
            continue
        acc = (rec.get("accessionNumber") or [""] * len(forms))[i]
        doc = (rec.get("primaryDocument") or [""] * len(forms))[i]
        if not re.fullmatch(r"[0-9-]{10,25}", acc or "") or not re.fullmatch(r"[\w.\-]{1,120}", doc or ""):
            continue
        items = [x.strip() for x in str((rec.get("items") or [""] * len(forms))[i] or "").split(",") if x.strip()]
        labels = [ITEM_LABELS[x] for x in items if x in ITEM_LABELS and x != "9.01"]
        out.append({
            "symbol": symbol, "date": date, "form": form, "items": items,
            "label": FORM_LABELS[form] + (": " + ", ".join(labels) if labels else ""),
            "link": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
        })
    return out[:20]


def fetch_filings(symbol: str, user_agent: str, session=None, timeout: float = 10.0,
                  today: datetime | None = None) -> list[dict]:
    cik = CIK.get(symbol)
    if cik is None:
        return []
    s = session or requests
    resp = s.get(SEC_SUBMISSIONS.format(cik=cik), headers={"User-Agent": user_agent}, timeout=timeout)
    resp.raise_for_status()
    since = ((today or datetime.now(timezone.utc)) - timedelta(days=FILING_DAYS)).strftime("%Y-%m-%d")
    return parse_filings(resp.json(), symbol, cik, since)


def collect(symbols: list[str], sec_user_agent: str | None = None, workers: int = 4) -> dict[str, dict]:
    """종목별 {'news': [...], 'filings': [...], 'errors': [...]}. 예외는 밖으로 내보내지 않음."""

    def one(sym: str) -> tuple[str, dict]:
        res = {"news": [], "filings": [], "errors": []}
        try:
            res["news"] = fetch_news(sym)
        except Exception as err:  # noqa: BLE001 - 참고 표시용, 실패는 개수만 보고
            res["errors"].append(f"news: {type(err).__name__}")
        if sec_user_agent:
            try:
                res["filings"] = fetch_filings(sym, sec_user_agent)
            except Exception as err:  # noqa: BLE001
                res["errors"].append(f"sec: {type(err).__name__}")
        return sym, res

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:   # SEC 권장 초당 10회 이하
        return dict(pool.map(one, symbols))


def default_fetcher(symbols: list[str]) -> dict[str, dict]:
    return collect(symbols, os.environ.get(SEC_ENV) or None)


def recent(news: list[dict], now: datetime | None = None, days: int = NEWS_DAYS) -> list[dict]:
    cut = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat(timespec="seconds")
    return [n for n in news if n.get("published", "") >= cut]


def append_archive(path: str | Path, collected: dict[str, dict], seen_at: datetime | None = None) -> int:
    """처음 본 헤드라인만 data/news.jsonl 에 추가 (종목·guid 기준). 나중에 뉴스 예측력 검증용 원자료."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                keys.add((r.get("symbol"), r.get("guid")))
            except json.JSONDecodeError:
                continue
    stamp = (seen_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(timespec="seconds")
    new = []
    for sym, res in collected.items():
        for n in res.get("news", []):
            k = (sym, n["guid"])
            if k in keys:
                continue
            keys.add(k)
            new.append(dict(n, first_seen=stamp))
    if new:
        with p.open("a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    return len(new)
