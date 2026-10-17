"""Company and market news from Google News (India edition), with a
finance-aware sentiment score and event tagging.

News matters because price setups don't know about results, block deals,
regulatory action or downgrades. A great chart with a SEBI probe in the
headlines is not a buy; a results date next week means gap risk.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

from .config import CACHE_DIR

NEWS_DIR = CACHE_DIR / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}

POSITIVE = {
    "surge": 2, "surges": 2, "soar": 2, "soars": 2, "jumps": 2, "jump": 1.5, "rally": 1.5, "rallies": 1.5,
    "record high": 2, "all-time high": 2, "beats": 2, "beat estimates": 2.5, "upgrade": 2.5, "upgraded": 2.5,
    "buy rating": 2, "outperform": 1.5, "order win": 2.5, "bags order": 2.5, "wins order": 2.5, "secures": 1.5,
    "contract": 1, "profit rises": 2, "profit jumps": 2.5, "net profit up": 2, "revenue growth": 1.5,
    "strong": 1, "robust": 1.5, "expansion": 1, "buyback": 2, "dividend": 1, "bonus": 1.5, "approval": 1.5,
    "approves": 1, "acquires": 1, "stake buy": 1.5, "target price raised": 2.5, "raises target": 2.5,
    "gains": 1, "rises": 1, "higher": 0.5, "margin expansion": 2, "debt free": 1.5, "inflow": 1,
    "partnership": 1, "launch": 0.5, "guidance raised": 2.5, "beats street": 2.5, "multibagger": 1,
}
NEGATIVE = {
    "plunge": 2.5, "plunges": 2.5, "crash": 2.5, "crashes": 2.5, "tanks": 2, "slumps": 2, "falls": 1,
    "drops": 1, "declines": 1, "lower": 0.5, "misses": 2, "miss estimates": 2.5, "downgrade": 2.5,
    "downgraded": 2.5, "sell rating": 2, "underperform": 1.5, "loss widens": 2.5, "net loss": 2,
    "profit falls": 2, "profit declines": 2, "weak": 1, "sebi": 1.5, "probe": 2.5, "raid": 2.5, "fraud": 3,
    "penalty": 2, "fine": 1, "ban": 2, "default": 3, "resigns": 2, "resignation": 2, "pledge": 1.5,
    "block deal": 0.5, "stake sale": 1.5, "offloads": 1.5, "cut target": 2.5, "target price cut": 2.5,
    "warning": 1.5, "lawsuit": 2, "show cause": 2, "margin pressure": 2, "guidance cut": 2.5,
    "outflow": 1, "investigation": 2.5, "insolvency": 3, "shutdown": 2, "strike": 1.5, "recall": 1.5,
}
EVENTS = {
    "Results": r"\b(q[1-4]|quarter|results?|earnings|net profit|revenue|ebitda)\b",
    "Rating change": r"\b(upgrades?|downgrades?|target price|rating|brokerage|initiates coverage)\b",
    "Order / deal": r"\b(order|contract|deal|acquisition|acquires|merger|partnership|tie-up)\b",
    "Corporate action": r"\b(dividend|bonus|split|buyback|rights issue|demerger|record date)\b",
    "Regulatory": r"\b(sebi|rbi|probe|penalty|ban|notice|investigation|court|tribunal|nclt)\b",
    "Management": r"\b(ceo|cfo|md|chairman|resigns|appoints?|board)\b",
    "Block / stake": r"\b(block deal|bulk deal|stake|promoter|pledge|offloads?|fii|dii)\b",
}
EVENTS = {k: re.compile(v, re.I) for k, v in EVENTS.items()}
_POS_RX = [(re.compile(r"\b" + re.escape(k) + r"\b", re.I), w) for k, w in POSITIVE.items()]
_NEG_RX = [(re.compile(r"\b" + re.escape(k) + r"\b", re.I), w) for k, w in NEGATIVE.items()]


def _score_text(text: str) -> float:
    pos = sum(w for rx, w in _POS_RX if rx.search(text))
    neg = sum(w for rx, w in _NEG_RX if rx.search(text))
    if pos == neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / (pos + neg + 1)))


def _cache_get(key: str, max_age: float):
    p = NEWS_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json"
    if p.exists() and time.time() - p.stat().st_mtime < max_age:
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _cache_put(key: str, val) -> None:
    (NEWS_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json").write_text(json.dumps(val))


def google_news(query: str, days: int = 10, limit: int = 15, max_age: float = 3600) -> list[dict]:
    key = f"{query}|{days}|{limit}"
    hit = _cache_get(key, max_age)
    if hit is not None:
        return hit
    url = (f"https://news.google.com/rss/search?q={quote_plus(query + f' when:{days}d')}"
           "&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError):
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items, seen = [], set()
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        source = src_el.text.strip() if src_el is not None and src_el.text else ""
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        norm = re.sub(r"\W+", " ", title.lower())[:80]
        if not title or norm in seen:
            continue
        seen.add(norm)
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except (TypeError, ValueError):
            dt = datetime.now(timezone.utc)
        if dt < cutoff:
            continue
        items.append({"title": title, "source": source, "link": it.findtext("link"),
                      "published": dt.isoformat(), "sentiment": _score_text(title),
                      "events": [k for k, rx in EVENTS.items() if rx.search(title)]})
    items.sort(key=lambda x: x["published"], reverse=True)
    items = items[:limit]
    _cache_put(key, items)
    return items


_SUFFIX = re.compile(r"\b(ltd|limited|industries|corporation|company|co|inc|plc)\b\.?", re.I)


def _clean_name(name: str) -> str:
    n = _SUFFIX.sub("", name)
    n = re.sub(r"\(.*?\)", "", n)
    return re.sub(r"\s+", " ", n).strip(" .&-,") or name


def _relevant(title: str, symbol: str, base: str) -> bool:
    t = title.lower()
    if re.search(r"\b" + re.escape(symbol.lower()) + r"\b", t):
        return True
    words = [w for w in re.findall(r"[a-z0-9&]+", base.lower()) if len(w) > 2 and w not in {"india", "the", "and"}]
    if not words:
        return False
    # first distinctive word (e.g. "infosys", "bajaj") plus most of the rest
    hits = sum(1 for w in words if w in t)
    return words[0] in t and hits >= max(1, len(words) - 1)


def stock_news(symbol: str, name: str, days: int = 10, limit: int = 12) -> list[dict]:
    base = _clean_name(name)
    items = google_news(f'"{base}" share OR stock', days, limit * 2)
    if len(items) < 4:
        items += google_news(f'{symbol} share price NSE', days, limit * 2)
    out, seen = [], set()
    for it in items:
        if it["title"] in seen or not _relevant(it["title"], symbol, base):
            continue
        if re.search(r"share price (today|live)|stock price live", it["title"], re.I):
            continue          # quote pages, not news
        seen.add(it["title"])
        out.append(it)
    return out[:limit]


def market_news(days: int = 2, limit: int = 15) -> list[dict]:
    return google_news("Sensex Nifty stock market today", days, limit, max_age=1800)


def summarise(items: list[dict]) -> dict:
    if not items:
        return {"count": 0, "sentiment": 0.0, "label": "No recent news", "events": {}}
    now = datetime.now(timezone.utc)
    w, tot = 0.0, 0.0
    for it in items:
        age = (now - datetime.fromisoformat(it["published"])).total_seconds() / 86400
        wt = 0.5 ** (age / 3)                      # half-life of 3 days
        w += wt
        tot += wt * it["sentiment"]
    s = tot / w if w else 0.0
    ev: dict[str, int] = {}
    for it in items:
        for e in it["events"]:
            ev[e] = ev.get(e, 0) + 1
    label = ("Positive" if s > 0.2 else "Mildly positive" if s > 0.05 else
             "Negative" if s < -0.2 else "Mildly negative" if s < -0.05 else "Neutral")
    return {"count": len(items), "sentiment": round(s, 3), "label": label, "events": ev}
