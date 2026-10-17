"""Claude-powered analyst layer.

Claude reads what the quantitative engine produced (setup, levels,
indicators, ML rank, historical edge of the setup, market regime) plus the
recent headlines, and writes a second opinion in plain English: does the
news support or contradict the chart, what are the catalysts and risks,
and what should you watch before entering.

Every response is cached on disk for 12 hours, so re-opening the dashboard
does not spend API credits. Without an API key the module falls back to a
rule-based analyst with the same output shape.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time

from .config import CACHE_DIR, load_config

AI_DIR = CACHE_DIR / "ai"
AI_DIR.mkdir(parents=True, exist_ok=True)
CACHE_SECONDS = 12 * 3600

SYSTEM = """You are a senior Indian equity research analyst advising a retail trader \
who places their own orders on Zerodha, Groww or Dhan. You are given output from a \
quantitative screening engine and recent news headlines.

Rules:
- Ground every statement in the data provided. Never invent prices, numbers, \
results dates or news that are not in the input. If news is missing or thin, say so.
- Judge whether the news supports, contradicts or is irrelevant to the technical setup. \
Upcoming results, regulatory action, promoter selling, downgrades or pledges are red \
flags for a fresh swing entry; order wins, upgrades and strong results are tailwinds.
- Weigh the market regime and the setup's historical edge heavily. A setup with no \
proven edge in a bearish market deserves "Wait" or "Avoid" even if the chart looks good.
- Be candid and specific. Prefer "wait for a close above 1,250" over vague advice.
- Use simple English that a newer trader understands. Use Indian conventions \
(Rs, lakh, crore, NSE).
- This is decision support, not a guarantee. Never promise profit."""

STOCK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["Strong setup", "Good setup", "Wait", "Avoid"]},
        "confidence": {"type": "integer"},
        "news_sentiment": {"type": "number"},
        "news_view": {"type": "string", "enum": ["Supports the setup", "Contradicts the setup",
                                                "Mixed", "No meaningful news"]},
        "news_summary": {"type": "string"},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "what_to_watch": {"type": "array", "items": {"type": "string"}},
        "plain_english": {"type": "string"},
        "tips": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "confidence", "news_sentiment", "news_view", "news_summary",
                 "catalysts", "risks", "what_to_watch", "plain_english", "tips"],
    "additionalProperties": False,
}

MARKET_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "stance": {"type": "string", "enum": ["Aggressive", "Normal", "Defensive", "Stay in cash"]},
        "summary": {"type": "string"},
        "news_takeaways": {"type": "array", "items": {"type": "string"}},
        "sectors_to_focus": {"type": "array", "items": {"type": "string"}},
        "sectors_to_avoid": {"type": "array", "items": {"type": "string"}},
        "tips": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "stance", "summary", "news_takeaways", "sectors_to_focus",
                 "sectors_to_avoid", "tips"],
    "additionalProperties": False,
}


def _cache_key(kind: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(f"{kind}|{raw}".encode()).hexdigest()[:32]


def _cached(key: str):
    p = AI_DIR / f"{key}.json"
    if p.exists() and time.time() - p.stat().st_mtime < CACHE_SECONDS:
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _store(key: str, val: dict) -> None:
    (AI_DIR / f"{key}.json").write_text(json.dumps(val, indent=2))


def available() -> bool:
    cfg = load_config()["ai"]
    if not cfg.get("enabled", True) or not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    return importlib.util.find_spec("anthropic") is not None


class AIError(RuntimeError):
    pass


def _call(schema: dict, prompt: str, max_tokens: int = 6000) -> dict:
    import anthropic

    cfg = load_config()["ai"]
    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=cfg.get("model", "claude-opus-5-5"),
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            output_config={"effort": cfg.get("effort", "medium"),
                           "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as e:
        raise AIError("Claude API key is invalid or revoked") from e
    except anthropic.RateLimitError as e:
        raise AIError("Claude API rate limit hit - try again in a minute") from e
    except anthropic.APIStatusError as e:
        raise AIError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AIError("Could not reach the Claude API (network)") from e
    if resp.stop_reason == "refusal":
        raise AIError("Claude declined this request")
    if resp.stop_reason == "max_tokens":
        raise AIError("Claude response was cut off")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        raise AIError("Claude returned malformed JSON") from e
    out["_source"] = f"Claude ({resp.model})"
    return out


# ----------------------------------------------------------------- stock view

def _stock_prompt(ctx: dict) -> str:
    return ("Analyse this trade idea for an Indian retail trader.\n\n"
            "<engine_output>\n" + json.dumps(ctx, indent=2, default=str) + "\n</engine_output>\n\n"
            "Give your verdict, a confidence 0-100, a news sentiment from -1 (very negative) to "
            "+1 (very positive), whether the news supports the setup, 2-4 catalysts, 2-4 risks, "
            "2-4 concrete things to watch before or after entry, a 3-5 sentence plain-English "
            "explanation, and 2-4 practical tips for executing this specific trade.")


def rule_based_stock_view(ctx: dict) -> dict:
    score = ctx.get("technical_score", 50)
    ml_rank = ctx.get("ml_rank")
    edge = ctx.get("setup_track_record", {}) or {}
    news = ctx.get("news_summary", {}) or {}
    regime = ctx.get("market", {}).get("label", "Neutral")
    days_to_results = ctx.get("days_to_results")
    conf = score
    if ml_rank is not None:
        conf = 0.6 * conf + 0.4 * ml_rank
    if edge.get("verdict") == "Proven edge":
        conf += 8
    elif edge.get("verdict") == "No edge - avoid":
        conf -= 15
    s = news.get("sentiment", 0) or 0
    conf += s * 15
    risks = list(ctx.get("risks", []))[:3]
    if days_to_results is not None and 0 <= days_to_results <= 7:
        conf -= 12
        risks.insert(0, f"Results due in {days_to_results} days - price can gap either way")
    if regime == "Bearish":
        conf -= 10
    conf = int(max(0, min(100, conf)))
    verdict = "Strong setup" if conf >= 75 else "Good setup" if conf >= 60 else "Wait" if conf >= 45 else "Avoid"
    view = ("No meaningful news" if news.get("count", 0) == 0 else
            "Supports the setup" if s > 0.1 else "Contradicts the setup" if s < -0.1 else "Mixed")
    tips = ["Place the stop-loss order in your broker app immediately after entry (GTT on Zerodha, "
            "Forever order on Dhan) - don't keep it in your head.",
            "Book half at Target 1 and move the stop to your entry price - the rest of the trade is then free.",
            "If the stock gaps up more than 1 ATR above the entry, skip it rather than chase."]
    if days_to_results is not None and 0 <= days_to_results <= 7:
        tips.insert(0, "Results are close: either wait for the numbers or take half the normal size.")
    return {
        "verdict": verdict, "confidence": conf, "news_sentiment": round(s, 2), "news_view": view,
        "news_summary": (f"{news.get('count', 0)} headlines in the last {ctx.get('news_days', 10)} days; "
                         f"tone {news.get('label', 'Neutral').lower()}."
                         + (f" Events: {', '.join(news.get('events', {}))}." if news.get("events") else "")),
        "catalysts": list(ctx.get("reasons", []))[:3],
        "risks": risks or ["No specific risk flagged - normal market risk applies"],
        "what_to_watch": [f"Entry trigger at Rs {ctx.get('entry', 0):,.2f}",
                          f"Stop-loss at Rs {ctx.get('stop', 0):,.2f} - exit without second thoughts",
                          "Volume on the entry day should be above average"],
        "plain_english": (f"The engine rates this {ctx.get('setup', 'setup')} at {score:.0f}/100 technically"
                          + (f", the ML model ranks it in the top {100 - ml_rank:.0f}% of stocks" if ml_rank is not None and ml_rank >= 50 else "")
                          + f". The market regime is {regime.lower()}. "
                          + ("Historically this setup has made money after costs. " if edge.get("verdict") == "Proven edge" else
                             "Historically this setup has NOT made money after costs, so be selective. " if edge.get("verdict") == "No edge - avoid" else "")
                          + "Size the position so that hitting the stop costs no more than your planned risk."),
        "tips": tips,
        "_source": "Rule-based (add ANTHROPIC_API_KEY for Claude analysis)",
    }


def stock_view(ctx: dict, use_ai: bool = True) -> dict:
    key = _cache_key("stock", ctx)
    hit = _cached(key)
    if hit:
        return hit
    if use_ai and available():
        try:
            out = _call(STOCK_SCHEMA, _stock_prompt(ctx))
            out["confidence"] = int(max(0, min(100, out.get("confidence", 50))))
            out["news_sentiment"] = float(max(-1, min(1, out.get("news_sentiment", 0))))
            _store(key, out)
            return out
        except AIError as e:
            fb = rule_based_stock_view(ctx)
            fb["_source"] = f"Rule-based ({e})"
            return fb
    return rule_based_stock_view(ctx)


# ---------------------------------------------------------------- market view

def rule_based_market_view(m: dict) -> dict:
    label = m.get("label", "Neutral")
    stance = {"Bullish": "Normal", "Neutral-positive": "Normal", "Cautious": "Defensive"}.get(label, "Stay in cash")
    if label == "Bullish" and m.get("breadth_above50", 0) > 70:
        stance = "Aggressive"
    sectors = m.get("sectors", [])
    return {
        "headline": f"Market regime: {label} (score {m.get('score', 0):.0f}/100)",
        "stance": stance,
        "summary": m.get("note", ""),
        "news_takeaways": [n["title"] for n in m.get("market_news", [])[:4]],
        "sectors_to_focus": [s["sector"] for s in sectors[:3]],
        "sectors_to_avoid": [s["sector"] for s in sectors[-3:]],
        "tips": ["Trade with the trend of the market - most stocks follow NIFTY.",
                 "Cut size when the regime weakens; capital preserved in a bad phase compounds later.",
                 "Prefer stocks from the strongest sectors; leadership tends to persist for weeks."],
        "_source": "Rule-based (add ANTHROPIC_API_KEY for Claude analysis)",
    }


def market_view(m: dict, use_ai: bool = True) -> dict:
    key = _cache_key("market", m)
    hit = _cached(key)
    if hit:
        return hit
    if use_ai and available():
        prompt = ("Write today's market brief for an Indian retail trader from this data.\n\n"
                  "<market_data>\n" + json.dumps(m, indent=2, default=str) + "\n</market_data>\n\n"
                  "Give a one-line headline, a stance, a 4-6 sentence summary that connects the "
                  "news to the regime and breadth, 3-5 news takeaways, the sectors to focus on and "
                  "avoid (from the data), and 3-4 tips for today.")
        try:
            out = _call(MARKET_SCHEMA, prompt)
            _store(key, out)
            return out
        except AIError as e:
            fb = rule_based_market_view(m)
            fb["_source"] = f"Rule-based ({e})"
            return fb
    return rule_based_market_view(m)


# ------------------------------------------------------------------ free chat

def ask(question: str, context: dict, history: list[dict] | None = None) -> str:
    """Free-form Q&A about a stock or the market, grounded in engine data."""
    if not available():
        return "Add ANTHROPIC_API_KEY to the .env file to chat with the AI analyst."
    import anthropic

    cfg = load_config()["ai"]
    msgs = list(history or [])
    msgs.append({"role": "user", "content": "<engine_data>\n" + json.dumps(context, default=str)
                 + "\n</engine_data>\n\n" + question})
    try:
        resp = anthropic.Anthropic().beta.messages.create(
            model=cfg.get("model", "claude-opus-5-5"), max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            system=SYSTEM, output_config={"effort": "low"}, messages=msgs)
    except anthropic.APIError as e:
        return f"AI analyst unavailable: {e.__class__.__name__}"
    if resp.stop_reason == "refusal":
        return "The AI analyst declined to answer that."
    return next((b.text for b in resp.content if b.type == "text"), "")
