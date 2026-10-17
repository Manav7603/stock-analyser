"""Shared types for every strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Signal:
    ticker: str
    symbol: str
    name: str
    sector: str
    horizon: str              # "swing" | "longterm" | "intraday"
    setup: str                # e.g. "Breakout", "Trend pullback"
    side: str                 # "BUY" | "SELL"
    date: str
    close: float
    entry: float              # suggested entry / trigger
    entry_type: str           # "buy above" | "at open" | "limit"
    stop: float
    target1: float
    target2: float
    score: float              # 0-100 technical quality
    grade: str = ""
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    ml_prob: float | None = None
    qty: int = 0
    capital_needed: float = 0.0
    risk_amount: float = 0.0
    cost_breakeven_pct: float = 0.0
    news: list[dict] = field(default_factory=list)
    news_sentiment: float | None = None
    ai: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_risk(self) -> float:
        r = self.risk_per_share
        return abs(self.target2 - self.entry) / r if r else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_per_share"] = self.risk_per_share
        d["reward_risk"] = self.reward_risk
        return d


def grade(score: float) -> str:
    return "A" if score >= 75 else "B" if score >= 60 else "C"


def position_size(capital: float, risk_pct: float, entry: float, stop: float,
                  max_position_pct: float, exposure: float = 1.0) -> int:
    """Shares such that a stop-out loses risk_pct of capital, capped by position size."""
    per_share = abs(entry - stop)
    if per_share <= 0 or entry <= 0:
        return 0
    by_risk = capital * risk_pct / 100 * exposure / per_share
    by_cap = capital * max_position_pct / 100 / entry
    return int(max(0, min(by_risk, by_cap)))
