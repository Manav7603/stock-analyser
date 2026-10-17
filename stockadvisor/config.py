"""Loads config.toml into a nested dict with safe defaults."""

from __future__ import annotations

import copy
import os
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
for _d in (DATA_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env without overriding real environment variables."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

DEFAULTS: dict = {
    "account": {
        "capital": 500000,
        "risk_per_trade_pct": 1.0,
        "max_open_positions": 8,
        "max_position_pct": 20.0,
        "broker": "zerodha",
    },
    "universe": {"index": "nifty200", "history_years": 6},
    "swing": {
        "min_avg_traded_value_cr": 20,
        "stop_atr": 2.0,
        "target1_r": 1.5,
        "target2_r": 3.0,
        "max_hold_days": 20,
    },
    "longterm": {"portfolio_size": 15, "max_per_sector": 3, "rebalance": "monthly"},
    "intraday": {"interval": "15m", "risk_per_trade_pct": 0.5},
    "ai": {"model": "claude-opus-5-5", "effort": "medium", "enabled": True},
    "news": {"lookback_days": 10, "max_items": 12},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.toml"
    if not p.exists():
        return copy.deepcopy(DEFAULTS)
    with p.open("rb") as f:
        return _merge(DEFAULTS, tomllib.load(f))
