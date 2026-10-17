"""Command line:  python -m stockadvisor <command>

  report     today's market regime, swing ideas and long-term picks
  train      walk-forward train the ML ranking model
  backtest   swing + long-term backtests with the current config
  dashboard  launch the visual dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .config import ROOT, load_config


def _ctx(args):
    from .market import load_context
    print(f"Loading {args.universe or load_config()['universe']['index']} …", flush=True)
    return load_context(args.universe, refresh=args.refresh)


def cmd_report(args):
    from .advisor import build_report
    ctx = _ctx(args)
    r = build_report(ctx, use_ai=not args.no_ai, ai_top=args.ai_top)
    m, mv = r["market"], r["market_view"]
    print(f"\n=== {r['asof']} · Market: {m['label']} ({m['score']:.0f}/100) · size x{m['exposure']} ===")
    print(mv["headline"])
    for x in m["reasons"]:
        print("  -", x)
    print("\nSetup track record (last 4y, after costs):")
    for e in r["edge"].to_dict("records"):
        print(f"  {e['setup']:<16} {e['expectancy_r']:+.2f}R  win {e['win_rate_pct']:.0f}%  {e['verdict']}")
    print("\nSwing ideas:")
    for s in r["swing"][:10]:
        v = (s.ai or {}).get("verdict", "-")
        print(f"  [{s.grade}] {s.symbol:<12} {s.setup:<16} score {s.score:5.1f}  entry {s.entry:>10,.2f} "
              f"stop {s.stop:>10,.2f}  T2 {s.target2:>10,.2f}  qty {s.qty:>5}  AI: {v}")
    print("\nLong-term portfolio:")
    for t, p in r["longterm"]["portfolio"].iterrows():
        print(f"  {t.removesuffix('.NS'):<12} {p['weight_pct']:>5.1f}%  qty {p['qty']:>5}  {p['sector']}")
    print("\nOpen the dashboard for charts, news and AI analysis:  python -m stockadvisor dashboard")


def cmd_train(args):
    from . import ml
    r = ml.train(_ctx(args))
    print(r["folds"].round(3).to_string(index=False))
    print(r["meta"])


def cmd_backtest(args):
    from .backtest import SwingParams, run_swing, setup_edge
    from .metrics import equity_stats, trade_stats
    from .strategies.longterm import backtest_monthly
    ctx = _ctx(args)
    cfg = load_config()
    res = run_swing(ctx, SwingParams(start=args.start, capital=cfg["account"]["capital"],
                                     risk_pct=cfg["account"]["risk_per_trade_pct"], broker=cfg["account"]["broker"]))
    print("\nSWING:", {k: round(v, 2) for k, v in equity_stats(res["equity"]).items()})
    print("TRADES:", {k: round(float(v), 2) for k, v in trade_stats(res["trades"]).items()})
    print(setup_edge(res["trades"]).round(2).to_string(index=False))
    lt = backtest_monthly(ctx, start=args.start, capital=cfg["account"]["capital"])
    print("\nLONG-TERM:", {k: round(v, 2) for k, v in equity_stats(lt["equity"]).items()})
    print("NIFTY:    ", {k: round(v, 2) for k, v in equity_stats(lt["benchmark"]).items()})


def cmd_dashboard(_args):
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(prog="stockadvisor")
    ap.add_argument("--universe", choices=["nifty50", "nifty100", "nifty200", "nifty500"])
    ap.add_argument("--refresh", action="store_true", help="force re-download of recent prices")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--no-ai", action="store_true")
    p.add_argument("--ai-top", type=int, default=5)
    p.set_defaults(func=cmd_report)
    sub.add_parser("train").set_defaults(func=cmd_train)
    p = sub.add_parser("backtest")
    p.add_argument("--start", default="2021-01-01")
    p.set_defaults(func=cmd_backtest)
    sub.add_parser("dashboard").set_defaults(func=cmd_dashboard)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
