"""Indian equity trading costs per broker (NSE cash segment).

Statutory charges (STT, exchange, SEBI, stamp duty, GST) are the same at
every broker; brokerage and DP charges differ. Figures are approximate
(checked late 2025) - confirm on your broker's charges page and edit
BROKERS if they change.
"""

from __future__ import annotations

from dataclasses import dataclass

STT_DELIVERY = 0.001        # 0.1% on buy and sell
STT_INTRADAY_SELL = 0.00025 # 0.025% on sell only
NSE_TXN = 0.0000297         # 0.00297% of turnover
SEBI_FEE = 0.000001         # Rs 10 per crore
STAMP_DELIVERY_BUY = 0.00015
STAMP_INTRADAY_BUY = 0.00003
GST = 0.18


@dataclass(frozen=True)
class Broker:
    name: str
    delivery_pct: float     # brokerage % of order value (0 = free)
    delivery_cap: float     # max brokerage per order, Rs
    delivery_min: float     # min brokerage per order, Rs
    intraday_pct: float
    intraday_cap: float
    dp_charge: float        # per scrip per sell day, incl. GST


BROKERS = {
    "zerodha": Broker("Zerodha", 0.0, 0.0, 0.0, 0.0003, 20.0, 15.93),
    "dhan": Broker("Dhan", 0.0, 0.0, 0.0, 0.0003, 20.0, 14.75),
    "groww": Broker("Groww", 0.001, 20.0, 5.0, 0.001, 20.0, 15.93),
}


def _brokerage(value: float, pct: float, cap: float, floor: float = 0.0) -> float:
    if pct == 0:
        return 0.0
    return max(floor, min(cap, value * pct))


def trade_cost(buy_value: float, sell_value: float, intraday: bool = False,
               broker: str = "zerodha") -> dict:
    """All-in cost of one round trip (buy then sell), in rupees."""
    b = BROKERS.get(broker.lower(), BROKERS["zerodha"])
    if intraday:
        brk = (_brokerage(buy_value, b.intraday_pct, b.intraday_cap)
               + _brokerage(sell_value, b.intraday_pct, b.intraday_cap))
        stt = sell_value * STT_INTRADAY_SELL
        stamp = buy_value * STAMP_INTRADAY_BUY
        dp = 0.0
    else:
        brk = (_brokerage(buy_value, b.delivery_pct, b.delivery_cap, b.delivery_min)
               + _brokerage(sell_value, b.delivery_pct, b.delivery_cap, b.delivery_min))
        stt = (buy_value + sell_value) * STT_DELIVERY
        stamp = buy_value * STAMP_DELIVERY_BUY
        dp = b.dp_charge
    turnover = buy_value + sell_value
    txn = turnover * NSE_TXN
    sebi = turnover * SEBI_FEE
    gst = (brk + txn + sebi) * GST
    total = brk + stt + stamp + txn + sebi + gst + dp
    return {"brokerage": brk, "stt": stt, "stamp": stamp, "exchange": txn, "sebi": sebi,
            "gst": gst, "dp": dp, "total": total,
            "pct_of_turnover": total / turnover * 100 if turnover else 0.0}


def breakeven_move_pct(price: float, qty: int, intraday: bool = False,
                       broker: str = "zerodha") -> float:
    """How far price must move before the trade covers its own costs."""
    v = price * qty
    return trade_cost(v, v, intraday, broker)["total"] / v * 100 if v else 0.0
