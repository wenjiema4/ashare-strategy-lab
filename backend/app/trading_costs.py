"""Cash-denominated A-share costs shared by backtests and paper execution.

Prices are raw execution references; slippage is applied exactly once by
cash_value. Explicit tax rates (including zero) override the dated schedule.
Supported history starts 2022-04-29, when the current stock transfer rate
took effect. Earlier schedules and adjusted share accounting are unsupported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

DEFAULT_COMMISSION_PCT = 0.00025
CURRENT_STAMP_TAX_PCT = 0.0005
MIN_COMMISSION = 5.0
TRANSFER_FEE_PCT = 0.00001
STAMP_HALVED_FROM = date(2023, 8, 28)
SUPPORTED_FROM = date(2022, 4, 29)


def _money(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class FeeCharge:
    commission: float
    stamp_tax: float
    transfer_fee: float

    @property
    def total(self) -> float:
        return _money(self.commission + self.stamp_tax + self.transfer_fee)


@dataclass(frozen=True)
class FeePolicy:
    commission_pct: float = DEFAULT_COMMISSION_PCT
    stamp_tax_pct: float | None = None
    min_commission: float = MIN_COMMISSION
    transfer_fee_pct: float = TRANSFER_FEE_PCT
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_pct", "stamp_tax_pct", "min_commission", "transfer_fee_pct", "slippage_bps"):
            value = getattr(self, name)
            if name == "stamp_tax_pct" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if self.slippage_bps >= 10000:
            raise ValueError("slippage_bps must be less than 10000")

    def execution_price(self, price: float, side: str) -> float:
        if not math.isfinite(price) or price <= 0 or side not in ("buy", "sell"):
            raise ValueError("Invalid price or side")
        return round(max(price * (1 + (1 if side == "buy" else -1) * self.slippage_bps / 10000), 0.01), 4)

    def charge(self, qty: float, price: float, side: str, trade_date: date, asset_type: str) -> FeeCharge:
        if isinstance(qty, bool) or not math.isfinite(qty) or qty <= 0:
            raise ValueError("Invalid quantity")
        if not math.isfinite(price) or price <= 0 or side not in ("buy", "sell"):
            raise ValueError("Invalid price or side")
        if asset_type not in ("stock", "etf") or not isinstance(trade_date, date):
            raise ValueError("Explicit trade date and stock/etf asset type required")
        if trade_date < SUPPORTED_FROM:
            raise ValueError("金额费用模型仅支持 2022-04-29 及以后的成交")
        turnover = qty * price
        if not math.isfinite(turnover):
            raise ValueError("Invalid turnover")
        tax = self.stamp_tax_pct
        if tax is None:
            tax = CURRENT_STAMP_TAX_PCT if trade_date >= STAMP_HALVED_FROM else 0.001
        return FeeCharge(
            commission=_money(max(turnover * self.commission_pct, self.min_commission)),
            stamp_tax=_money(turnover * tax) if side == "sell" and asset_type == "stock" else 0.0,
            transfer_fee=_money(turnover * self.transfer_fee_pct) if asset_type == "stock" else 0.0,
        )

    def cash_value(self, qty: float, price: float, side: str, trade_date: date, asset_type: str) -> float:
        executed = self.execution_price(price, side)
        fees = self.charge(qty, executed, side, trade_date, asset_type)
        return _money(qty * executed + (fees.total if side == "buy" else -fees.total))

    def affordable_shares(self, budget: float, price: float, trade_date: date, asset_type: str) -> int:
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("Invalid budget")
        executed = self.execution_price(price, "buy")
        lo, hi = 0, int(budget / executed / 100)
        # Binary search also handles cent rounding and a minimum fee larger
        # than several lots, without iterating over every potential order lot.
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cash_value(mid * 100, price, "buy", trade_date, asset_type) <= budget + 1e-8:
                lo = mid
            else:
                hi = mid - 1
        return lo * 100
