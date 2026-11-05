from datetime import date

import pytest

from app.trading_costs import FeePolicy


@pytest.mark.parametrize(("day", "stamp", "total"), [
    (date(2023, 8, 25), 100.0, 126.0),
    (date(2023, 8, 28), 50.0, 76.0),
])
def test_sell_fee_uses_execution_date_tax_schedule(day, stamp, total):
    charge = FeePolicy().charge(10_000, 10.0, "sell", day, "stock")
    assert charge.commission == 25.0
    assert charge.stamp_tax == stamp
    assert charge.transfer_fee == 1.0
    assert charge.total == total


def test_minimum_commission_and_fund_exemptions():
    policy = FeePolicy()
    buy = policy.charge(100, 10, "buy", date(2024, 1, 2), "stock")
    assert buy.total == 5.01
    assert buy.stamp_tax == 0
    etf = policy.charge(10_000, 10, "sell", date(2024, 1, 2), "etf")
    assert etf.total == 25
    assert etf.stamp_tax == 0
    assert etf.transfer_fee == 0


def test_explicit_zero_stamp_overrides_schedule():
    charge = FeePolicy(stamp_tax_pct=0).charge(1000, 10, "sell", date(2024, 1, 2), "stock")
    assert charge.total == 5.1


def test_cash_values_charge_fees_on_adverse_execution_price():
    policy = FeePolicy(slippage_bps=5)
    assert policy.cash_value(100, 10, "buy", date(2024, 1, 2), "stock") == 1005.51
    assert policy.cash_value(100, 10, "sell", date(2024, 1, 2), "stock") == 993.99


@pytest.mark.parametrize(("budget", "shares"), [(1000, 0), (1005, 0), (1005.01, 100), (2005.02, 200)])
def test_affordable_shares_reserves_minimum_fee(budget, shares):
    assert FeePolicy().affordable_shares(budget, 10, date(2024, 1, 2), "stock") == shares


@pytest.mark.parametrize("kwargs", [
    {"commission_pct": -1}, {"stamp_tax_pct": float("nan")},
    {"slippage_bps": float("inf")}, {"transfer_fee_pct": -1},
    {"min_commission": -1}, {"slippage_bps": 10000},
])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FeePolicy(**kwargs)


@pytest.mark.parametrize(("qty", "price", "side", "asset"), [
    (-100, 10, "buy", "stock"), (100, 0, "buy", "stock"),
    (100, float("nan"), "buy", "stock"), (100, 10, "hold", "stock"),
    (100, 10, "buy", "unknown"), (True, 10, "buy", "stock"),
])
def test_invalid_trade_fails_closed(qty, price, side, asset):
    with pytest.raises(ValueError):
        FeePolicy().cash_value(qty, price, side, date(2024, 1, 2), asset)


def test_fee_schedule_rejects_unsupported_history():
    with pytest.raises(ValueError, match="2022-04-29"):
        FeePolicy().charge(100, 10, "sell", date(2022, 4, 28), "stock")
