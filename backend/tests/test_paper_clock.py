"""Explicit replay clocks preserve Beijing timestamps and T+1 boundaries."""
from datetime import UTC, datetime

from app.strategy import paper

SYMBOL = "600519.SH"


def test_explicit_clock_controls_order_fill_and_t1_capacity(tmp_path):
    paper.create_account(tmp_path, 100_000, slippage_bps=0)
    created = datetime(2024, 6, 3, 9, 31)
    order, error = paper.create_order(
        tmp_path, SYMBOL, "buy", qty=100, ref_price=10, as_of=created,
    )
    assert error is None
    assert order["created_at"] == "2024-06-03T09:31:00+08:00"
    fill = paper._fill_order(
        tmp_path, order, 10, "2024-06-03", as_of=datetime(2024, 6, 3, 9, 32),
    )
    assert fill["ts"] == "2024-06-03T09:32:00+08:00"
    assert fill["seq"] == 1717378320000
    assert paper.get_order(tmp_path, order["id"])["filled_at"] == fill["ts"]
    assert paper.sell_capacity(tmp_path, SYMBOL, as_of=created) == (0, 0)
    sell, error = paper.create_order(tmp_path, SYMBOL, "sell", qty=100, as_of=created)
    assert sell is None and "T+1" in error
    tomorrow = datetime(2024, 6, 4, 9, 31)
    assert paper.sell_capacity(tmp_path, SYMBOL, as_of=tomorrow) == (100, 0)
    sell, error = paper.create_order(tmp_path, SYMBOL, "sell", qty=100, as_of=tomorrow)
    assert error is None
    assert paper.sell_capacity(tmp_path, SYMBOL, as_of=tomorrow) == (100, 100)


def test_explicit_aware_clock_normalizes_to_beijing(tmp_path):
    paper.create_account(tmp_path, 100_000, slippage_bps=0)
    order, error = paper.create_order(
        tmp_path, SYMBOL, "buy", qty=100, ref_price=10,
        as_of=datetime(2024, 6, 3, 1, 31, tzinfo=UTC),
    )
    assert error is None
    assert order["created_at"] == "2024-06-03T09:31:00+08:00"


def test_explicit_clock_controls_limit_queue_timestamp(tmp_path, monkeypatch):
    paper.create_account(tmp_path, 100_000, slippage_bps=0, queue_limit_orders=True)
    clock = datetime(2024, 6, 3, 9, 31)
    order, error = paper.create_order(tmp_path, SYMBOL, "buy", qty=100, as_of=clock)
    assert error is None
    monkeypatch.setattr(paper, "_prev_close", lambda *_: 10)
    assert paper._fill_order(tmp_path, order, 11, "2024-06-03", as_of=clock) is None
    assert paper.get_order(tmp_path, order["id"])["queued_at"] == "2024-06-03T09:31:00+08:00"


def test_strategy_signal_orders_are_not_filled_by_generic_matchers(tmp_path, monkeypatch):
    paper.create_account(tmp_path, 100_000, slippage_bps=0)
    order, error = paper.create_order(
        tmp_path, SYMBOL, "buy", qty=100, order_type="t_signal",
        as_of=datetime(2024, 6, 3, 9, 31),
    )
    assert error is None
    monkeypatch.setattr(paper, "cn_today", lambda: datetime(2024, 6, 3).date())
    monkeypatch.setattr(paper, "read_daily_bar", lambda *_: {"open": 10, "close": 10})
    assert paper.evaluate_intraday(tmp_path, {SYMBOL: 10}) == []
    assert paper.settle_day(tmp_path, "2024-06-03")["filled"] == 0
    assert paper.get_order(tmp_path, order["id"])["status"] == "pending"


def test_sell_capacity_accepts_explicit_date(tmp_path):
    paper.create_account(tmp_path, 100_000, slippage_bps=0)
    clock = datetime(2024, 6, 3, 9, 31)
    order, _ = paper.create_order(tmp_path, SYMBOL, "buy", qty=100, as_of=clock)
    paper._fill_order(tmp_path, order, 10, "2024-06-03", as_of=clock)
    assert paper.sell_capacity(tmp_path, SYMBOL, as_of=datetime(2024, 6, 4).date()) == (100, 0)


def test_limit_rejected_strategy_order_cannot_escape_to_generic_queue(tmp_path, monkeypatch):
    paper.create_account(tmp_path, 100_000, slippage_bps=0, queue_limit_orders=True)
    clock = datetime(2024, 6, 3, 9, 31)
    order, error = paper.create_order(
        tmp_path, SYMBOL, "buy", qty=100, order_type="t_signal", as_of=clock,
    )
    assert error is None
    monkeypatch.setattr(paper, "_prev_close", lambda *_: 10)
    assert paper._fill_order(tmp_path, order, 11, "2024-06-03", as_of=clock) is None
    reloaded = paper.get_order(tmp_path, order["id"])
    assert reloaded["order_type"] == "t_signal"
    assert reloaded["status"] == "expired"
    assert "涨停" in reloaded["reason"]
    monkeypatch.setattr(paper, "cn_today", lambda: datetime(2024, 6, 4).date())
    monkeypatch.setattr(paper, "read_daily_bar", lambda *_: {"open": 10, "close": 10})
    assert paper.evaluate_intraday(tmp_path, {SYMBOL: 10}) == []
    assert paper.settle_day(tmp_path, "2024-06-04")["filled"] == 0
    assert paper.get_order(tmp_path, order["id"])["status"] == "expired"
    assert paper.load_fills(tmp_path) == []


def test_default_clock_keeps_zero_argument_timestamp_override(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "_now_iso", lambda: "2024-06-03T09:31:00+08:00")
    paper.create_account(tmp_path, 100_000, slippage_bps=0, queue_limit_orders=True)
    order, error = paper.create_order(tmp_path, SYMBOL, "buy", qty=100)
    assert error is None
    assert order["created_at"] == "2024-06-03T09:31:00+08:00"
    fill = paper._fill_order(tmp_path, order, 10, "2024-06-03")
    assert fill["ts"] == "2024-06-03T09:31:00+08:00"
    queued, error = paper.create_order(tmp_path, SYMBOL, "buy", qty=100)
    assert error is None
    monkeypatch.setattr(paper, "_prev_close", lambda *_: 10)
    assert paper._fill_order(tmp_path, queued, 11, "2024-06-03") is None
    assert paper.get_order(tmp_path, queued["id"])["queued_at"] == "2024-06-03T09:31:00+08:00"
