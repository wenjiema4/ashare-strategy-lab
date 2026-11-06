"""Gross, per-share corporate entitlements with no invented release dates."""
from datetime import date, datetime

import polars as pl
import pytest

from app.strategy import paper

SYM = "600519.SH"
EX = date(2024, 6, 4)


def clock(day):
    return datetime.combine(day, datetime.min.time()).replace(hour=10)


def seed(root, qty=1000):
    paper.create_account(root, 100_000, slippage_bps=0)
    order, error = paper.create_order(root, SYM, "buy", qty=qty, as_of=clock(date(2024, 6, 3)))
    assert error is None
    assert paper._fill_order(root, order, 10, "2024-06-03", as_of=clock(date(2024, 6, 3)))
    return paper.get_account(root)["cash"]


def events(root, rows):
    folder = root / "adj_factor"
    folder.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(folder / "all.parquet")


def event(**extra):
    return {"symbol": SYM, "trade_date": EX, "ex_factor": 10 / 9,
            "dividend": 1.0, "bonus": 0.0, "allot": 0.0, **extra}


def test_cash_dividend_is_receivable_not_shares_and_replays_once(tmp_path):
    cash = seed(tmp_path)
    events(tmp_path, [event()])
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat(), as_of=clock(EX)) == 1
    assert paper.get_account(tmp_path)["cash"] == cash
    assert paper.load_positions(tmp_path)[SYM]["qty"] == 1000
    assert paper.receivable_balance(tmp_path) == 1000
    nav = paper.daily_nav(tmp_path, EX.isoformat(), {SYM: 9})
    assert nav == {"date": EX.isoformat(), "cash": cash, "mv": 9000, "receivable": 1000, "nav": cash + 10000}
    overview = paper.overview(tmp_path, {SYM: 9})
    assert overview["receivable"] == 1000
    assert overview["total"] == cash + 10000
    rows = paper.load_fills(tmp_path)
    assert rows[-1]["schema_version"] == 2
    assert rows[-1]["share_multiplier"] == 1
    assert rows[-1]["cash_delta"] == 0
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat(), as_of=clock(EX)) == 0
    assert paper.load_fills(tmp_path) == rows
    paper.rebuild_positions(tmp_path)
    assert paper.get_account(tmp_path)["cash"] == cash
    assert paper.receivable_balance(tmp_path) == 1000


def test_known_paydate_transfers_receivable_once_and_unknown_date_can_be_completed(tmp_path):
    cash = seed(tmp_path)
    events(tmp_path, [event()])
    paper.apply_corporate_actions(tmp_path, EX.isoformat(), as_of=clock(EX))
    pay = date(2024, 6, 7)
    events(tmp_path, [event(pay_date=pay)])
    # First observation of a future payment date is durable non-cash evidence.
    assert paper.apply_corporate_actions(tmp_path, "2024-06-06") == 1
    assert paper.receivable_balance(tmp_path) == 1000
    assert paper.apply_corporate_actions(tmp_path, pay.isoformat()) == 1
    assert paper.get_account(tmp_path)["cash"] == cash + 1000
    assert paper.receivable_balance(tmp_path) == 0
    assert paper.apply_corporate_actions(tmp_path, "2024-06-10") == 0
    paper.rebuild_positions(tmp_path)
    assert paper.get_account(tmp_path)["cash"] == cash + 1000


def test_bonus_keeps_old_shares_sellable_and_extra_locked_until_listing(tmp_path):
    cash = seed(tmp_path)
    events(tmp_path, [event(dividend=0.0, bonus=0.1, ex_factor=1.1)])
    paper.apply_corporate_actions(tmp_path, EX.isoformat(), as_of=clock(EX))
    assert paper.load_positions(tmp_path)[SYM]["qty"] == 1100
    assert paper.sell_capacity(tmp_path, SYM, as_of=EX) == (1000, 0)
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 10)) == (1000, 0)
    nav = paper.daily_nav(tmp_path, EX.isoformat(), {SYM: 100 / 11})
    assert nav["nav"] == round(cash + 10000, 2)
    events(tmp_path, [event(dividend=0.0, bonus=0.1, ex_factor=1.1, listing_date=date(2024, 6, 7))])
    paper.apply_corporate_actions(tmp_path, "2024-06-07")
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 6)) == (1000, 0)
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 7)) == (1100, 0)
    paper.rebuild_positions(tmp_path)
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 7)) == (1100, 0)


def test_exday_buy_is_not_entitled_and_corporate_event_precedes_fill(tmp_path):
    seed(tmp_path)
    events(tmp_path, [event()])
    order, error = paper.create_order(tmp_path, SYM, "buy", qty=100, as_of=clock(EX))
    assert error is None
    assert paper.receivable_balance(tmp_path) == 1000
    assert paper._fill_order(tmp_path, order, 9, EX.isoformat(), as_of=clock(EX))
    assert paper.receivable_balance(tmp_path) == 1000
    assert paper.load_positions(tmp_path)[SYM]["qty"] == 1100
    assert paper.sell_capacity(tmp_path, SYM, as_of=EX) == (1000, 0)
    assert [f["kind"] for f in paper.load_fills(tmp_path)] == ["fill", "corp_action", "fill"]


@pytest.mark.parametrize("bad", [
    {"allot": 0.2}, {"dividend": None}, {"bonus": float("nan")},
    {"bonus": 0.0015}, {"pay_date": "invalid"},
])
def test_unsupported_event_fails_before_any_ledger_or_cash_change(tmp_path, bad):
    cash = seed(tmp_path)
    before = paper.load_fills(tmp_path)
    events(tmp_path, [event(**bad)])
    with pytest.raises(ValueError):
        paper.apply_corporate_actions(tmp_path, EX.isoformat(), as_of=clock(EX))
    assert paper.load_fills(tmp_path) == before
    assert paper.get_account(tmp_path)["cash"] == cash


def test_missing_per_share_metadata_rejects_new_ashare_records(tmp_path):
    seed(tmp_path)
    before = paper.load_fills(tmp_path)
    events(tmp_path, [{"symbol": SYM, "trade_date": EX, "ex_factor": 1.1}])
    with pytest.raises(ValueError, match=r"明细|metadata"):
        paper.apply_corporate_actions(tmp_path, EX.isoformat())
    assert paper.load_fills(tmp_path) == before


def test_skipped_days_and_all_events_are_validated_before_partial_writes(tmp_path):
    seed(tmp_path)
    before = paper.load_fills(tmp_path)
    events(tmp_path, [event(), event(trade_date=date(2024, 6, 5), allot=0.1)])
    with pytest.raises(ValueError):
        paper.apply_corporate_actions(tmp_path, "2024-06-10")
    assert paper.load_fills(tmp_path) == before
    events(tmp_path, [event()])
    assert paper.apply_corporate_actions(tmp_path, "2024-06-10") == 1
    assert paper.receivable_balance(tmp_path) == 1000


def test_ex_reference_uses_event_details_while_prev_close_stays_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "_prev_close", lambda *_: 10)
    events(tmp_path, [event()])
    assert paper.ex_reference_close(tmp_path, SYM, "stock", EX.isoformat()) == 9
    assert paper._prev_close(tmp_path, SYM, "stock", EX.isoformat()) == 10
    assert paper.ex_reference_close(tmp_path, SYM, "stock", "2024-06-05") == 10


def test_initial_known_payment_and_listing_dates_gate_once(tmp_path):
    cash = seed(tmp_path)
    events(tmp_path, [event(bonus=0.1, pay_date=date(2024, 6, 7), listing_date=date(2024, 6, 6))])
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat()) == 1
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 5)) == (1000, 0)
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 6)) == (1100, 0)
    # A known payment schedule is retained in the immutable event even if source disappears.
    (tmp_path / "adj_factor/all.parquet").unlink()
    assert paper.apply_corporate_actions(tmp_path, "2024-06-07") == 1
    assert paper.receivable_balance(tmp_path) == 0
    assert paper.get_account(tmp_path)["cash"] == cash + 1000
    assert paper.apply_corporate_actions(tmp_path, "2024-06-10") == 0


def test_locked_bonus_remains_after_original_shares_sell_and_restart(tmp_path):
    seed(tmp_path)
    events(tmp_path, [event(dividend=0.0, bonus=0.1, ex_factor=1.1)])
    order, error = paper.create_order(tmp_path, SYM, "sell", qty=1000, as_of=clock(EX))
    assert error is None
    assert paper._fill_order(tmp_path, order, 100 / 11, EX.isoformat(), as_of=clock(EX))
    assert paper.load_positions(tmp_path)[SYM]["qty"] == 100
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 10)) == (0, 0)
    paper.rebuild_positions(tmp_path)
    assert paper.load_positions(tmp_path)[SYM]["qty"] == 100
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 10)) == (0, 0)


def test_exday_first_buy_has_no_entitlement(tmp_path):
    paper.create_account(tmp_path, 100_000, slippage_bps=0)
    events(tmp_path, [event()])
    order, error = paper.create_order(tmp_path, SYM, "buy", qty=100, as_of=clock(EX))
    assert error is None
    paper._fill_order(tmp_path, order, 9, EX.isoformat(), as_of=clock(EX))
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat()) == 0
    assert paper.receivable_balance(tmp_path) == 0
    assert len(paper.load_fills(tmp_path)) == 1
    assert paper.sell_capacity(tmp_path, SYM, as_of=EX) == (0, 0)


def test_unsupported_later_event_blocks_order_before_any_action_or_order_write(tmp_path):
    seed(tmp_path)
    before = paper.load_fills(tmp_path)
    orders = paper.load_orders(tmp_path)
    events(tmp_path, [event(), event(trade_date=date(2024, 6, 5), allot=0.1)])
    with pytest.raises(ValueError):
        paper.create_order(tmp_path, SYM, "sell", qty=100, as_of=clock(date(2024, 6, 6)))
    assert paper.load_fills(tmp_path) == before
    assert paper.load_orders(tmp_path) == orders


def test_restart_repairs_account_cash_after_payment_append(tmp_path):
    cash = seed(tmp_path)
    events(tmp_path, [event(pay_date=EX)])
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat()) == 2
    acc = paper.get_account(tmp_path)
    acc["cash"] = cash  # model interruption after ledger publish but before account update
    paper.save_account(tmp_path, acc)
    rows = paper.load_fills(tmp_path)
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat()) == 0
    assert paper.get_account(tmp_path)["cash"] == cash + 1000
    assert paper.load_fills(tmp_path) == rows


def test_frozen_and_legacy_records_are_never_reinterpreted(tmp_path):
    seed(tmp_path)
    paper._append_fill(tmp_path, {"kind": "corp_action", "date": EX.isoformat(), "symbol": SYM, "factor": 1.1})
    paper.rebuild_positions(tmp_path)
    events(tmp_path, [event()])
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat()) == 0
    assert paper.load_positions(tmp_path)[SYM]["qty"] == 1100
    assert paper.receivable_balance(tmp_path) == 0
    paper.update_settings(tmp_path, status="frozen")
    before = paper.load_fills(tmp_path)
    assert paper.apply_corporate_actions(tmp_path, "2024-06-10") == 0
    assert paper.load_fills(tmp_path) == before


def test_exday_first_buy_rejects_unsupported_details_before_saving_order(tmp_path):
    paper.create_account(tmp_path, 100_000, slippage_bps=0)
    events(tmp_path, [event(allot=0.1)])
    with pytest.raises(ValueError, match=r"配股|rights"):
        paper.create_order(tmp_path, SYM, "buy", qty=100, as_of=clock(EX))
    assert paper.load_orders(tmp_path) == []
    assert paper.load_fills(tmp_path) == []


def test_stats_explicitly_warn_that_fifo_realized_profit_excludes_corporate_adjustments(tmp_path):
    seed(tmp_path)
    events(tmp_path, [event()])
    paper.apply_corporate_actions(tmp_path, EX.isoformat())
    assert any("公司行动" in warning for warning in paper.stats(tmp_path)["warnings"])


def test_restart_repairs_stale_materialized_bonus_position(tmp_path):
    seed(tmp_path)
    old = (paper._root(tmp_path) / "positions.json").read_text()
    events(tmp_path, [event(dividend=0.0, bonus=0.1, ex_factor=1.1)])
    paper.apply_corporate_actions(tmp_path, EX.isoformat())
    (paper._root(tmp_path) / "positions.json").write_text(old)
    assert paper.apply_corporate_actions(tmp_path, EX.isoformat()) == 0
    assert paper.load_positions(tmp_path)[SYM]["qty"] == 1100


def test_requested_unsupported_symbol_cannot_publish_other_held_events_first(tmp_path):
    seed(tmp_path)
    other = "000001.SZ"
    order, error = paper.create_order(tmp_path, other, "buy", qty=100, as_of=clock(date(2024, 6, 3)))
    assert error is None
    before = paper.load_fills(tmp_path)
    events(tmp_path, [event(), event(symbol=other, allot=0.1)])
    with pytest.raises(ValueError):
        paper.create_order(tmp_path, other, "buy", qty=100, as_of=clock(EX))
    assert paper.load_fills(tmp_path) == before
    with pytest.raises(ValueError):
        paper._fill_order(tmp_path, order, 9, EX.isoformat(), as_of=clock(EX))
    assert paper.load_fills(tmp_path) == before


def snapshot_account(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in paper._root(root).rglob("*") if p.is_file()}


@pytest.mark.parametrize("case", ["product", "receivable_sum", "cash_sum"])
def test_derived_overflow_rejects_entire_batch_without_mutating_account_files(tmp_path, case):
    seed(tmp_path)
    rows = [event(dividend=1e308)] if case == "product" else [
        event(dividend=1e305, pay_date=EX if case == "cash_sum" else None),
        event(trade_date=date(2024, 6, 5), dividend=1e305,
              pay_date=date(2024, 6, 5) if case == "cash_sum" else None),
    ]
    events(tmp_path, rows)
    before = snapshot_account(tmp_path)
    with pytest.raises(ValueError, match=r"有限|finite|overflow|金额"):
        paper.apply_corporate_actions(tmp_path, "2024-06-06")
    assert snapshot_account(tmp_path) == before


def test_late_future_payment_evidence_survives_source_loss_and_restart(tmp_path):
    cash = seed(tmp_path)
    events(tmp_path, [event()])
    paper.apply_corporate_actions(tmp_path, EX.isoformat())
    original = paper.load_fills(tmp_path)[-1]
    events(tmp_path, [event(pay_date=date(2024, 6, 7))])
    assert paper.apply_corporate_actions(tmp_path, "2024-06-05") == 1
    assert paper.load_fills(tmp_path)[-1]["kind"] == "corp_evidence"
    assert paper.load_fills(tmp_path)[1] == original
    assert paper.get_account(tmp_path)["cash"] == cash
    assert paper.receivable_balance(tmp_path) == 1000
    (tmp_path / "adj_factor/all.parquet").unlink()
    paper.rebuild_positions(tmp_path)
    assert paper.apply_corporate_actions(tmp_path, "2024-06-07") == 1
    assert paper.get_account(tmp_path)["cash"] == cash + 1000
    assert paper.receivable_balance(tmp_path) == 0
    rows = paper.load_fills(tmp_path)
    assert paper.apply_corporate_actions(tmp_path, "2024-06-10") == 0
    assert paper.load_fills(tmp_path) == rows


@pytest.mark.parametrize("field", ["pay_date", "listing_date"])
@pytest.mark.parametrize("late", [False, True])
def test_conflicting_initial_or_late_date_evidence_rejects_before_any_writes(tmp_path, field, late):
    seed(tmp_path)
    details = {field: date(2024, 6, 7)}
    rows = event(bonus=0.1, **({} if late else details))
    events(tmp_path, [rows])
    paper.apply_corporate_actions(tmp_path, EX.isoformat())
    if late:
        events(tmp_path, [event(bonus=0.1, **details)])
        paper.apply_corporate_actions(tmp_path, "2024-06-05")
    before = snapshot_account(tmp_path)
    events(tmp_path, [event(bonus=0.1, **{field: date(2024, 6, 10)}),
                      event(trade_date=date(2024, 6, 5))])
    with pytest.raises(ValueError, match=r"日期|date|证据"):
        paper.apply_corporate_actions(tmp_path, "2024-06-06")
    assert snapshot_account(tmp_path) == before


def test_future_listing_and_payment_evidence_together_survive_source_loss(tmp_path):
    cash = seed(tmp_path)
    events(tmp_path, [event(bonus=0.1)])
    paper.apply_corporate_actions(tmp_path, EX.isoformat())
    events(tmp_path, [event(bonus=0.1, pay_date=date(2024, 6, 7), listing_date=date(2024, 6, 10))])
    assert paper.apply_corporate_actions(tmp_path, "2024-06-05") == 1
    evidence = paper.load_fills(tmp_path)[-1]
    assert evidence["kind"] == "corp_evidence"
    assert evidence["pay_date"] == "2024-06-07"
    assert evidence["listing_date"] == "2024-06-10"
    assert paper.apply_corporate_actions(tmp_path, "2024-06-06") == 0
    (tmp_path / "adj_factor/all.parquet").unlink()
    paper.rebuild_positions(tmp_path)
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 7)) == (1000, 0)
    assert paper.apply_corporate_actions(tmp_path, "2024-06-07") == 1
    assert paper.get_account(tmp_path)["cash"] == cash + 1000
    assert paper.sell_capacity(tmp_path, SYM, as_of=date(2024, 6, 10)) == (1100, 0)


def test_older_listing_release_record_is_also_binding_evidence(tmp_path):
    seed(tmp_path)
    events(tmp_path, [event(bonus=0.1)])
    paper.apply_corporate_actions(tmp_path, EX.isoformat())
    entitlement = paper.load_fills(tmp_path)[-1]
    paper._append_fill(tmp_path, {
        "schema_version": 2, "kind": "corp_release", "event_id": entitlement["event_id"],
        "symbol": SYM, "date": "2024-06-05", "available_date": "2024-06-07",
    })
    events(tmp_path, [event(bonus=0.1, listing_date=date(2024, 6, 10))])
    before = snapshot_account(tmp_path)
    with pytest.raises(ValueError, match=r"日期|证据"):
        paper.apply_corporate_actions(tmp_path, "2024-06-06")
    assert snapshot_account(tmp_path) == before
