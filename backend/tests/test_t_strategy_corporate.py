"""End-to-end T-page accounting across raw ex-date price discontinuities."""
import json
from datetime import date, datetime

import polars as pl
import pytest
from test_t_strategy import D1, D2, SYMBOL, client, config, minutes, setup_repo

from app.strategy import paper


def corporate_repo(root, *, dividend=1.0, bonus=0.0, price=9.0, **extra):
    repo = setup_repo(root, [minutes(D1), minutes(D2, price)])
    daily_path = root / "kline_daily" / "all.parquet"
    daily = pl.read_parquet(daily_path).with_columns(
        *[pl.when(pl.col("date") == D2).then(price).otherwise(pl.col(k)).alias(k)
          for k in ("open", "high", "low", "close")]
    )
    daily.write_parquet(daily_path)
    factor = root / "adj_factor" / "all.parquet"
    pl.DataFrame([{"symbol": SYMBOL, "trade_date": D2, "ex_factor": 10 / price,
                   "dividend": dividend, "bonus": bonus, "allot": 0.0, **extra}]).write_parquet(factor)
    repo.rebuild_views()
    return repo


def test_page_dividend_preserves_economic_nav_without_spendable_fake_cash(tmp_path):
    c = client(corporate_repo(tmp_path))
    response = c.post("/api/t-strategy/backtest", json=config(
        base_qty=1000, commission_pct=0, slippage_bps=0))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["stats"]["fills"] == 1  # Only the initial purchase; no fake dividend trade.
    assert result["stats"]["final_nav"] == 99994.9
    assert result["stats"]["benchmark_nav"] == 99994.9
    assert result["stats"]["excess_pnl"] == 0
    assert result["stats"]["receivable"] == 1000
    assert result["nav"][-1]["cash"] == 89994.9
    assert result["nav"][-1]["mv"] == 9000
    assert result["nav"][-1]["receivable"] == 1000
    imported = c.post(f"/api/t-strategy/runs/{result['id']}/paper").json()
    aid = imported["account"]
    assert imported["frozen"]
    assert paper.load_positions(tmp_path, aid)[SYMBOL]["qty"] == 1000
    assert paper.get_account(tmp_path, aid)["cash"] == 89994.9
    assert paper.receivable_balance(tmp_path, aid) == 1000
    paper.rebuild_positions(tmp_path, aid)
    assert paper.receivable_balance(tmp_path, aid) == 1000


def test_page_bonus_uses_share_entitlement_and_keeps_unknown_listing_locked(tmp_path):
    c = client(corporate_repo(tmp_path, dividend=0, bonus=0.1, price=9.09))
    response = c.post("/api/t-strategy/backtest", json=config(
        base_qty=1000, commission_pct=0, slippage_bps=0))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["stats"]["final_nav"] == 99993.9
    assert result["stats"]["benchmark_nav"] == 99993.9
    aid = c.post(f"/api/t-strategy/runs/{result['id']}/paper").json()["account"]
    assert paper.load_positions(tmp_path, aid)[SYMBOL]["qty"] == 1100
    assert paper.sell_capacity(tmp_path, SYMBOL, aid, as_of=D2)[0] == 1000
    assert any("上市" in warning or "锁定" in warning for warning in result["warnings"])


@pytest.mark.parametrize("extra", [{"allot": 0.2}, {"dividend": None}])
def test_unsupported_corporate_window_never_publishes_partial_run(tmp_path, extra):
    c = client(corporate_repo(tmp_path, **extra))
    response = c.post("/api/t-strategy/backtest", json=config())
    assert response.status_code == 400
    assert ("配股" if "allot" in extra else "明细") in response.json()["detail"]
    assert paper.list_account_ids(tmp_path) == []
    assert not list((tmp_path / "research" / "t_strategy").glob("*/result.json"))


def test_manual_step_keeps_pre_ex_day_dividend_after_current_day_sale(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = corporate_repo(tmp_path, price=9.5, ex_factor=10 / 9)
    stamp = datetime(2026, 11, 5, 9, 45, 10)
    monkeypatch.setattr(t_strategy, "_naive_now", lambda: stamp)
    path = tmp_path / "kline_minute" / f"date={D2}" / "part.parquet"
    minutes(D2, 9.5).head(15).write_parquet(path)
    paper.create_account(tmp_path, 100000, account_id="sim_t", commission_pct=0, slippage_bps=0)
    order, error = paper.create_order(tmp_path, SYMBOL, "buy", account_id="sim_t",
                                      qty=1000, ref_price=10, as_of=datetime(2026, 11, 4, 10))
    assert error is None
    paper._fill_order(tmp_path, order, 10, str(D1), "sim_t", as_of=datetime(2026, 11, 4, 10))
    c = client(repo)
    body = config(base_qty=1000, commission_pct=0, slippage_bps=0, account="sim_t")
    first = c.post("/api/t-strategy/manual/step", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "order_created"
    assert first.json()["order"]["side"] == "sell"  # Uses ex-reference 9, not yesterday's 10.
    assert paper.receivable_balance(tmp_path, "sim_t") == 1000

    assert paper.get_account(tmp_path, "sim_t")["cash"] == 89994.9
    stamp = datetime(2026, 11, 5, 9, 47, 10)
    minutes(D2, 9.5).head(17).write_parquet(path)
    second = c.post("/api/t-strategy/manual/step", json=body)
    assert second.status_code == 200, second.text
    assert len(second.json()["filled"]) == 1
    assert paper.load_positions(tmp_path, "sim_t")[SYMBOL]["qty"] == 700
    assert paper.receivable_balance(tmp_path, "sim_t") == 1000


def seed_t_fill(root, day, side, qty, price, purpose, *, symbol=SYMBOL):
    stamp = datetime.combine(day, datetime.min.time()).replace(hour=10)
    order, error = paper.create_order(
        root, symbol, side, qty=qty, ref_price=price, account_id="sim_t",
        source="t_strategy", order_type="t_signal", as_of=stamp,
    )
    assert error is None
    order.update(t_purpose=purpose, t_signal_at=stamp.isoformat())
    paper.save_order(root, order, "sim_t")
    assert paper._fill_order(root, order, price, str(day), "sim_t", as_of=stamp)


@pytest.mark.parametrize("bonus_symbol,sold,expected", [
    ("000001.SZ", 300, 300),
    (SYMBOL, 600, 1200),
    (SYMBOL, 300, 600),
])
def test_manual_bonus_debt_is_symbol_scoped_and_survives_restart(
    tmp_path, monkeypatch, bonus_symbol, sold, expected,
):
    from app.services import t_strategy

    repo = corporate_repo(tmp_path, dividend=0.0, bonus=1.0, price=5.0)
    factors = tmp_path / "adj_factor" / "all.parquet"
    pl.read_parquet(factors).with_columns(pl.lit(bonus_symbol).alias("symbol")).write_parquet(factors)
    daily_path = tmp_path / "kline_daily" / "all.parquet"
    daily = pl.read_parquet(daily_path)
    pl.concat([daily, daily.with_columns(pl.lit("000001.SZ").alias("symbol"))]).write_parquet(
        daily_path)
    repo.rebuild_views()
    paper.create_account(tmp_path, 100000, account_id="sim_t", slippage_bps=0)
    seed_t_fill(tmp_path, date(2026, 11, 3), "buy", 600, 10.0, "base")
    seed_t_fill(tmp_path, D1, "sell", sold, 10.0, "sell")
    if bonus_symbol != SYMBOL:
        stamp = datetime(2026, 11, 4, 10)
        order, error = paper.create_order(
            tmp_path, bonus_symbol, "buy", qty=100, ref_price=10.0,
            account_id="sim_t", as_of=stamp,
        )
        assert error is None
        assert paper._fill_order(tmp_path, order, 10.0, str(D1), "sim_t", as_of=stamp)
    minute_path = tmp_path / "kline_minute" / f"date={D2}" / "part.parquet"
    minutes(D2, 5.0).head(15).write_parquet(minute_path)
    monkeypatch.setattr(t_strategy, "_naive_now", lambda: datetime(2026, 11, 5, 9, 45, 10))
    body = config(base_qty=600, trade_qty=sold, account="sim_t")
    response = client(repo).post("/api/t-strategy/manual/step", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["order"]["qty"] == expected
    assert response.json()["order"]["t_purpose"] == "restore"
    actions = [f for f in paper.load_fills(tmp_path, "sim_t") if f["kind"] == "corp_action"]
    assert len(actions) == (0 if sold == 600 else 1)
    assert paper.receivable_balance(tmp_path, "sim_t") == 0
    if sold == 600:
        assert paper.load_positions(tmp_path, "sim_t")[SYMBOL]["qty"] == 0

    # New service instances and repeat requests reconstruct once from economic
    # evidence plus fills, even when a stale snapshot claims an inflated debt.
    state_path = paper.accounts_base(tmp_path) / "sim_t" / "t_strategy.json"
    state = json.loads(state_path.read_text())
    state["obligation"] = 999900
    if bonus_symbol == SYMBOL and sold == 300:
        state.pop("corporate_events", None)  # Own entitlement ledger retains the economics.
    state.setdefault("corporate_events", []).append({
        "symbol": "000002.SZ", "trade_date": str(D2), "dividend": 0.0,
        "bonus": 9.0, "allot": 0.0,
    })
    state_path.write_text(json.dumps(state))
    # Economic evidence must survive source retention even when the account
    # had no ex-date inventory and therefore no entitlement ledger record.
    if bonus_symbol == SYMBOL:
        pl.read_parquet(factors).head(0).write_parquet(factors)
        repo.rebuild_views()
    for _ in range(2):
        again = client(repo).post("/api/t-strategy/manual/step", json=body)
        assert again.status_code == 200, again.text
        assert again.json()["status"] == "already_evaluated"
    minutes(D2, 5.0).head(17).write_parquet(minute_path)
    monkeypatch.setattr(t_strategy, "_naive_now", lambda: datetime(2026, 11, 5, 9, 47, 10))
    later = client(repo).post("/api/t-strategy/manual/step", json=body)
    assert later.status_code == 200, later.text
    assert [f["qty"] for f in later.json()["filled"]] == [expected]
    assert len(paper.load_orders(tmp_path, "sim_t")) == 3 + (bonus_symbol != SYMBOL)


@pytest.mark.parametrize("bonus", [0.1, 0.0001, 1e308])
def test_zero_position_unsupported_debt_units_reject_before_any_order_write(
    tmp_path, monkeypatch, bonus,
):
    from app.services import t_strategy

    repo = corporate_repo(tmp_path, dividend=0.0, bonus=bonus, price=5.0)
    paper.create_account(tmp_path, 100000, account_id="sim_t", slippage_bps=0)
    seed_t_fill(tmp_path, date(2026, 11, 3), "buy", 600, 10.0, "base")
    seed_t_fill(tmp_path, D1, "sell", 600, 10.0, "sell")
    minute_path = tmp_path / "kline_minute" / f"date={D2}" / "part.parquet"
    minutes(D2, 5.0).head(15).write_parquet(minute_path)
    monkeypatch.setattr(t_strategy, "_naive_now", lambda: datetime(2026, 11, 5, 9, 45, 10))
    account_before = paper.get_account(tmp_path, "sim_t")
    fills_before = paper.load_fills(tmp_path, "sim_t")
    orders_before = paper.load_orders(tmp_path, "sim_t")
    response = client(repo).post("/api/t-strategy/manual/step", json=config(
        base_qty=600, trade_qty=600, account="sim_t",
    ))
    assert response.status_code == 400, response.text
    assert "恢复底仓义务" in response.json()["detail"]
    assert paper.get_account(tmp_path, "sim_t") == account_before
    assert paper.load_fills(tmp_path, "sim_t") == fills_before
    assert paper.load_orders(tmp_path, "sim_t") == orders_before


def test_manual_deferred_entitlement_is_applied_before_same_ex_day_t_sell(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = corporate_repo(tmp_path, dividend=0.0, bonus=1.0, price=5.0)
    factors = tmp_path / "adj_factor" / "all.parquet"
    event = pl.read_parquet(factors).with_columns(pl.lit(D1).alias("trade_date"))
    event.head(0).write_parquet(factors)
    repo.rebuild_views()
    paper.create_account(tmp_path, 100000, account_id="sim_t", slippage_bps=0)
    seed_t_fill(tmp_path, date(2026, 11, 3), "buy", 600, 10.0, "base")
    seed_t_fill(tmp_path, D1, "sell", 300, 10.0, "sell")
    # The source was first learned after the ex-day sell, so append order in
    # the entitlement ledger differs from the economic event chronology.
    event.write_parquet(factors)
    repo.rebuild_views()
    minutes(D2, 5.0).head(15).write_parquet(
        tmp_path / "kline_minute" / f"date={D2}" / "part.parquet")
    monkeypatch.setattr(t_strategy, "_naive_now", lambda: datetime(2026, 11, 5, 9, 45, 10))
    response = client(repo).post("/api/t-strategy/manual/step", json=config(account="sim_t"))
    assert response.status_code == 200, response.text
    assert response.json()["order"]["qty"] == 300
    assert paper.load_positions(tmp_path, "sim_t")[SYMBOL]["qty"] == 900


def test_replay_full_exit_bonus_changes_debt_without_fabricating_entitlement(tmp_path):
    previous = date(2026, 11, 3)
    sell_day = minutes(D1, 10.2).with_columns(
        *[pl.when(pl.col("datetime") >= datetime(2026, 11, 4, 9, 46))
          .then(11.0).otherwise(pl.col(k)).alias(k) for k in ("close", "high")],
        pl.when(pl.col("datetime") > datetime(2026, 11, 4, 9, 46))
        .then(11.0).otherwise(pl.col("open")).alias("open"),
    )
    repo = setup_repo(tmp_path, [minutes(previous), sell_day, minutes(D2, 5.5)])
    daily_path = tmp_path / "kline_daily" / "all.parquet"
    daily = pl.read_parquet(daily_path)
    prior = daily.head(1).with_columns(pl.lit(date(2026, 11, 2)).alias("date"))
    daily = pl.concat([prior, daily]).with_columns(
        *[pl.when(pl.col("date") == D1).then(11.0)
          .when(pl.col("date") == D2).then(5.5).otherwise(pl.col(k)).alias(k)
          for k in ("open", "high", "low", "close")],
    )
    daily.write_parquet(daily_path)
    pl.DataFrame([{"symbol": SYMBOL, "trade_date": D2, "ex_factor": 2.0,
                   "dividend": 0.0, "bonus": 1.0, "allot": 0.0}]).write_parquet(
                       tmp_path / "adj_factor" / "all.parquet")
    repo.rebuild_views()
    c = client(repo)
    response = c.post("/api/t-strategy/backtest", json=config(
        start=str(previous), trade_qty=600, slippage_bps=0,
    ))
    assert response.status_code == 200, response.text
    run = response.json()
    assert [(f["date"], f["side"], f["qty"]) for f in run["fills"]] == [
        (str(previous), "buy", 600), (str(D1), "sell", 600), (str(D2), "buy", 1200),
    ]
    assert run["stats"]["round_trips"] == 1
    assert not [e for e in run["events"] if e.get("kind") == "corp_action"]
    aid = c.post(f"/api/t-strategy/runs/{run['id']}/paper").json()["account"]
    assert paper.load_positions(tmp_path, aid)[SYMBOL]["qty"] == 1200
    assert paper.receivable_balance(tmp_path, aid) == 0


def test_same_minute_dedup_still_retains_new_economic_evidence(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = corporate_repo(tmp_path, dividend=0.0, bonus=1.0, price=5.0)
    factor_path = tmp_path / "adj_factor" / "all.parquet"
    event = pl.read_parquet(factor_path)
    event.head(0).write_parquet(factor_path)
    paper.create_account(tmp_path, 100000, account_id="sim_t", slippage_bps=0)
    seed_t_fill(tmp_path, date(2026, 11, 3), "buy", 600, 10.0, "base")
    seed_t_fill(tmp_path, D1, "sell", 600, 10.0, "sell")
    minute_path = tmp_path / "kline_minute" / f"date={D2}" / "part.parquet"
    minutes(D2, 5.0).head(15).write_parquet(minute_path)
    now = datetime(2026, 11, 5, 9, 45, 10)
    monkeypatch.setattr(t_strategy, "_naive_now", lambda: now)
    c = client(repo)
    body = config(base_qty=600, trade_qty=600, account="sim_t")
    first = c.post("/api/t-strategy/manual/step", json=body).json()
    assert first["order"]["qty"] == 600
    paper.cancel_order(tmp_path, first["order"]["id"], "sim_t")
    order_count = len(paper.load_orders(tmp_path, "sim_t"))
    event.write_parquet(factor_path)
    second = c.post("/api/t-strategy/manual/step", json=body)
    assert second.status_code == 200, second.text
    assert second.json()["status"] == "already_evaluated"
    assert len(paper.load_orders(tmp_path, "sim_t")) == order_count
    event.head(0).write_parquet(factor_path)
    now = datetime(2026, 11, 5, 9, 46, 10)
    minutes(D2, 5.0).head(16).write_parquet(minute_path)
    later = c.post("/api/t-strategy/manual/step", json=body)
    assert later.status_code == 200, later.text
    assert later.json()["order"]["qty"] == 1200
    assert not [f for f in paper.load_fills(tmp_path, "sim_t") if f.get("kind") == "corp_action"]
