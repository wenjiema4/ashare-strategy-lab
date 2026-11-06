"""Cost/VWAP guard must be prefix-only and must never suppress restoration."""
from datetime import datetime, timedelta

import pytest

from app.services.t_strategy import TConfig, decision, with_signal_context
from app.trading_costs import FeePolicy


def bar(close=10.2, **extra):
    return {"datetime": datetime(2026, 11, 5, 10), "close": close,
            "volume": 1000.0, "t_vwap": close / 1.01,
            "t_prev_close": close + 0.01, **extra}


def test_context_uses_only_prefix_and_resets_each_day():
    stamp = datetime(2026, 11, 4, 9, 31)
    rows = [{"datetime": stamp, "close": 10.0, "volume": 100},
            {"datetime": stamp + timedelta(minutes=1), "close": 12.0, "volume": 300},
            {"datetime": stamp + timedelta(days=1), "close": 8.0, "volume": 100}]
    result = with_signal_context(rows)
    assert result[0]["t_vwap"] == 10
    assert result[1]["t_vwap"] == 11.5
    assert result[1]["t_prev_close"] == 10
    assert result[2]["t_vwap"] == 8
    assert result[2]["t_prev_close"] is None
    assert with_signal_context(rows[:1])[0] == result[0]
    assert "t_vwap" not in rows[0]


def test_zero_volume_context_is_unavailable():
    assert with_signal_context([bar(volume=0)])[0]["t_vwap"] is None


def test_guard_rejects_uneconomic_small_trade_but_classic_is_unchanged():
    config = TConfig(mode="cost_vwap", trade_qty=100)
    assert decision(config, {"initialized": True}, bar(1.02), 1, 600, 600) is None
    assert decision(TConfig(trade_qty=100), {"initialized": True}, bar(1.02), 1, 600, 600)["side"] == "sell"


@pytest.mark.parametrize("extra", [
    {"t_vwap": 10.19}, {"t_prev_close": 10.1}, {"t_vwap": None},
    {"t_prev_close": None}, {"volume": 0},
])
def test_guard_skips_unconfirmed_or_missing_context(extra):
    assert decision(TConfig(mode="cost_vwap"), {"initialized": True}, bar(**extra), 10, 600, 600) is None


def test_guard_passes_economic_reversal_and_uses_actual_account_fees():
    config = TConfig(mode="cost_vwap", slippage_bps=0)
    assert decision(config, {"initialized": True}, bar(), 10, 600, 600)["side"] == "sell"
    expensive = FeePolicy(commission_pct=0.01, stamp_tax_pct=0.001, slippage_bps=100)
    assert decision(config, {"initialized": True}, bar(), 10, 600, 600, fee_policy=expensive) is None


def test_guard_never_blocks_existing_debt_or_tail_restore():
    config = TConfig(mode="cost_vwap", min_net_profit=10000)
    assert decision(config, {"obligation": 300, "sell_price": 10.4}, bar(10), 10, 300, 0)["purpose"] == "restore"
    tail = bar(12, datetime=datetime(2026, 11, 5, 14, 55), t_vwap=None)
    assert decision(config, {"obligation": 300, "sell_price": 10.4}, tail, 10, 300, 0)["qty"] == 300


@pytest.mark.parametrize("params", [{"mode": "unknown"}, {"vwap_deviation_pct": -1},
                                    {"min_net_profit": float("nan")}, {"min_net_profit": -1}])
def test_guard_config_rejects_invalid_inputs(params):
    with pytest.raises(ValueError):
        TConfig(**params)


def guarded_repo(tmp_path):
    import polars as pl
    from test_t_strategy import D1, D2, minutes, setup_repo
    frame = minutes(D2).with_columns(
        pl.when(pl.col("datetime").dt.time() == datetime(2026, 1, 1, 9, 45).time())
        .then(10.3)
        .when(pl.col("datetime").dt.time() == datetime(2026, 1, 1, 9, 46).time())
        .then(10.2).otherwise(10.0).alias("close"),
        pl.when(pl.col("datetime").dt.time() == datetime(2026, 1, 1, 9, 47).time())
        .then(10.2).otherwise(10.0).alias("open"),
        pl.lit(10.35).alias("high"), pl.lit(9.95).alias("low"),
    )
    return setup_repo(tmp_path, [minutes(D1), frame])


def test_guarded_http_replay_waits_for_reversal_and_fills_later(tmp_path):
    from test_t_strategy import client, config
    response = client(guarded_repo(tmp_path)).post(
        "/api/t-strategy/backtest", json=config(mode="cost_vwap", slippage_bps=0),
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["stats"]["round_trips"] == 1
    assert result["fills"][1]["ts"][11:19] == "09:46:00"
    assert result["fills"][1]["price"] == 10.2
    assert result["params"]["mode"] == "cost_vwap"


@pytest.mark.parametrize("commission,request_commission,expected", [
    (0.01, 0.00025, "evaluated"), (0.00025, 0.01, "order_created"),
])
def test_manual_guard_uses_account_fee_rules(tmp_path, monkeypatch, commission, request_commission, expected):
    import polars as pl
    from test_t_strategy import D1, SYMBOL, client, config

    from app.services import t_strategy
    from app.strategy import paper
    repo = guarded_repo(tmp_path)
    paper.create_account(tmp_path, 100000, account_id="sim_t", commission_pct=commission, slippage_bps=0)
    stamp = datetime.combine(D1, datetime.min.time()).replace(hour=9, minute=31)
    order, error = paper.create_order(tmp_path, SYMBOL, "buy", account_id="sim_t", qty=600,
                                     order_type="t_signal", source="t_strategy", ref_price=10, as_of=stamp)
    assert error is None
    order.update(t_purpose="base", t_signal_at=stamp.isoformat())
    paper.save_order(tmp_path, order, "sim_t")
    assert paper._fill_order(tmp_path, order, 10, str(D1), "sim_t", as_of=stamp)
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 5, 9, 46, 10))
    original = repo.get_minute_range
    monkeypatch.setattr(repo, "get_minute_range", lambda *a, **kw: original(*a, **kw).filter(
        pl.col("datetime") <= datetime(2026, 11, 5, 9, 46)))
    response = client(repo).post("/api/t-strategy/manual/step", json={
        **config(mode="cost_vwap", commission_pct=request_commission, slippage_bps=0), "account": "sim_t",
    })
    assert response.status_code == 200, response.text
    assert response.json()["status"] == expected


def test_old_classic_binding_remains_compatible(tmp_path, monkeypatch):
    import json

    import polars as pl
    from test_t_strategy import client, config, setup_repo

    from app.services import t_strategy
    from app.strategy import paper
    repo = setup_repo(tmp_path)
    original = repo.get_minute_range
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 5, 10, 0, 10))
    monkeypatch.setattr(repo, "get_minute_range", lambda *a, **kw: original(*a, **kw).filter(
        pl.col("datetime") <= datetime(2026, 11, 5, 10)))
    c = client(repo)
    body = {**config(), "account": "sim_t"}
    assert c.post("/api/t-strategy/manual/step", json=body).status_code == 200
    path = paper.accounts_base(tmp_path) / "sim_t" / "t_strategy.json"
    state = json.loads(path.read_text())
    for key in ("mode", "vwap_deviation_pct", "min_net_profit"):
        state["params"].pop(key)
    path.write_text(json.dumps(state))
    response = c.post("/api/t-strategy/manual/step", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "already_evaluated"
