"""Real local repository and private paper ledger contracts for the T page."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.minute_adjust import mark_minute_basis_raw
from app.strategy import paper
from app.tickflow.repository import DataStore, KlineRepository

SYMBOL = "600000.SH"
D1, D2 = date(2026, 11, 4), date(2026, 11, 5)


def minutes(day, price=10.0):
    stamps = [
        datetime.combine(day, datetime.min.time()) + timedelta(minutes=m)
        for m in [*range(571, 691), *range(781, 901)]
    ]
    return pl.DataFrame(
        {
            "symbol": [SYMBOL] * 240,
            "datetime": stamps,
            "open": [price] * 240,
            "high": [price + 0.02] * 240,
            "low": [price - 0.02] * 240,
            "close": [price] * 240,
            "volume": [1000.0] * 240,
            "amount": [price * 100000] * 240,
        }
    )


def setup_repo(tmp_path, frames=None):
    mark_minute_basis_raw(tmp_path)
    frames = frames if frames is not None else [minutes(D1), minutes(D2)]
    for frame in frames:
        day = frame["datetime"][0].date()
        target = tmp_path / "kline_minute" / f"date={day}"
        target.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target / "part.parquet")
    target = tmp_path / "kline_daily"
    target.mkdir(exist_ok=True)
    pl.DataFrame(
        {
            "symbol": [SYMBOL] * 3,
            "date": [date(2026, 11, 3), D1, D2],
            "open": [10.0] * 3,
            "high": [11.0] * 3,
            "low": [9.0] * 3,
            "close": [10.0] * 3,
            "volume": [100000.0] * 3,
            "amount": [1000000.0] * 3,
        }
    ).write_parquet(target / "all.parquet")
    repo = KlineRepository(DataStore(tmp_path))
    return repo


def client(repo):
    from app.custom.t_strategy import router

    app = FastAPI()
    app.state.repo = repo
    app.include_router(router)
    return TestClient(app)


def config(**kwargs):
    return {"symbol": SYMBOL, "start": str(D1), "end": str(D2), **kwargs}


def test_config_bounds_and_empty_window_leave_accounts_untouched(tmp_path):
    c = client(setup_repo(tmp_path))
    for override in (
        {"base_qty": 650},
        {"trade_qty": 700},
        {"initial_cash": -1},
        {"rise_pct": 0},
        {"start_time": "12:00"},
        {"end": "2027-03-01"},
        {"symbol": "../bad"},
        {"unknown": 1},
    ):
        assert c.post("/api/t-strategy/backtest", json=config(**override)).status_code == 422
    response = c.post("/api/t-strategy/backtest", json=config(start="2025-11-04", end="2025-11-05"))
    assert response.status_code == 400
    assert paper.list_account_ids(tmp_path) == []
    assert not (tmp_path / "alerts").exists()


def test_repository_raw_accessor_requires_evidence_and_bypasses_projection(tmp_path):
    repo = setup_repo(tmp_path)
    factors = tmp_path / "adj_factor"
    pl.DataFrame(
        {"symbol": [SYMBOL], "trade_date": [date(2026, 11, 6)], "ex_factor": [1.25]}
    ).write_parquet(factors / "all.parquet")
    raw = repo.get_minute_range([SYMBOL], D1, D1, raw=True)
    adjusted = repo.get_minute_range([SYMBOL], D1, D1)
    assert raw["close"][0] == 10
    assert adjusted["close"][0] == 8
    (tmp_path / "kline_minute" / ".raw_basis").unlink()
    with pytest.raises(ValueError, match="原始"):
        repo.get_minute_range([SYMBOL], D1, D1, raw=True)


def test_replay_real_fees_t1_next_minute_and_frozen_import(tmp_path):
    d2 = minutes(D2).with_columns(
        pl.when(pl.col("datetime").dt.time() == datetime(2026, 1, 1, 9, 45).time())
        .then(10.2)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("datetime").dt.time() == datetime(2026, 1, 1, 9, 46).time())
        .then(10.2)
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.lit(10.3).alias("high"),
    )
    c = client(setup_repo(tmp_path, [minutes(D1), d2]))
    response = c.post("/api/t-strategy/backtest", json=config(slippage_bps=0))
    assert response.status_code == 200, response.text
    run = response.json()
    assert [(f["date"], f["side"], f["qty"], f["price"]) for f in run["fills"]] == [
        (str(D1), "buy", 600, 10),
        (str(D2), "sell", 300, 10.2),
        (str(D2), "buy", 300, 10),
    ]
    assert [f["ts"][11:16] for f in run["fills"]] == ["09:31", "09:45", "09:46"]
    assert run["stats"]["round_trips"] == 1
    assert run["stats"]["fees"] == 16.65
    assert run["stats"]["final_nav"] == 100043.35
    assert run["coverage"]["bars"] == 480
    assert paper.list_account_ids(tmp_path) == []
    imported = c.post(f"/api/t-strategy/runs/{run['id']}/paper", json={}).json()
    assert imported["frozen"] is True
    assert c.post(f"/api/t-strategy/runs/{run['id']}/paper", json={}).json() == imported
    assert len(paper.load_fills(tmp_path, imported["account"])) == 3
    assert paper.get_account(tmp_path, imported["account"])["status"] == "frozen"
    assert c.get("/api/t-strategy/runs/../../bad").status_code == 404


@pytest.mark.parametrize("bad", ["missing", "duplicate", "lunch"])
def test_bad_minute_session_fails_closed_without_run_or_account(tmp_path, bad):
    frame = minutes(D1)
    if bad == "missing":
        frame = frame.slice(1)
    elif bad == "duplicate":
        frame = pl.concat([frame.slice(1), frame.slice(1, 1)])
    else:
        frame = frame.with_columns(
            pl.when(pl.col("datetime") == datetime(2026, 11, 4, 13, 1))
            .then(datetime(2026, 11, 4, 12, 1))
            .otherwise(pl.col("datetime"))
            .alias("datetime")
        )
    c = client(setup_repo(tmp_path, [frame, minutes(D2)]))
    response = c.post("/api/t-strategy/backtest", json=config())
    assert response.status_code == 400, response.text
    assert paper.list_account_ids(tmp_path) == []
    assert not list((tmp_path / "research" / "t_strategy").glob("*/result.json"))


def test_manual_stale_rejected_then_idempotent_signal_and_later_bar_fill(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    c = client(repo)
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 5, 10, 0, 20))
    stale = c.post("/api/t-strategy/manual/step", json={"account": "sim_t", **config()})
    assert stale.status_code == 200
    assert stale.json()["status"] == "stale_data"
    assert paper.list_account_ids(tmp_path) == []
    # Keep only bars that have really completed; calls never fill using an already printed open.
    original = repo.get_minute_range
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original(*a, **kw).filter(
            pl.col("datetime") <= datetime(2026, 11, 5, 10, 0)
        ),
    )
    first = c.post("/api/t-strategy/manual/step", json={"account": "sim_t", **config()}).json()
    assert first["status"] == "order_created", first
    assert not paper.load_fills(tmp_path, "sim_t")
    again = c.post("/api/t-strategy/manual/step", json={"account": "sim_t", **config()}).json()
    assert again["status"] == "already_evaluated"
    assert len(paper.load_orders(tmp_path, "sim_t")) == 1
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 5, 10, 2, 10))
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original(*a, **kw).filter(
            pl.col("datetime") <= datetime(2026, 11, 5, 10, 2)
        ),
    )
    later = c.post("/api/t-strategy/manual/step", json={"account": "sim_t", **config()}).json()
    assert len(later["filled"]) == 1, later
    assert later["filled"][0]["ts"][11:19] == "10:01:00"


def test_nonunit_corporate_window_rejected_before_account_creation(tmp_path):
    repo = setup_repo(tmp_path)
    pl.DataFrame({"symbol": [SYMBOL], "trade_date": [D2], "ex_factor": [0.9]}).write_parquet(
        tmp_path / "adj_factor" / "all.parquet"
    )
    repo.rebuild_views()
    response = client(repo).post("/api/t-strategy/backtest", json=config())
    assert response.status_code == 400
    assert "公司行动" in response.json()["detail"]
    assert paper.list_account_ids(tmp_path) == []


def test_rejected_fills_and_unrestored_obligation_are_reported(tmp_path):
    d2 = minutes(D2).with_columns(
        pl.when(pl.col("datetime") == datetime(2026, 11, 5, 9, 45))
        .then(10.2)
        .otherwise(10.0)
        .alias("close"),
        pl.when(pl.col("datetime") == datetime(2026, 11, 5, 9, 46))
        .then(10.2)
        .when(pl.col("datetime") >= datetime(2026, 11, 5, 9, 47))
        .then(11.0)
        .otherwise(10.0)
        .alias("open"),
        pl.lit(11.1).alias("high"),
    )
    response = client(setup_repo(tmp_path, [minutes(D1), d2])).post(
        "/api/t-strategy/backtest", json=config(slippage_bps=0)
    )
    assert response.status_code == 200, response.text
    run = response.json()
    assert len(run["fills"]) == 2
    assert run["stats"]["round_trips"] == 0
    assert any("300" in warning for warning in run["warnings"])
    assert any(event["status"] == "rejected" for event in run["events"])
    assert run["events"][-1]["status"] == "expired"


def test_failed_replay_cleans_private_work_without_publishing(tmp_path, monkeypatch):
    repo = setup_repo(tmp_path)

    def fail(*args, **kwargs):
        raise ValueError("injected storage failure")

    monkeypatch.setattr(paper, "_fill_order", fail)
    response = client(repo).post("/api/t-strategy/backtest", json=config())
    assert response.status_code == 400
    assert paper.list_account_ids(tmp_path) == []
    assert not list((tmp_path / "research" / "t_strategy").iterdir())


def test_replay_never_fills_across_lunch_gap(tmp_path):
    d2 = minutes(D2).with_columns(
        pl.when(pl.col("datetime") == datetime(2026, 11, 5, 11, 30))
        .then(10.2)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.lit(10.3).alias("high"),
    )
    run = (
        client(setup_repo(tmp_path, [minutes(D1), d2]))
        .post("/api/t-strategy/backtest", json=config(slippage_bps=0))
        .json()
    )
    assert len(run["fills"]) == 1
    assert any(
        event["status"] == "expired" and "午休" in event["message"] for event in run["events"]
    )


def test_boolean_cash_is_not_a_one_yuan_budget(tmp_path):
    response = client(setup_repo(tmp_path)).post(
        "/api/t-strategy/backtest", json=config(initial_cash=True)
    )
    assert response.status_code == 422


def test_manual_recovers_restoration_debt_from_ledger_after_state_loss(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    paper.create_account(tmp_path, 100000, account_id="sim_t", slippage_bps=0)
    for day, side, qty, price, purpose in (
        (D1, "buy", 600, 10, "base"),
        (D2, "sell", 300, 10.2, "sell"),
    ):
        stamp = datetime.combine(day, datetime(2026, 1, 1, 9, 45).time())
        order, error = paper.create_order(
            tmp_path,
            SYMBOL,
            side,
            qty=qty,
            ref_price=price,
            account_id="sim_t",
            source="t_strategy",
            order_type="t_signal",
            as_of=stamp,
        )
        assert error is None
        order.update(t_purpose=purpose, t_signal_at=stamp.isoformat())
        paper.save_order(tmp_path, order, "sim_t")
        assert paper._fill_order(tmp_path, order, price, str(day), "sim_t", as_of=stamp)
    original = repo.get_minute_range
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 5, 10, 0, 20))
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original(*a, **kw).filter(
            pl.col("datetime") <= datetime(2026, 11, 5, 10, 0)
        ),
    )
    response = (
        client(repo)
        .post(
            "/api/t-strategy/manual/step",
            json={"account": "sim_t", **config(start="1990-01-01", end="2099-01-01")},
        )
        .json()
    )
    assert response["status"] == "order_created", response
    assert response["order"]["t_purpose"] == "restore"
    assert response["order"]["qty"] == 300
    assert paper.get_account(tmp_path, "sim_t")["slippage_bps"] == 0


def test_manual_foreign_pending_order_is_untouched(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    paper.create_account(tmp_path, 100000, account_id="existing", slippage_bps=0)
    order, _ = paper.create_order(
        tmp_path, SYMBOL, "buy", qty=100, ref_price=10, account_id="existing"
    )
    original = repo.get_minute_range
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 5, 10, 0, 20))
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original(*a, **kw).filter(
            pl.col("datetime") <= datetime(2026, 11, 5, 10, 0)
        ),
    )
    response = client(repo).post(
        "/api/t-strategy/manual/step", json={"account": "existing", **config()}
    )
    assert response.status_code == 400
    assert paper.get_order(tmp_path, order["id"], "existing")["status"] == "pending"
    assert not paper.load_fills(tmp_path, "existing")
    assert paper.get_account(tmp_path, "existing")["cash"] == 100000


def test_manual_reports_limit_rejection_without_creating_fill(tmp_path, monkeypatch):
    from app.services import t_strategy

    frame = minutes(D2).with_columns(
        pl.when(pl.col("datetime") == datetime(2026, 11, 5, 10, 2))
        .then(11.0)
        .otherwise(pl.col("open"))
        .alias("open"),
        pl.lit(11.1).alias("high"),
    )
    repo = setup_repo(tmp_path, [minutes(D1), frame])
    original = repo.get_minute_range
    cutoff = datetime(2026, 11, 5, 10, 0)
    monkeypatch.setattr(t_strategy, "cn_now", lambda: cutoff + timedelta(seconds=20))
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original(*a, **kw).filter(pl.col("datetime") <= cutoff),
    )
    c = client(repo)
    assert (
        c.post("/api/t-strategy/manual/step", json={"account": "sim_t", **config()}).json()[
            "status"
        ]
        == "order_created"
    )
    cutoff = datetime(2026, 11, 5, 10, 2)
    response = c.post("/api/t-strategy/manual/step", json={"account": "sim_t", **config()}).json()
    assert response["status"] == "rejected", response
    assert "涨停" in response["message"]
    assert not response["filled"]
    assert not paper.load_fills(tmp_path, "sim_t")


def test_status_defaults_use_last_complete_session_and_90_day_coverage(tmp_path, monkeypatch):
    from app.services import t_strategy

    older = minutes(date(2026, 9, 21)).with_columns(pl.lit("000001.SZ").alias("symbol"))
    full = minutes(D2).with_columns(pl.lit("000001.SZ").alias("symbol"))
    partial = minutes(date(2026, 11, 6)).head(119).with_columns(pl.lit("000001.SZ").alias("symbol"))
    repo = setup_repo(tmp_path, [older, full, partial])
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 6, 11, 30))
    result = client(repo).get("/api/t-strategy/status").json()
    assert result["coverage"] == {
        "start": "2026-09-21",
        "end": "2026-11-05",
        "sessions": 2,
        "bars": 480,
    }
    assert result["defaults"]["start"] == "2026-11-05"
    assert result["defaults"]["end"] == "2026-11-05"
    assert paper.list_account_ids(tmp_path) == []


def test_status_without_minute_data_has_empty_coverage_and_valid_defaults(tmp_path, monkeypatch):
    from app.services import t_strategy

    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 6, 11, 30))
    result = client(setup_repo(tmp_path, [])).get("/api/t-strategy/status").json()
    assert result["coverage"] == {"start": None, "end": None, "sessions": 0, "bars": 0}
    assert result["defaults"]["start"] == "2026-10-07"
    assert result["defaults"]["end"] == "2026-11-06"
    assert paper.list_account_ids(tmp_path) == []


def test_status_does_not_count_duplicate_minutes_as_a_complete_session(tmp_path, monkeypatch):
    from app.services import t_strategy

    bad = minutes(D1)
    bad = pl.concat([bad.slice(1), bad.slice(1, 1)]).with_columns(
        pl.lit("000001.SZ").alias("symbol")
    )
    good = minutes(D2).with_columns(pl.lit("000001.SZ").alias("symbol"))
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 6, 11, 30))
    result = client(setup_repo(tmp_path, [bad, good])).get("/api/t-strategy/status").json()
    assert result["coverage"] == {
        "start": "2026-11-05",
        "end": "2026-11-05",
        "sessions": 1,
        "bars": 240,
    }


def test_manual_delayed_read_uses_actual_creation_clock_and_never_printed_open(
    tmp_path, monkeypatch
):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    service = t_strategy.TStrategyService(repo)
    clock = {"now": datetime(2026, 11, 5, 10, 0, 59), "cutoff": datetime(2026, 11, 5, 10, 0)}
    monkeypatch.setattr(t_strategy, "cn_now", lambda: clock["now"])
    original_read = repo.get_minute_range
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original_read(*a, **kw).filter(pl.col("datetime") <= clock["cutoff"]),
    )
    original_market = service._market

    def delayed_read(*args, **kwargs):
        result = original_market(*args, **kwargs)
        clock["now"] = datetime(2026, 11, 5, 10, 1, 2)
        return result

    monkeypatch.setattr(service, "_market", delayed_read)
    request = t_strategy.ManualRequest(**config(), account="sim_t")
    first = service.manual_step(request)
    assert first["order"]["created_at"] == "2026-11-05T10:01:02+08:00"
    monkeypatch.setattr(service, "_market", original_market)
    clock.update(now=datetime(2026, 11, 5, 10, 2, 10), cutoff=datetime(2026, 11, 5, 10, 2))
    second = service.manual_step(request)
    assert second["status"] == "pending"
    assert not second["filled"]
    assert not paper.load_fills(tmp_path, "sim_t")
    clock.update(now=datetime(2026, 11, 5, 10, 3, 10), cutoff=datetime(2026, 11, 5, 10, 3))
    third = service.manual_step(request)
    assert third["filled"][0]["ts"] == "2026-11-05T10:02:00+08:00"


@pytest.mark.parametrize(
    "delayed", [datetime(2026, 11, 5, 11, 30, 2), datetime(2026, 11, 6, 9, 40)]
)
def test_manual_rechecks_session_and_date_after_read_without_creating_account(
    tmp_path, monkeypatch, delayed
):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    service = t_strategy.TStrategyService(repo)
    clock = {"now": datetime(2026, 11, 5, 11, 29, 59)}
    monkeypatch.setattr(t_strategy, "cn_now", lambda: clock["now"])
    original_read = repo.get_minute_range
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original_read(*a, **kw).filter(
            pl.col("datetime") <= datetime(2026, 11, 5, 11, 29)
        ),
    )
    original_market = service._market

    def delayed_read(*args, **kwargs):
        result = original_market(*args, **kwargs)
        clock["now"] = delayed
        return result

    monkeypatch.setattr(service, "_market", delayed_read)
    result = service.manual_step(t_strategy.ManualRequest(**config()))
    assert result["status"] in ("outside_session", "stale_data")
    assert not (tmp_path / "paper").exists()


@pytest.mark.parametrize("start,end", [("2026-11-04", "2026-11-04"), ("2026-11-05", "2026-11-05")])
def test_backtest_rejects_uncompleted_today_and_future_sessions_without_publishing(
    tmp_path, monkeypatch, start, end
):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 4, 10, 0))
    response = client(repo).post("/api/t-strategy/backtest", json=config(start=start, end=end))
    assert response.status_code == 400, response.text
    assert not list((tmp_path / "research" / "t_strategy").glob("*/result.json"))
    assert paper.list_account_ids(tmp_path) == []


def test_backtest_does_not_degrade_current_partial_session_to_a_valid_replay(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = setup_repo(tmp_path, [minutes(D1).head(30)])
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 4, 10, 0, 20))
    response = client(repo).post(
        "/api/t-strategy/backtest", json=config(start=str(D1), end=str(D1))
    )
    assert response.status_code == 400
    assert not list((tmp_path / "research" / "t_strategy").glob("*/result.json"))
    assert paper.list_account_ids(tmp_path) == []


@pytest.mark.parametrize(
    "initial,cutoff,delayed,expected",
    [
        (
            datetime(2026, 11, 5, 11, 29, 59),
            datetime(2026, 11, 5, 11, 29),
            datetime(2026, 11, 5, 11, 30, 2),
            "outside_session",
        ),
        (
            datetime(2026, 11, 5, 10, 0, 59),
            datetime(2026, 11, 5, 10, 0),
            datetime(2026, 11, 5, 10, 2, 1),
            "stale_data",
        ),
    ],
)
def test_manual_rechecks_session_and_freshness_after_lock_wait(
    tmp_path, monkeypatch, initial, cutoff, delayed, expected
):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    clock = {"now": initial}
    monkeypatch.setattr(t_strategy, "cn_now", lambda: clock["now"])
    original_read = repo.get_minute_range
    monkeypatch.setattr(
        repo,
        "get_minute_range",
        lambda *a, **kw: original_read(*a, **kw).filter(pl.col("datetime") <= cutoff),
    )
    original_lock = paper.PAPER_LOCK

    class DelayedLock:
        def __enter__(self):
            original_lock.acquire()
            clock["now"] = delayed
            return self

        def __exit__(self, *args):
            original_lock.release()

    monkeypatch.setattr(paper, "PAPER_LOCK", DelayedLock())
    result = t_strategy.TStrategyService(repo).manual_step(t_strategy.ManualRequest(**config()))
    assert result["status"] == expected
    assert not (tmp_path / "paper").exists()


def test_backtest_rejects_requested_future_end_even_without_future_rows(tmp_path, monkeypatch):
    from app.services import t_strategy

    repo = setup_repo(tmp_path)
    monkeypatch.setattr(t_strategy, "cn_now", lambda: datetime(2026, 11, 6, 10, 0))
    response = client(repo).post("/api/t-strategy/backtest", json=config(end="2026-11-07"))
    assert response.status_code == 400
    assert not list((tmp_path / "research" / "t_strategy").glob("*/result.json"))
    assert paper.list_account_ids(tmp_path) == []
