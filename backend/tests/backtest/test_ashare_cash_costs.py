from datetime import date

import polars as pl
import pytest

from app.backtest.engine import BacktestEngine, MatcherConfig
from app.strategy import paper
from app.tickflow.repository import DataStore, KlineRepository

METHODS = ["simulate_independent_candidates", "simulate_independent_candidates_legacy",
           "simulate_portfolio", "simulate_portfolio_legacy"]


def _panel():
    return pl.DataFrame({
        "symbol": ["600519.SH"] * 2, "name": ["样本"] * 2,
        "date": [date(2023, 8, 25), date(2023, 8, 28)],
        "open": [10.0] * 2, "high": [10.0] * 2, "low": [10.0] * 2,
        "close": [10.0] * 2, "volume": [100_000.0] * 2,
        "raw_close": [10.0] * 2,
    })


@pytest.mark.parametrize("method", METHODS)
def test_matchers_use_cash_fees_and_sell_execution_date(method):
    config = MatcherConfig(fee_model="ashare_v1", commission_pct=0.00025,
                           slippage_bps=0, max_hold_days=1, max_positions=1,
                           initial_capital=1005.01)
    result = getattr(BacktestEngine(repo=None), method)(
        _panel(), pl.Series([True, False]), pl.Series([False, True]), config)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.shares == 100
    assert trade.entry_value == 1005.01
    assert trade.exit_value == 994.49
    assert trade.pnl_amount == -10.52


@pytest.mark.parametrize("method", ["simulate_portfolio", "simulate_portfolio_legacy"])
def test_backtest_does_not_spend_minimum_commission_reserve(method):
    result = getattr(BacktestEngine(repo=None), method)(
        _panel(), pl.Series([True, False]), pl.Series([False, True]),
        MatcherConfig(fee_model="ashare_v1", slippage_bps=0, initial_capital=1005,
                      max_positions=1))
    assert not result.trades


def test_paper_and_backtest_use_same_cash_ledger(tmp_path, monkeypatch):
    panel = _panel()
    KlineRepository(DataStore(tmp_path)).append_daily(panel)
    acc = paper.create_account(tmp_path, 1005.01, slippage_bps=0)
    assert acc["fee_model"] == "ashare_v1"
    assert acc["stamp_tax_pct"] is None
    monkeypatch.setattr(paper, "_now_iso", lambda: "2023-08-25T10:00:00+08:00")
    paper.create_order(tmp_path, "600519.SH", "buy", qty=100, order_type="close")
    paper.settle_day(tmp_path, "2023-08-25")
    assert paper.get_account(tmp_path)["cash"] == 0
    monkeypatch.setattr(paper, "_now_iso", lambda: "2023-08-28T10:00:00+08:00")
    paper.create_order(tmp_path, "600519.SH", "sell", qty=100, order_type="close")
    paper.settle_day(tmp_path, "2023-08-28")
    assert paper.get_account(tmp_path)["cash"] == 994.49


def test_old_paper_account_keeps_configured_fees(tmp_path, monkeypatch):
    acc = paper.create_account(tmp_path, 2000, stamp_tax_pct=0.001, slippage_bps=0)
    acc.pop("fee_model", None)
    paper.save_account(tmp_path, acc)
    KlineRepository(DataStore(tmp_path)).append_daily(_panel())
    monkeypatch.setattr(paper, "_now_iso", lambda: "2023-08-25T10:00:00+08:00")
    paper.create_order(tmp_path, "600519.SH", "buy", qty=100, order_type="close")
    paper.settle_day(tmp_path, "2023-08-25")
    assert paper.get_account(tmp_path)["cash"] == 995


def test_unknown_fee_model_is_rejected():
    with pytest.raises(ValueError, match="fee_model"):
        MatcherConfig(fee_model="unknown")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("basis", ["adjusted", "missing"])
def test_cash_fees_reject_unverified_or_adjusted_share_prices(method, basis):
    panel = _panel()
    panel = panel.with_columns(pl.lit(20.0).alias("raw_close")) if basis == "adjusted" else panel.drop("raw_close")
    with pytest.raises(ValueError, match="原始价"):
        getattr(BacktestEngine(repo=None), method)(
            panel, pl.Series([True, False]), pl.Series([False, True]),
            MatcherConfig(fee_model="ashare_v1", slippage_bps=0))


@pytest.mark.parametrize("method", METHODS)
def test_cash_fees_reject_partial_missing_basis_on_fillable_open(method):
    panel = pl.DataFrame({
        "symbol": ["600519.SH"] * 3,
        "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        "open": [10.0, 5.0, 5.0], "high": [10.0, 5.0, 5.0], "low": [10.0, 5.0, 5.0],
        "close": [10.0, None, 5.0], "raw_close": [10.0, None, 5.0], "volume": [100000.0] * 3,
    })
    with pytest.raises(ValueError, match="原始价"):
        getattr(BacktestEngine(repo=None), method)(
            panel, pl.Series([True, False, False]), pl.Series([False, True, False]),
            MatcherConfig(fee_model="ashare_v1", matching="open_t+1", slippage_bps=0,
                          initial_capital=1005.01, max_positions=1))
