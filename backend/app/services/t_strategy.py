"""Explicit, local-data-only base-position T replay and manual simulation.

Signals use completed minute closes. Execution uses a subsequent minute's open,
whose timestamp is its end label minus one minute. Paper remains the ledger.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import threading
import time as monotonic_time
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market_time import CN_TZ, cn_now
from app.services.fs_utils import atomic_write_text
from app.services.minute_adjust import minute_basis_is_raw
from app.strategy import corporate_actions as corporate
from app.strategy import paper
from app.trading_costs import SUPPORTED_FROM

_RUN_LOCK = threading.Lock()
MAX_RUNS = 20
MINUTE_TIMES = tuple([time(m // 60, m % 60) for m in (*range(571, 691), *range(781, 901))])


class TConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    symbol: str = Field(default="000001.SZ", pattern=r"^\d{6}\.(SH|SZ|BJ)$")
    start: date = Field(default_factory=lambda: cn_now().date() - timedelta(days=30))
    end: date = Field(default_factory=lambda: cn_now().date())
    initial_cash: float = Field(default=100000, gt=0, le=1_000_000_000, strict=True)
    base_qty: int = Field(default=600, gt=0, le=1_000_000, multiple_of=100, strict=True)
    trade_qty: int = Field(default=300, gt=0, le=1_000_000, multiple_of=100, strict=True)
    rise_pct: float = Field(default=1.0, gt=0, le=30, strict=True)
    pullback_pct: float = Field(default=1.0, gt=0, le=30, strict=True)
    start_time: str = "09:45"
    restore_time: str = "14:55"
    commission_pct: float = Field(default=0.00025, ge=0, le=0.01, strict=True)
    slippage_bps: float = Field(default=5, ge=0, le=100, strict=True)

    @model_validator(mode="after")
    def bounds(self):
        if self.trade_qty > self.base_qty:
            raise ValueError("做T数量不能大于底仓")
        if (
            self.start < SUPPORTED_FROM
            or self.end < self.start
            or (self.end - self.start).days > 90
        ):
            raise ValueError("回放窗口须从2022-04-29起且不超过90个自然日")
        for value in (self.start_time, self.restore_time):
            try:
                parsed = time.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("时间须为HH:MM交易时段") from exc
            if value != parsed.strftime("%H:%M") or parsed not in MINUTE_TIMES:
                raise ValueError("时间须为HH:MM交易时段")
        if self.start_time >= self.restore_time or self.restore_time > "14:59":
            raise ValueError("开始时间须早于恢复时间，恢复时间最晚14:59")
        return self


class ManualRequest(TConfig):
    account: str = "sim_t"

    @model_validator(mode="before")
    @classmethod
    def ignore_historical_dates(cls, values):
        if isinstance(values, dict):
            return {key: value for key, value in values.items() if key not in ("start", "end")}
        return values

    @model_validator(mode="after")
    def account_id(self):
        paper.validate_account_id(self.account)
        return self


def decision(
    config: TConfig, state: dict, bar: dict, reference: float, qty: int, available: int
) -> dict | None:
    """Shared replay/manual decision; only filled sells create restoration debt."""
    stamp = bar["datetime"]
    day, hhmm = stamp.date().isoformat(), stamp.strftime("%H:%M")
    if state.get("obligation", 0):
        if (
            bar["close"] <= state["sell_price"] * (1 - config.pullback_pct / 100)
            or hhmm >= config.restore_time
        ):
            return {"side": "buy", "qty": state["obligation"], "purpose": "restore"}
        return None
    if not state.get("initialized"):
        if qty == 0:
            return {"side": "buy", "qty": config.base_qty, "purpose": "base"}
        state["initialized"] = True
    if (
        state.get("cycle_day") != day
        and config.start_time <= hhmm < config.restore_time
        and qty >= config.base_qty
        and available >= config.trade_qty
        and bar["close"] >= reference * (1 + config.rise_pct / 100)
    ):
        return {"side": "sell", "qty": config.trade_qty, "purpose": "sell"}
    return None


def _filled_state(state: dict, fill: dict, purpose: str) -> None:
    if purpose == "base":
        state["initialized"] = True
    elif purpose == "sell":
        state.update(obligation=fill["qty"], sell_price=fill["price"])
    elif purpose == "restore":
        state.update(obligation=0, cycle_day=fill["date"], cycles=state.get("cycles", 0) + 1)


def _corporate_state(state: dict, event: dict, symbol: str) -> None:
    """An un-restored share obligation changes units on a bonus ex-date too."""
    if event.get("symbol") != symbol:
        return
    obligation = state.get("obligation", 0)
    multiplier = 1 + event["bonus"]
    if obligation and multiplier != 1:
        adjusted = obligation * multiplier
        if (not math.isfinite(adjusted)
                or abs(adjusted - round(adjusted)) > 1e-7 or round(adjusted) % 100):
            raise ValueError("送股后恢复底仓义务不是整手，请人工处理后再评估")
        state["obligation"] = round(adjusted)


def _write_json(path: Path, data: dict) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2))


def _naive_now() -> datetime:
    now = cn_now()
    return now.astimezone(CN_TZ).replace(tzinfo=None) if now.tzinfo else now


def _manual_session(now: datetime) -> bool:
    return now.weekday() < 5 and (
        time(9, 31) <= now.time() < time(11, 30) or time(13, 1) <= now.time() < time(15, 0)
    )


def _manual_gate(now: datetime, stamp: datetime) -> dict | None:
    if not _manual_session(now):
        return {"status": "outside_session", "message": "交易时段已结束，请下次交易时段重新评估"}
    if stamp.date() != now.date() or not 0 <= (now - stamp).total_seconds() <= 120:
        return {"status": "stale_data", "message": "等待期间本地分钟数据已过期，请同步后重新评估"}
    return None


def _validate_bars(
    frame: pl.DataFrame,
    days: list[date],
    *,
    completed: datetime | None = None,
    allow_partial: bool = False,
) -> list[dict]:
    if frame.is_empty():
        raise ValueError("没有本地分钟数据，请先同步所选股票和日期")
    required = {"symbol", "datetime", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError("分钟数据字段不完整，请重新同步")
    rows = frame.sort("datetime").to_dicts()
    actual: dict[date, list[datetime]] = {}
    for row in rows:
        stamp = row["datetime"]
        if not isinstance(stamp, datetime) or stamp.tzinfo is not None:
            raise ValueError("分钟数据必须是北京墙钟且标记已完成分钟结束时间")
        if completed is not None and stamp > completed:
            raise ValueError("本地分钟数据包含尚未完成或未来的分钟，请重新同步")
        actual.setdefault(stamp.date(), []).append(stamp)
        for field in ("open", "high", "low", "close", "volume"):
            value = row[field]
            if (
                value is None
                or not math.isfinite(value)
                or (value <= 0 if field != "volume" else value < 0)
            ):
                raise ValueError("分钟价格或成交量无效，请重新同步")
        if row["high"] < max(row["open"], row["close"]) or row["low"] > min(
            row["open"], row["close"]
        ):
            raise ValueError("分钟OHLC关系无效，请重新同步")
    if set(actual) != set(days):
        raise ValueError("分钟数据未覆盖日线交易日期，请补齐数据")
    for day in days:
        expected = [datetime.combine(day, t) for t in MINUTE_TIMES]
        if completed is not None and allow_partial:
            expected = [stamp for stamp in expected if stamp <= completed]
        if actual[day] != expected:
            raise ValueError(f"{day}分钟数据缺失、重复或不符合240根完整交易时段，请重新同步")
    return rows


class TStrategyService:
    def __init__(self, repo):
        self.repo = repo
        self.data_dir = Path(repo.store.data_dir)
        self.root = self.data_dir / "research" / "t_strategy"

    def _run_path(self, run_id: str) -> Path:
        try:
            if str(uuid.UUID(run_id)) != run_id:
                raise ValueError
        except (ValueError, AttributeError) as exc:
            raise ValueError("回放记录ID非法") from exc
        return self.root / run_id

    def get_run(self, run_id: str) -> dict:
        path = self._run_path(run_id) / "result.json"
        if not path.is_file():
            raise FileNotFoundError("回放记录不存在")
        return json.loads(path.read_text())

    def _market(self, config: TConfig, *, manual_day: date | None = None):
        if self.repo.resolve_asset_type(config.symbol) != "stock" or config.symbol[:2] in (
            "51",
            "56",
            "58",
            "15",
        ):
            raise ValueError("底仓做T仅支持股票，不支持ETF或指数")
        if not minute_basis_is_raw(self.data_dir):
            raise ValueError("分钟数据缺少未复权原始价基准，请先完成原始基准迁移")
        start, end = (manual_day, manual_day) if manual_day else (config.start, config.end)
        daily = self.repo.execute_all(
            "SELECT symbol,date,open,high,low,close,volume,amount FROM kline_daily "
            "WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
            [config.symbol, start - timedelta(days=90), end],
        )
        if not daily:
            raise ValueError("没有本地原始日线，请先同步股票日线数据")
        daily_rows = [
            dict(
                zip(
                    ("symbol", "date", "open", "high", "low", "close", "volume", "amount"),
                    row,
                    strict=True,
                )
            )
            for row in daily
        ]
        if len({row["date"] for row in daily_rows}) != len(daily_rows):
            raise ValueError("本地日线重复，请先修复日线数据")
        for row in daily_rows:
            if any(
                row[key] is None or not math.isfinite(row[key]) or row[key] <= 0
                for key in ("open", "close")
            ):
                raise ValueError("本地原始日线价格无效")
        # Validate economic details, not price factors, before publishing a run.
        for event in self._corporate_rows(config.symbol, end):
            if start <= event["trade_date"] <= end:
                corporate.normalize(event)
        bars = self.repo.get_minute_range([config.symbol], start, end, raw=True)
        return daily_rows, bars

    def _corporate_rows(self, symbol: str, through: date) -> list[dict]:
        if not any((self.data_dir / "adj_factor").glob("**/*.parquet")):
            return []
        available = {row[0] for row in self.repo.execute_all("DESCRIBE adj_factor")}
        columns = [name for name in (
            "symbol", "trade_date", "ex_factor", "dividend", "bonus", "allot",
            "allot_price", "prev_close", "pay_date", "listing_date",
        ) if name in available]
        if not {"symbol", "trade_date"} <= set(columns):
            raise ValueError("公司行动缺少股票或事件日期明细")
        rows = self.repo.execute_all(
            f"SELECT {','.join(columns)} FROM adj_factor WHERE symbol=? AND trade_date<=? ORDER BY trade_date",
            [symbol, through],
        )
        return [dict(zip(columns, row, strict=True)) for row in rows]

    def _corporate_timeline(
        self, symbol: str, since: date, through: date, *, ledger=(), persisted=(),
    ) -> list[dict]:
        """Economic share units, independent of the account's entitlement quantity.

        Persist observed source details so a no-position ex-date remains known
        after a restart or source retention. Entitlement records supply the same
        evidence when present; neither source creates account cash or shares.
        """
        events: dict[str, dict] = {}

        def merge(row):
            if row.get("symbol") != symbol:
                return
            day = corporate.iso_day(row.get("trade_date"), "trade_date")
            if not since.isoformat() <= day <= through.isoformat():
                return
            event = corporate.normalize({**row, "trade_date": day})
            if event is None:
                return
            canonical = {key: event[key] for key in (
                "symbol", "trade_date", "dividend", "bonus", "pay_date", "listing_date",
            )}
            canonical["allot"] = 0.0
            existing = events.get(day)
            if existing:
                if any(existing[key] != canonical[key] for key in ("dividend", "bonus")):
                    raise ValueError("做T公司行动已观察明细冲突，请人工复核")
                for key in ("pay_date", "listing_date"):
                    if existing[key] and canonical[key] and existing[key] != canonical[key]:
                        raise ValueError("做T公司行动已观察日期冲突，请人工复核")
                    canonical[key] = existing[key] or canonical[key]
            events[day] = canonical

        for row in persisted:
            merge(row)
        for event in ledger:
            if (event.get("symbol") == symbol and event.get("kind") == "corp_action"
                    and event.get("schema_version") == 2):
                merge({
                    "symbol": symbol, "trade_date": event["date"],
                    "dividend": event.get("dividend_per_share"),
                    "bonus": event.get("bonus_per_share"), "allot": 0.0,
                    "pay_date": event.get("pay_date"),
                    "listing_date": event.get("available_date"),
                })
        seen = set()
        for row in self._corporate_rows(symbol, through):
            if row.get("symbol") != symbol:
                continue
            day = corporate.iso_day(row["trade_date"], "trade_date")
            if day in seen:
                raise ValueError("公司行动重复事件，请先核对数据")
            seen.add(day)
            merge(row)
        return [events[day] for day in sorted(events)]

    @staticmethod
    def _references(daily: list[dict], days: list[date]) -> dict[date, float]:
        references = {}
        for day in days:
            prior = [row for row in daily if row["date"] < day]
            if not prior:
                raise ValueError(f"{day}缺少前一交易日原始收盘价，请同步更早日线")
            references[day] = float(prior[-1]["close"])
        return references

    @staticmethod
    def _position(root: Path, account: str, symbol: str, stamp: datetime) -> tuple[int, int]:
        position = paper.load_positions(root, account).get(symbol, {})
        capacity = paper.sell_capacity(root, symbol, account, as_of=stamp)
        return int(position.get("qty", 0)), max(0, capacity[0] - capacity[1]) if capacity else 0

    @staticmethod
    def _order(root: Path, account: str, config: TConfig, signal: dict, bar: dict, stamp: datetime):
        order, error = paper.create_order(
            root,
            config.symbol,
            signal["side"],
            account_id=account,
            qty=signal["qty"],
            order_type="t_signal",
            asset_type="stock",
            ref_price=bar["close"],
            source="t_strategy",
            as_of=stamp,
        )
        if order:
            order.update(t_purpose=signal["purpose"], t_signal_at=bar["datetime"].isoformat())
            paper.save_order(root, order, account)
        return order, error

    def backtest(self, config: TConfig) -> dict:
        if not _RUN_LOCK.acquire(blocking=False):
            raise ValueError("已有回放任务运行中，请稍后重试")
        work = None
        try:
            if config.end > _naive_now().date():
                raise ValueError("回放结束日期不能晚于当前北京时间日期")
            daily, frame = self._market(config)
            days = [row["date"] for row in daily if config.start <= row["date"] <= config.end]
            rows = _validate_bars(frame, days, completed=_naive_now())
            self._references(daily, days)  # Validate pre-window raw history before publishing.
            self.root.mkdir(parents=True, exist_ok=True)
            if len(list(self.root.glob("*/result.json"))) >= MAX_RUNS:
                raise ValueError("最多保留20次回放，请先归档或清理已有研究记录")
            run_id = str(uuid.uuid4())
            work = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
            raw_dir = work / "kline_daily"
            raw_dir.mkdir()
            pl.DataFrame(daily).write_parquet(raw_dir / "selected.parquet")
            actions = self._corporate_rows(config.symbol, config.end)
            unit_events = iter(self._corporate_timeline(config.symbol, config.start, config.end))
            next_unit_event = next(unit_events, None)
            if actions:
                (work / "adj_factor").mkdir()
                pl.DataFrame(actions).write_parquet(work / "adj_factor" / "all.parquet")
            paper.create_account(
                work,
                config.initial_cash,
                commission_pct=config.commission_pct,
                slippage_bps=config.slippage_bps,
                name="做T历史回放",
            )
            paper.create_account(
                work, config.initial_cash, account_id="benchmark",
                commission_pct=config.commission_pct, slippage_bps=config.slippage_bps,
                name="同底仓买入持有基线",
            )
            state, events, nav = {}, [], []
            pending = None
            current_day = None
            reference = None
            started = monotonic_time.monotonic()
            for index, bar in enumerate(rows):
                if monotonic_time.monotonic() - started > 120:
                    raise ValueError("回放超过120秒预算，请缩短日期范围")
                stamp, day = bar["datetime"], bar["datetime"].date()
                with paper.PAPER_LOCK:
                    if current_day != day:
                        for account in ("default", "benchmark"):
                            paper.apply_corporate_actions(
                                work, str(day), account,
                                as_of=datetime.combine(day, time(9)),
                            )
                        while next_unit_event and next_unit_event["trade_date"] <= str(day):
                            _corporate_state(state, next_unit_event, config.symbol)
                            next_unit_event = next(unit_events, None)
                        reference = paper.ex_reference_close(
                            work, config.symbol, "stock", str(day),
                        )
                        if reference is None:
                            raise ValueError(f"{day}缺少除权参考价，无法安全撮合")
                        current_day = day
                    if pending:
                        signal_stamp = datetime.fromisoformat(pending["t_signal_at"])
                        if stamp - signal_stamp == timedelta(minutes=1):
                            fill = paper._fill_order(
                                work,
                                pending,
                                bar["open"],
                                str(day),
                                as_of=stamp - timedelta(minutes=1),
                            )
                            if fill:
                                _filled_state(state, fill, pending["t_purpose"])
                                if pending["t_purpose"] == "base":
                                    benchmark_order, error = paper.create_order(
                                        work, config.symbol, "buy", account_id="benchmark",
                                        qty=fill["qty"], order_type="t_signal", asset_type="stock",
                                        ref_price=bar["open"], as_of=stamp - timedelta(minutes=1),
                                    )
                                    if error or not paper._fill_order(
                                        work, benchmark_order, bar["open"], str(day),
                                        "benchmark", as_of=stamp - timedelta(minutes=1),
                                    ):
                                        raise ValueError("同底仓基线建仓失败，禁止输出错误超额收益")
                            else:
                                events.append(
                                    {
                                        "date": str(day),
                                        "status": "rejected",
                                        "message": pending.get("reason"),
                                        "order_id": pending["id"],
                                    }
                                )
                        else:
                            paper._expire(work, pending, "下一分钟不相邻，禁止跨午休或隔夜补成交")
                            events.append(
                                {
                                    "date": str(day),
                                    "status": "expired",
                                    "message": pending["reason"],
                                }
                            )
                        pending = None
                    qty, available = self._position(work, "default", config.symbol, stamp)
                    signal = decision(config, state, bar, reference, qty, available)
                    if signal:
                        pending, error = self._order(work, "default", config, signal, bar, stamp)
                        if error:
                            events.append(
                                {"date": str(day), "status": "rejected", "message": error}
                            )
                    if index == len(rows) - 1 or rows[index + 1]["datetime"].date() != day:
                        point = paper.daily_nav(work, str(day), {config.symbol: bar["close"]})
                        paper._write_nav_line(work, str(day), point)
                        nav.append(point)
            if pending:
                paper._expire(work, pending, "窗口结束，无后续相邻分钟，未成交")
                events.append(
                    {
                        "date": str(rows[-1]["datetime"].date()),
                        "status": "expired",
                        "message": pending["reason"],
                    }
                )
            warnings = ["分钟撮合仅校验已知涨跌停与资金/T+1，不模拟Level2排队或部分成交。"]
            if state.get("obligation"):
                warnings.append(f"仍有{state['obligation']}股恢复底仓义务，未虚构成交")
            ledger = paper.load_fills(work)
            fills = [fill for fill in ledger if fill.get("kind", "fill") == "fill"]
            action_records = [fill for fill in ledger if fill.get("schema_version") == 2]
            for event in action_records:
                events.append({**event, "message": (
                    f"公司行动 {event['kind']} · 应收变动 {event.get('cash_receivable', 0):.2f}元"
                    f" · 送股 {event.get('bonus_qty', 0)}股"
                )})
            if action_records:
                warnings += ["分红按税前应收计入净资产；未提供到账日时不计入可用现金，个人红利税未模拟。",
                             "送股没有确认上市日期时锁定；回合FIFO统计未分摊公司行动，绩效以含权益净值为准。"]
            final_nav = nav[-1]["nav"]
            benchmark_nav = paper.daily_nav(
                work, str(days[-1]), {config.symbol: rows[-1]["close"]}, "benchmark",
            )["nav"]
            result = {
                "id": run_id,
                "params": config.model_dump(mode="json"),
                "stats": {
                    "initial_cash": config.initial_cash,
                    "final_nav": final_nav,
                    "return_pct": (final_nav / config.initial_cash - 1) * 100,
                    "benchmark_nav": benchmark_nav,
                    "excess_pnl": round(final_nav - benchmark_nav, 2),
                    "round_trips": state.get("cycles", 0),
                    "fills": len(fills),
                    "fees": round(sum(fill.get("fee", 0) for fill in fills), 2),
                    "receivable": paper.receivable_balance(work),
                },
                "nav": nav,
                "fills": fills,
                "events": events,
                "warnings": warnings,
                "coverage": {
                    "start": str(days[0]),
                    "end": str(days[-1]),
                    "sessions": len(days),
                    "bars": len(rows),
                },
            }
            _write_json(work / "result.json", result)
            work.rename(self._run_path(run_id))
            work = None
            return result
        finally:
            if work:
                shutil.rmtree(work)
            _RUN_LOCK.release()

    def import_paper(self, run_id: str) -> dict:
        self.get_run(run_id)
        account_id = "t_" + uuid.UUID(run_id).hex[:28]
        with paper.PAPER_LOCK:
            existing = paper.get_account(self.data_dir, account_id)
            if existing:
                if existing.get("t_strategy_run") != run_id or existing.get("status") != "frozen":
                    raise ValueError("目标历史账户已存在且不匹配，拒绝覆盖")
                return {"account": account_id, "frozen": True}
            source = self._run_path(run_id) / "paper" / "accounts" / "default"
            base = paper.accounts_base(self.data_dir)
            base.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".import-", dir=base))
            try:
                shutil.copytree(source, stage, dirs_exist_ok=True)
                account = json.loads((stage / "account.json").read_text())
                account.update(
                    id=account_id,
                    name=f"做T历史 {run_id[:8]}",
                    status="frozen",
                    t_strategy_run=run_id,
                )
                _write_json(stage / "account.json", account)
                # get_account creates an empty directory when probing. Publish the staged ledger atomically.
                dest = base / account_id
                if dest.exists() and not any(dest.iterdir()):
                    dest.rmdir()
                stage.rename(dest)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            return {"account": account_id, "frozen": True}

    def status(self) -> dict:
        runs = []
        if self.root.exists():
            for path in sorted(
                self.root.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )[:MAX_RUNS]:
                if path.parent.name.startswith("."):
                    continue
                try:
                    run = self.get_run(path.parent.name)
                    runs.append({key: run[key] for key in ("id", "params", "stats", "coverage")})
                except (ValueError, OSError):
                    continue
        coverage = {"start": None, "end": None, "sessions": 0, "bars": 0}
        defaults = TConfig()
        if minute_basis_is_raw(self.data_dir):
            now = _naive_now()
            frame = self.repo.get_minute_range(
                [defaults.symbol], now.date() - timedelta(days=89), now.date(), raw=True
            )
            if not frame.is_empty():
                sessions: dict[date, list[datetime]] = {}
                for stamp in frame["datetime"].to_list():
                    if isinstance(stamp, datetime) and stamp.tzinfo is None:
                        sessions.setdefault(stamp.date(), []).append(stamp)
                # Coverage is a timestamp/session summary. Replay still validates
                # OHLC, daily references, corporate data and all requested dates.
                complete = sorted(
                    day
                    for day, stamps in sessions.items()
                    if sorted(stamps) == [datetime.combine(day, t) for t in MINUTE_TIMES]
                    and stamps
                    and max(stamps) <= now
                )
                if complete:
                    end = complete[-1]
                    start = next(
                        (day for day in complete if day >= end - timedelta(days=30)), complete[0]
                    )
                    defaults = defaults.model_copy(update={"start": start, "end": end})
                    coverage = {
                        "start": str(complete[0]),
                        "end": str(end),
                        "sessions": len(complete),
                        "bars": len(complete) * 240,
                    }
        return {
            "defaults": defaults.model_dump(mode="json"),
            "coverage": coverage,
            "runs": runs,
            "accounts": paper.list_accounts(self.data_dir),
        }

    def manual_step(self, request: ManualRequest) -> dict:
        now = _naive_now()
        account_id = request.account
        response = {"account": account_id, "filled": []}
        if not _manual_session(now):
            return {
                **response,
                "status": "outside_session",
                "message": "请在交易日连续竞价时段点击评估",
            }
        read_started = monotonic_time.monotonic()
        _, frame = self._market(request, manual_day=now.date())
        if monotonic_time.monotonic() - read_started > 5:
            return {
                **response,
                "status": "data_timeout",
                "message": "读取本地数据超过5秒，请稍后重试",
            }
        # Future bars indicate a historical/replayed file and must never drive a current order.
        if (
            frame.is_empty()
            or frame["datetime"].max() > now
            or (now - frame["datetime"].max()).total_seconds() > 120
        ):
            return {
                **response,
                "status": "stale_data",
                "message": "缺少当前已完成分钟，请先同步最新本地分钟数据",
            }
        rows = _validate_bars(
            frame, [now.date()], completed=frame["datetime"].max(), allow_partial=True
        )
        reference = paper.ex_reference_close(self.data_dir, request.symbol, "stock", str(now.date()))
        if reference is None:
            raise ValueError("缺少前日原价/除权参考价，请同步日线与公司行动")
        bar, stamp = rows[-1], rows[-1]["datetime"]
        with paper.PAPER_LOCK:
            # Local reads and lock contention can cross a minute/session boundary.
            # Revalidate with the actual lock-owner clock before touching accounts.
            now = _naive_now()
            gate = _manual_gate(now, stamp)
            if gate:
                return {**response, **gate}
            account = paper.get_account(self.data_dir, account_id)
            if account is None:
                if account_id != "sim_t":
                    raise ValueError("模拟账户不存在，请选择已有账户")
                account = paper.create_account(
                    self.data_dir,
                    request.initial_cash,
                    account_id=account_id,
                    commission_pct=request.commission_pct,
                    slippage_bps=request.slippage_bps,
                    name="手动做T模拟",
                )
            if account.get("status") != "active":
                raise ValueError("账户已冻结，请选择活动模拟账户")
            if account.get("fee_model") != "ashare_v1":
                raise ValueError("请选择使用ashare_v1金额费用模型的模拟账户")
            state_path = paper.accounts_base(self.data_dir) / account_id / "t_strategy.json"
            params = request.model_dump(
                mode="json",
                exclude={
                    "account",
                    "start",
                    "end",
                    "initial_cash",
                    "commission_pct",
                    "slippage_bps",
                },
            )
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get("params") and state["params"] != params:
                raise ValueError("此账户已绑定其他做T参数，请使用独立账户")
            orders = paper.load_orders(self.data_dir, account_id)
            if any(
                order["status"] == "pending" and order.get("source") != "t_strategy"
                for order in orders
            ):
                raise ValueError("账户存在其他来源挂单，请先处理后再评估")
            own_orders = {
                order["id"]: order for order in orders if order.get("source") == "t_strategy"
            }
            if any(order["symbol"] != request.symbol for order in own_orders.values()):
                raise ValueError("此账户已有其他股票的做T记录，请使用独立账户")
            paper.apply_corporate_actions(
                self.data_dir, str(now.date()), account_id, as_of=now,
            )
            # The append-only ledger is authoritative even if a process stopped before
            # saving its strategy snapshot. Never lose an outstanding buyback debt.
            for key in ("initialized", "obligation", "sell_price", "cycle_day", "cycles"):
                state.pop(key, None)
            ledger = paper.load_fills(self.data_dir, account_id)
            strategy_fills = [fill for fill in ledger if fill.get("order_id") in own_orders
                              and fill.get("kind") == "fill" and fill["date"] <= str(now.date())]
            since = min((date.fromisoformat(fill["date"]) for fill in strategy_fills),
                        default=now.date())
            unit_events = self._corporate_timeline(
                request.symbol, since, now.date(), ledger=ledger,
                persisted=state.get("corporate_events", []),
            )
            state["corporate_events"] = unit_events
            # Ex-date unit changes precede every same-day fill, even if paper
            # learned/recorded an entitlement after one of those fills.
            timeline = [(event["trade_date"], 0, index, event)
                        for index, event in enumerate(unit_events)]
            timeline += [(fill["date"], 1, index, fill)
                         for index, fill in enumerate(strategy_fills)]
            for _, kind, _, fill in sorted(timeline):
                if kind == 0:
                    _corporate_state(state, fill, request.symbol)
                    continue
                order = own_orders.get(fill.get("order_id"))
                if order and fill.get("kind") == "fill":
                    if order["status"] == "pending":
                        raise ValueError("成交账本与订单状态不一致，请先重建核对账户，避免重复成交")
                    if order.get("t_purpose") not in ("base", "sell", "restore"):
                        raise ValueError("做T订单元数据缺失，请先核对账户")
                    _filled_state(state, fill, order["t_purpose"])
            if state.get("last_bar", "") >= stamp.isoformat():
                # Deduplicate orders, not newly observed corporate evidence.
                _write_json(state_path, state)
                return {
                    **response,
                    "status": "already_evaluated",
                    "message": "本分钟已评估，不重复下单",
                }
            pending = [order for order in own_orders.values() if order["status"] == "pending"]
            if len(pending) > 1 or any(
                "t_signal_at" not in order or "t_purpose" not in order for order in pending
            ):
                raise ValueError("做T挂单数量或元数据异常，请先核对账户")
            for order in pending:
                signal_stamp = datetime.fromisoformat(order["t_signal_at"])
                created = datetime.fromisoformat(order["created_at"])
                created = (
                    created.astimezone(CN_TZ).replace(tzinfo=None) if created.tzinfo else created
                )
                candidates = [
                    row
                    for row in rows
                    if row["datetime"] > signal_stamp
                    and row["datetime"] - timedelta(minutes=1) >= created
                ]
                if signal_stamp.date() != stamp.date() or (
                    signal_stamp.time() <= time(11, 30) and stamp.time() >= time(13, 1)
                ):
                    paper._expire(
                        self.data_dir, order, "信号已跨午休或交易日，禁止补成交", account_id
                    )
                elif candidates:
                    candidate = candidates[0]
                    fill = paper._fill_order(
                        self.data_dir,
                        order,
                        candidate["open"],
                        str(now.date()),
                        account_id,
                        as_of=candidate["datetime"] - timedelta(minutes=1),
                    )
                    if fill:
                        response["filled"].append(fill)
                        _filled_state(state, fill, order["t_purpose"])
                else:
                    state.update(params=params, last_bar=stamp.isoformat())
                    _write_json(state_path, state)
                    return {
                        **response,
                        "status": "pending",
                        "message": "等待下单以后开始的已完成分钟，需再次点击评估",
                        "order": order,
                    }
                if order["status"] != "filled":
                    state.update(params=params, last_bar=stamp.isoformat())
                    _write_json(state_path, state)
                    return {
                        **response,
                        "status": "rejected",
                        "message": order.get("reason") or "模拟订单未成交",
                        "order": order,
                    }
            qty, available = self._position(self.data_dir, account_id, request.symbol, stamp)
            signal = decision(request, state, bar, reference, qty, available)
            state.update(params=params, last_bar=stamp.isoformat())
            order = None
            error = None
            if signal:
                now = _naive_now()
                gate = _manual_gate(now, stamp)
                if gate:
                    return {**response, **gate}
                order, error = self._order(self.data_dir, account_id, request, signal, bar, now)
            _write_json(state_path, state)
            return {
                **response,
                "status": "order_created" if order else ("rejected" if error else "evaluated"),
                "message": error
                or ("已创建模拟信号单，等待未来分钟后再次点击撮合" if order else "已评估当前分钟"),
                **({"order": order} if order else {}),
            }
