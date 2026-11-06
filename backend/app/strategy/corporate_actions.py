"""Per-share corporate event contract; price factors are never share ratios."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import polars as pl


class UnsupportedCorporateActionError(ValueError):
    """The source cannot support an unambiguous cash/share entitlement."""


def finite_nonnegative(value: float, field: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise UnsupportedCorporateActionError(f"公司行动派生金额/数量必须有限且非负 finite: {field}")
    return value


def strict_json(value) -> str:
    """Validate complete candidate records/balances before any publication."""
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, OverflowError) as exc:
        raise UnsupportedCorporateActionError("公司行动包含非有限 finite 的金额或数量") from exc


def iso_day(value, field: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as exc:
        raise UnsupportedCorporateActionError(f"公司行动 {field} 日期无效: {value!r}") from exc


def rows_through(data_dir: Path, asset_type: str, symbols: set[str], day: str) -> list[dict]:
    path = data_dir / ("adj_factor_etf" if asset_type == "etf" else "adj_factor") / "all.parquet"
    if not path.exists() or not symbols:
        return []
    try:
        rows = pl.scan_parquet(path).filter(pl.col("symbol").is_in(sorted(symbols))).collect().to_dicts()
    except Exception as exc:
        raise UnsupportedCorporateActionError("公司行动数据无法读取") from exc
    out = []
    seen = set()
    for row in rows:
        ex_date = iso_day(row.get("trade_date"), "trade_date")
        if ex_date > day:
            continue
        key = row["symbol"], ex_date
        if key in seen:
            raise UnsupportedCorporateActionError(f"公司行动重复事件: {key}")
        seen.add(key)
        out.append({**row, "trade_date": ex_date, "asset_type": asset_type})
    return sorted(out, key=lambda row: (row["trade_date"], row["symbol"]))


def normalize(row: dict) -> dict | None:
    def number(field):
        value = row.get(field)
        if value is None or isinstance(value, bool):
            raise UnsupportedCorporateActionError(f"公司行动缺少每股明细 metadata: {field}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise UnsupportedCorporateActionError(f"公司行动明细无效: {field}") from exc
        if not math.isfinite(value) or value < 0:
            raise UnsupportedCorporateActionError(f"公司行动明细无效: {field}")
        return value

    # Pure identity rows do not represent an economic action.
    if row.get("ex_factor") == 1 and all(row.get(k) in (None, 0) for k in ("dividend", "bonus", "allot")):
        return None
    dividend, bonus, allot = (number(k) for k in ("dividend", "bonus", "allot"))
    if allot > 0:
        raise UnsupportedCorporateActionError("公司行动配股 rights issue 暂不支持")
    if not dividend and not bonus:
        if row.get("ex_factor") not in (None, 1):
            raise UnsupportedCorporateActionError("公司行动价格因子与每股明细不一致")
        return None
    ex_date = iso_day(row["trade_date"], "trade_date")
    out = {**row, "trade_date": ex_date, "dividend": dividend, "bonus": bonus}
    for field in ("pay_date", "listing_date"):
        value = row.get(field)
        out[field] = iso_day(value, field) if value is not None else None
        if out[field] is not None and out[field] < ex_date:
            raise UnsupportedCorporateActionError(f"公司行动 {field} 早于除权日")
    return out


def bonus_quantity(eligible_qty: float, ratio: float) -> int:
    qty = eligible_qty * ratio
    if not math.isfinite(qty) or abs(qty - round(qty)) > 1e-7:
        raise UnsupportedCorporateActionError("公司行动存在零碎送股 fractional entitlement，缺少处理规则")
    return round(qty)
