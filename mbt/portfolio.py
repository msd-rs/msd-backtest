# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
import logging
import numpy as np
import polars as pl
import alpha as al
from alpha.sim import PortfolioEngine

logger = logging.getLogger(__name__)

from .account import (
  ACTION_BUY,
  ACTION_KEEP,
  ACTION_SELL,
  Fee,
  Operation,
  Position,
)
from .data_provider import DataProvider
from .select.panel import Panel
from .select.selector import Selector, SelectorDataProvider
from .strategy import Context, Strategy, PortfolioStrategy


@dataclass
class PortfolioRecord:
  """Record of an executed trade/rebalance in a PortfolioAccount."""

  bar: int
  ts: int
  group: int
  obj: str
  action: str  # "BUY" or "SELL"
  shares: float
  deal_price: float
  exec_price: float
  amount: float
  fee: float
  cash: float
  pos_shares: float

  def __getitem__(self, item: str):
    return getattr(self, item)

  def to_dict(self) -> dict:
    return asdict(self)


class _EnginePositionProxy:
  """Zero-copy read view into PortfolioEngine positions during active execution."""

  def __init__(self, account: "PortfolioAccount"):
    self._acc = account

  def __len__(self) -> int:
    return self._acc.groups

  def __getitem__(self, idx: int | np.integer | slice) -> Position:
    if isinstance(idx, slice):
      return [self[i] for i in range(*idx.indices(self._acc.groups))]  # type: ignore[return-value]
    return self._acc.position(int(idx), current_bar=self._acc.dp.i)

  def __iter__(self):
    eng = self._acc._engine
    if eng is None:
      return iter(self._acc._positions)
    sh = eng.shares()
    rc = eng.raw_cost_price()
    rm = eng.raw_mark_price()
    ac = eng.adj_cost_price()
    div = eng.dividends()
    eb = eng.entry_bars()
    cur = self._acc.dp.i
    return (
      Position(
        float(sh[g]),
        float(rc[g]),
        raw_cost_price=float(rc[g]),
        raw_mark_price=float(rm[g]),
        dividends=float(div[g]),
        adj_cost_price=float(ac[g]),
        holding_days=max(0, cur - int(eb[g])) if eb[g] >= 0 else 0,
      )
      for g in range(self._acc.groups)
    )


class PortfolioAccount:
  """Shared-cash multi-asset portfolio account for backtesting."""

  def __init__(
    self,
    initial_cash: float,
    dp: DataProvider,
    /,
    fee: Fee | None = None,
    unit: int = 100,
    slippage: float = 0.0,
    tradable_start: int = 0,
    tolerance: float = 0.0,
    benchmark: str | None = None,
  ):
    """Shared-cash multi-asset portfolio account for backtesting.

    Args:
      initial_cash: Starting cash for the portfolio.
      dp: DataProvider supplying prices and corporate action data.
      fee: Fee configuration.
      unit: Lot size (shares per trading unit, default 100).
      slippage: Slippage fraction applied to executions.
      tradable_start: (Legacy) Index of the first tradable group. Defaults to 0.
      tolerance: Minimum weight deviation threshold required to trigger rebalancing.
      benchmark: Obj code of benchmark index (e.g. 'SH000300'). When provided, its
        corresponding group is marked untradable (`tradable_mask=False`) and used
        exclusively for benchmark tracking.
    """
    self.initial_cash = float(initial_cash)
    self._cash = float(initial_cash)
    self.dp = dp
    self.fee = fee if fee is not None else Fee()
    self.unit = unit
    self.slippage = slippage
    self.tolerance = float(tolerance)
    self._deducted = 0.0
    self._records: list[PortfolioRecord] | None = []
    self._trades_raw: dict[str, np.ndarray] | None = None
    self._engine: PortfolioEngine | None = None
    self.exec_mode: str = "close"
    self.pending_signals: list[dict] = []

    self.groups = dp.groups
    self.bars = dp.bars
    self._positions = [Position.zero() for _ in range(self.groups)]
    self._held_groups: set[int] = set()
    self.entry_bars = np.full(self.groups, -1, dtype=np.int64)
    self._obj_to_group: dict[str, int] = (
      {s: i for i, s in enumerate(dp.objs)} if hasattr(dp, "objs") else {}
    )
    self.benchmark_obj = benchmark
    self.benchmark_idx: int | None = None
    if benchmark is not None and benchmark in self._obj_to_group:
      self.benchmark_idx = self._obj_to_group[benchmark]

    self.tradable_mask = np.ones(self.groups, dtype=bool)
    if self.benchmark_idx is not None:
      self.tradable_mask[self.benchmark_idx] = False
    elif tradable_start > 0:
      self.tradable_mask[: max(0, int(tradable_start))] = False
    self.dates = (
      dp.dates[: self.bars].astype("datetime64[s]").astype(np.int64)
      if self.bars > 0
      else np.array([], dtype=np.int64)
    )

    self.portfolio_assets = np.full(self.bars, self.initial_cash, dtype=np.float64)
    self.portfolio_cash = np.full(self.bars, self.initial_cash, dtype=np.float64)
    self.portfolio_ror = np.zeros(self.bars, dtype=np.float64)

    self.actions = dp.all("actions", create_if_not_exist=np.int32)
    self.action_shares = dp.all("action_shares", create_if_not_exist=np.float64)
    self.actions.fill(ACTION_KEEP)
    self.action_shares.fill(0.0)

    self._dividends = dp.all("dividend") if "dividend" in dp.data else None
    self._transfer_shares = dp.all("transfer_shares") if "transfer_shares" in dp.data else None
    self._right_shares = dp.all("right_shares") if "right_shares" in dp.data else None
    self._right_prices = dp.all("right_price") if "right_price" in dp.data else None
    self._corp_groups_by_bar: dict[int, list[int]] | None = None

  @property
  def positions(self):
    if self._engine is not None:
      return _EnginePositionProxy(self)
    return self._positions

  @positions.setter
  def positions(self, val: list[Position]):
    self._positions = val

  @property
  def cash(self) -> float:
    return self._engine.cash if self._engine is not None else self._cash

  @cash.setter
  def cash(self, val: float):
    self._cash = float(val)

  @property
  def deducted(self) -> float:
    return self._engine.deducted if self._engine is not None else self._deducted

  @deducted.setter
  def deducted(self, val: float):
    self._deducted = float(val)

  @property
  def dividends(self) -> np.ndarray:
    if self._dividends is None:
      self._dividends = np.zeros(len(self.dp.dates), dtype=np.float64)
    return self._dividends

  @property
  def transfer_shares(self) -> np.ndarray:
    if self._transfer_shares is None:
      self._transfer_shares = np.zeros(len(self.dp.dates), dtype=np.float64)
    return self._transfer_shares

  @property
  def right_shares(self) -> np.ndarray:
    if self._right_shares is None:
      self._right_shares = np.zeros(len(self.dp.dates), dtype=np.float64)
    return self._right_shares

  @property
  def right_prices(self) -> np.ndarray:
    if self._right_prices is None:
      self._right_prices = np.zeros(len(self.dp.dates), dtype=np.float64)
    return self._right_prices

  @property
  def records(self) -> list[PortfolioRecord]:
    if self._records is None:
      self._records = self._materialize_records()
    return self._records

  @records.setter
  def records(self, val: list[PortfolioRecord]):
    self._records = val
    self._trades_raw = None

  def _materialize_records(self) -> list[PortfolioRecord]:
    raw = self._trades_raw
    if not raw or len(raw.get("bar", ())) == 0:
      return []
    objs = self.dp.objs
    n_objs = len(objs)
    t_groups = raw["group"].tolist()
    t_bars = raw["bar"].tolist()
    t_ts = raw["ts"].tolist()
    t_actions = raw["action"].tolist()
    t_shares = raw["shares"].tolist()
    t_deal = raw["deal_price"].tolist()
    t_exec = raw["raw_cost_price"].tolist()
    t_fees = raw["fee"].tolist()
    t_cash = raw["cash_after"].tolist()
    t_pos = raw["pos_shares"].tolist()
    recs: list[PortfolioRecord] = []
    for k in range(len(t_bars)):
      g = t_groups[k]
      sh = t_shares[k]
      ep = t_exec[k]
      act = "BUY" if t_actions[k] == ACTION_BUY else "SELL"
      obj = objs[g] if g < n_objs else str(g)
      recs.append(
        PortfolioRecord(
          bar=t_bars[k],
          ts=t_ts[k],
          group=g,
          obj=obj,
          action=act,
          shares=sh,
          deal_price=t_deal[k],
          exec_price=ep,
          amount=sh * ep,
          fee=t_fees[k],
          cash=t_cash[k],
          pos_shares=t_pos[k],
        )
      )
    return recs

  def _hydrate_from_engine(self):
    eng = self._engine
    if eng is None:
      return
    self._cash = eng.cash
    self._deducted = eng.deducted
    self.entry_bars = eng.entry_bars()
    sh_arr = eng.shares()
    rc_arr = eng.raw_cost_price()
    rm_arr = eng.raw_mark_price()
    ac_arr = eng.adj_cost_price()
    div_arr = eng.dividends()

    held_idx = np.flatnonzero(sh_arr > 0.0)
    self._held_groups = set(int(g) for g in held_idx)
    for g in range(self.groups):
      if sh_arr[g] > 0.0:
        eb = int(self.entry_bars[g])
        h_days = max(0, self.bars - eb) if eb >= 0 else 0
        self._positions[g] = Position(
          float(sh_arr[g]),
          float(rc_arr[g]),
          raw_cost_price=float(rc_arr[g]),
          raw_mark_price=float(rm_arr[g]),
          dividends=float(div_arr[g]),
          adj_cost_price=float(ac_arr[g]),
          holding_days=h_days,
        )
      else:
        self._positions[g] = Position.zero()

    self._trades_raw = eng.trades()
    self._records = None

  def _ensure_corp_index(self):
    if self._corp_groups_by_bar is not None:
      return
    self._corp_groups_by_bar = {}
    if (
      self.groups > 0
      and self.bars > 0
      and (self._dividends is not None or self._transfer_shares is not None or self._right_shares is not None)
    ):
      has_corp = (
        (np.nan_to_num(self.dividends, nan=0.0) > 0)
        | (np.nan_to_num(self.transfer_shares, nan=0.0) > 0)
        | (np.nan_to_num(self.right_shares, nan=0.0) > 0)
      ).reshape(self.groups, self.bars)
      for b in np.flatnonzero(has_corp.any(axis=0)):
        self._corp_groups_by_bar[int(b)] = np.flatnonzero(has_corp[:, int(b)]).tolist()

  @property
  def tradable_start(self) -> int:
    """Index of the first tradable group, maintained for backward compatibility."""
    for g in range(self.groups):
      if self.tradable_mask[g]:
        return g
    return 0

  @tradable_start.setter
  def tradable_start(self, val: int):
    self.tradable_mask = np.ones(self.groups, dtype=bool)
    if val > 0:
      self.tradable_mask[: max(0, int(val))] = False
    if self._engine is not None:
      self._engine.set_tradable_mask(self.tradable_mask)

  @property
  def market_value(self) -> float:
    if self._engine is not None:
      return float(self._engine.market_value)
    return float(sum(self.positions[g].value for g in self._held_groups if self.tradable_mask[g]))

  @property
  def asset(self) -> float:
    if self._engine is not None:
      return float(self._engine.asset)
    return self.cash + self.market_value

  def shares_array(self) -> np.ndarray:
    if self._engine is not None:
      return self._engine.shares()
    return np.array([p.shares for p in self.positions], dtype=np.float64)

  def mark_prices_array(self) -> np.ndarray:
    if self._engine is not None:
      return self._engine.raw_mark_price()
    return np.array(
      [p.raw_mark_price if p.raw_mark_price > 0 else p.raw_cost_price for p in self.positions],
      dtype=np.float64,
    )

  def weights(self) -> np.ndarray:
    total = self.asset
    if total <= 0:
      return np.zeros(self.groups, dtype=np.float64)
    if self._engine is not None:
      sh = self._engine.shares()
      rm = self._engine.raw_mark_price()
      rc = self._engine.raw_cost_price()
      eff_p = np.where(rm > 0.0, rm, rc)
      return np.where(self.tradable_mask, (sh * eff_p) / total, 0.0)
    return np.array([p.value / total if self.tradable_mask[g] else 0.0 for g, p in enumerate(self.positions)], dtype=np.float64)

  @property
  def held_groups(self) -> list[int]:
    """List of group indices that currently have held shares (> 0)."""
    if self._engine is not None:
      sh = self._engine.shares()
      return np.flatnonzero(self.tradable_mask & (sh > 0.0)).tolist()
    return sorted(
      g
      for g in self._held_groups
      if self.tradable_mask[g] and self.positions[g].shares > 0
    )

  @property
  def held_objs(self) -> list[str]:
    """List of obj symbols that currently have held shares (> 0)."""
    objs = self.dp.objs
    return [objs[g] for g in self.held_groups if g < len(objs)]

  def has_position(self, obj: str | int) -> bool:
    """Check whether the account currently holds shares in obj or group."""
    g = self._resolve_group_idx(obj)
    if self._engine is not None:
      return self._engine.state_of(g)[0] > 0.0
    return self.positions[g].shares > 0

  def position(self, obj: str | int, current_bar: int | None = None) -> Position:
    """Get the current Position object for an obj or group index."""
    g = self._resolve_group_idx(obj)
    if self._engine is not None:
      sh, rc, rm, ac, div, eb = self._engine.state_of(g)
      cur = self.bars if current_bar is None else current_bar
      h_days = max(0, cur - int(eb)) if eb >= 0 else 0
      if current_bar is not None and 0 <= current_bar < self.bars:
        j = g * self.bars + current_bar
        p = self.dp.all("price")[j]
        if not np.isnan(p) and p > 0:
          rm = float(p)
      return Position(
        sh,
        rc,
        raw_cost_price=rc,
        raw_mark_price=rm,
        dividends=div,
        adj_cost_price=ac,
        holding_days=h_days,
      )
    pos = self.positions[g]
    pos.holding_days = self.holding_days(g, current_bar=current_bar)
    if current_bar is not None and 0 <= current_bar < self.bars:
      j = g * self.bars + current_bar
      p = self.dp.all("price")[j]
      if not np.isnan(p) and p > 0:
        pos.raw_mark_price = float(p)
    return pos

  def runoff_objs(self, universe: set[str] | list[str] | None = None) -> list[str]:
    """Held stocks that are no longer in the active universe whitelist."""
    if universe is None:
      return []
    u_set = set(universe)
    return [s for s in self.held_objs if s not in u_set]

  def daily_snapshot(self, bar: int | None = None) -> dict:
    """Extract a structured daily snapshot of NAV, holdings, trades, and pending signals.

    Args:
      bar: Target bar index (defaults to the last bar `self.bars - 1`).

    Returns:
      Dictionary containing date, exec_mode, total_assets, cash, market_value,
      position_ratio, daily_ror, total_ror, holdings, trades, and pending_signals.
    """
    exec_mode = getattr(self, "exec_mode", "close")
    if self.bars <= 0:
      return {
        "bar": -1,
        "date": "",
        "exec_mode": exec_mode,
        "total_assets": float(self.initial_cash),
        "cash": float(self.initial_cash),
        "market_value": 0.0,
        "position_ratio": 0.0,
        "daily_ror": 0.0,
        "total_ror": 0.0,
        "holdings": [],
        "trades": [],
        "pending_signals": [],
      }

    idx = self.bars - 1 if bar is None else int(bar)
    if idx < 0 or idx >= self.bars:
      raise IndexError(f"bar index {idx} out of range [0, {self.bars - 1}]")

    raw_date = self.dp.dates[idx]
    if isinstance(raw_date, np.datetime64):
      date_str = str(np.datetime_as_string(raw_date, unit="D"))
    else:
      date_str = str(raw_date)[:10]

    total_assets = float(self.portfolio_assets[idx])
    cash = float(self.portfolio_cash[idx])
    market_value = max(0.0, total_assets - cash)
    position_ratio = (market_value / total_assets) if total_assets > 0 else 0.0
    prev_assets = float(self.portfolio_assets[idx - 1]) if idx > 0 else float(self.initial_cash)
    daily_ror = (total_assets / prev_assets - 1.0) if prev_assets > 0 else 0.0
    total_ror = float(self.portfolio_ror[idx])

    objs = self.dp.objs
    prices = self.dp.all("price")
    holdings: list[dict] = []
    for g in range(self.groups):
      pos = self.position(g, current_bar=idx)
      if pos.shares <= 0:
        continue
      mark_p = float(prices[g * self.bars + idx])
      if np.isnan(mark_p) or mark_p <= 0:
        mark_p = float(pos.raw_cost_price)
      mv = float(pos.shares) * mark_p
      holdings.append({
        "obj": objs[g] if g < len(objs) else str(g),
        "shares": float(pos.shares),
        "price": mark_p,
        "market_value": mv,
        "weight": (mv / total_assets) if total_assets > 0 else 0.0,
        "cost_price": float(pos.raw_cost_price),
        "ror": float(pos.ror),
        "unrealized_pnl": float(pos.unrealized_pnl),
        "holding_days": int(pos.holding_days),
      })
    holdings.sort(key=lambda h: h["weight"], reverse=True)

    trades = [r.to_dict() for r in self.records if r.bar == idx]
    pending = list(self.pending_signals) if idx == self.bars - 1 else []

    return {
      "bar": idx,
      "date": date_str,
      "exec_mode": exec_mode,
      "total_assets": total_assets,
      "cash": cash,
      "market_value": market_value,
      "position_ratio": position_ratio,
      "daily_ror": daily_ror,
      "total_ror": total_ror,
      "holdings": holdings,
      "trades": trades,
      "pending_signals": pending,
    }

  def apply_corporate_actions(self, ts: int):
    if not self._held_groups:
      return
    self._ensure_corp_index()
    assert self._corp_groups_by_bar is not None
    corp_groups = self._corp_groups_by_bar.get(ts)
    if not corp_groups:
      return
    for g in corp_groups:
      if g not in self._held_groups:
        continue
      pos = self.positions[g]
      if pos.shares <= 0:
        continue
      j = g * self.bars + ts
      div = self.dividends[j]
      trans = self.transfer_shares[j]
      r_share = self.right_shares[j]
      r_price = self.right_prices[j]

      if div > 0 and not np.isnan(div):
        self.cash += pos.shares * div
        pos.dividends += pos.shares * div
      if trans > 0 and not np.isnan(trans):
        old_shares = pos.shares
        pos.shares += pos.shares * trans
        if pos.shares > 0:
          ratio = old_shares / pos.shares
          pos.raw_cost_price *= ratio
          pos._adj_cost_price *= ratio
      if r_share > 0 and not np.isnan(r_share):
        max_r = pos.shares * r_share
        if r_price > 0 and not np.isnan(r_price):
          affordable = int(self.cash / r_price) if self.cash > 0 else 0
          actual_r = min(max_r, float(affordable))
          self.cash -= actual_r * r_price
          right_cost = actual_r * r_price
        else:
          actual_r = max_r
          right_cost = 0.0
        if actual_r > 0:
          old_cost = pos.cost_basis
          old_raw_cost = pos.raw_cost_price
          pos.shares += actual_r
          if pos.shares > 0:
            pos.raw_cost_price = (old_cost + right_cost) / pos.shares
            if old_raw_cost > 0:
              pos._adj_cost_price = pos._adj_cost_price * (pos.raw_cost_price / old_raw_cost)

  def _sell_group(self, g: int, ts: int, shares_to_sell: float, deal_price: float):
    pos = self.positions[g]
    if pos.shares <= 0 or shares_to_sell <= 0 or np.isnan(deal_price) or deal_price <= 0:
      return
    actual_shares = min(pos.shares, float(shares_to_sell))
    if actual_shares <= 0:
      return
    exec_price = deal_price * (1.0 - self.slippage)
    sell_pos = Position(actual_shares, exec_price)
    fee_item = self.fee.get(self.dates[ts])
    fee = fee_item.sell(sell_pos.value)
    self.cash += sell_pos.value - fee
    self._positions[g] = pos - sell_pos
    if self._positions[g].shares <= 0:
      self.entry_bars[g] = -1
      self._held_groups.discard(g)
    self.deducted += fee
    j = g * self.bars + ts
    self.actions[j] = ACTION_SELL
    self.action_shares[j] = actual_shares

    obj = self.dp.objs[g] if (hasattr(self.dp, "objs") and g < len(self.dp.objs)) else str(g)
    ts_val = int(self.dates[ts]) if (len(self.dates) > 0 and ts < len(self.dates)) else ts
    self.records.append(
      PortfolioRecord(
        bar=ts,
        ts=ts_val,
        group=g,
        obj=obj,
        action="SELL",
        shares=actual_shares,
        deal_price=float(deal_price),
        exec_price=float(exec_price),
        amount=float(sell_pos.value),
        fee=float(fee),
        cash=float(self.cash),
        pos_shares=float(self._positions[g].shares),
      )
    )

  def _buy_group(self, g: int, ts: int, cash_budget: float, deal_price: float):
    if self.cash <= 0 or cash_budget <= 0 or np.isnan(deal_price) or deal_price <= 0:
      return
    budget = min(self.cash, float(cash_budget))
    exec_price = deal_price * (1.0 + self.slippage)
    fee_item = self.fee.get(self.dates[ts])
    buy_pos = Position.from_cash(budget, exec_price, self.unit)
    fee = fee_item.buy(buy_pos.value)
    while buy_pos.shares > 0 and buy_pos.value + fee > budget:
      buy_pos = Position(max(buy_pos.shares - self.unit, 0.0), exec_price)
      fee = fee_item.buy(buy_pos.value)

    if buy_pos.shares <= 0:
      return

    # Calculate adjustment ratio if available
    j = g * self.bars + ts
    adj_cost = exec_price
    if hasattr(self.dp, "data") and "close" in self.dp.data and "price" in self.dp.data:
      raw_p = self.dp.data["price"][j]
      fw_p = self.dp.data["close"][j]
      if raw_p > 0 and not np.isnan(raw_p) and not np.isnan(fw_p):
        adj_cost = exec_price * (fw_p / raw_p)
    buy_pos._adj_cost_price = float(adj_cost)

    if self._positions[g].shares <= 0:
      self.entry_bars[g] = ts

    self.cash -= buy_pos.value + fee
    self._positions[g] = self._positions[g] + buy_pos
    self._held_groups.add(g)
    self.deducted += fee
    j = g * self.bars + ts
    self.actions[j] = ACTION_BUY
    self.action_shares[j] = buy_pos.shares

    obj = self.dp.objs[g] if (hasattr(self.dp, "objs") and g < len(self.dp.objs)) else str(g)
    ts_val = int(self.dates[ts]) if (len(self.dates) > 0 and ts < len(self.dates)) else ts
    self.records.append(
      PortfolioRecord(
        bar=ts,
        ts=ts_val,
        group=g,
        obj=obj,
        action="BUY",
        shares=float(buy_pos.shares),
        deal_price=float(deal_price),
        exec_price=float(exec_price),
        amount=float(buy_pos.value),
        fee=float(fee),
        cash=float(self.cash),
        pos_shares=float(self._positions[g].shares),
      )
    )

  def trades_df(self) -> pl.DataFrame:
    """Return trade and rebalance history as a Polars DataFrame."""
    if self._records is None and self._trades_raw is not None:
      raw = self._trades_raw
      n_tr = len(raw.get("bar", ()))
      if n_tr > 0:
        objs_arr = np.asarray(self.dp.objs, dtype=object)
        t_groups = raw["group"]
        t_actions = raw["action"]
        t_shares = raw["shares"]
        t_exec = raw["raw_cost_price"]
        obj_col = objs_arr[t_groups].tolist() if len(objs_arr) > 0 else [str(g) for g in t_groups.tolist()]
        act_col = np.where(t_actions == ACTION_BUY, "BUY", "SELL").tolist()
        return pl.DataFrame({
          "bar": raw["bar"].astype(np.int64),
          "ts": raw["ts"].astype(np.int64),
          "group": t_groups.astype(np.int64),
          "obj": obj_col,
          "action": act_col,
          "shares": t_shares,
          "deal_price": raw["deal_price"],
          "exec_price": t_exec,
          "amount": t_shares * t_exec,
          "fee": raw["fee"],
          "cash": raw["cash_after"],
          "pos_shares": raw["pos_shares"],
        })
    if not self.records:
      return pl.DataFrame(
        schema={
          "bar": pl.Int64,
          "ts": pl.Int64,
          "group": pl.Int64,
          "obj": pl.Utf8,
          "action": pl.Utf8,
          "shares": pl.Float64,
          "deal_price": pl.Float64,
          "exec_price": pl.Float64,
          "amount": pl.Float64,
          "fee": pl.Float64,
          "cash": pl.Float64,
          "pos_shares": pl.Float64,
        }
      )
    return pl.DataFrame([r.to_dict() for r in self.records])

  def records_df(self) -> pl.DataFrame:
    """Alias for trades_df()."""
    return self.trades_df()

  def holding_ror(self, obj: str | int) -> float:
    """Return the current holding rate of return for obj/group as a float (e.g. -0.05 for -5%)."""
    return self.position(obj).holding_ror

  def holding_pnl(self, obj: str | int) -> float:
    """Return the current holding profit/loss amount for obj/group."""
    return self.position(obj).holding_pnl

  def cost_price(self, obj: str | int, adjusted: bool = False) -> float:
    """Return acquisition cost price.

    Args:
      obj: Symbol or group index.
      adjusted: If True, returns adjusted equivalent cost (adj_cost_price),
        safe to compare with `ctx.close`. If False, returns raw execution cost price.
    """
    pos = self.position(obj)
    return pos.adj_cost_price if adjusted else pos.raw_cost_price

  def holding_days(self, obj: str | int, current_bar: int | None = None) -> int:
    """Return number of bars since position was first acquired."""
    g = self._resolve_group_idx(obj)
    if self._engine is not None:
      eb = int(self._engine.state_of(g)[5])
      if eb < 0:
        return 0
      cur = self.bars if current_bar is None else current_bar
      return max(0, cur - eb)
    if self.entry_bars[g] < 0:
      return 0
    cur = self.bars if current_bar is None else current_bar
    return max(0, cur - int(self.entry_bars[g]))

  def _resolve_group_idx(self, obj: str | int | np.integer) -> int:
    if isinstance(obj, str):
      g = self._obj_to_group.get(obj)
      if g is not None and 0 <= g < self.groups:
        return g
      raise ValueError(
        f"Symbol '{obj}' is not in the pre-loaded portfolio pool (total {self.groups} objs). "
        "Ensure all target symbols are included via -s/--objs."
      )
    elif isinstance(obj, (int, np.integer)):
      if 0 <= obj < self.groups:
        return int(obj)
      raise ValueError(f"Group index {obj} out of range [0, {self.groups})")
    raise TypeError(f"Expected str or int for obj, got {type(obj).__name__}")

  def update_mark_to_market(self, ts: int, mark_prices: np.ndarray):
    for g in self._held_groups:
      p = mark_prices[g]
      if not np.isnan(p) and p > 0:
        self.positions[g].raw_mark_price = float(p)
      if self.positions[g].shares > 0 and self.entry_bars[g] >= 0:
        self.positions[g].holding_days = max(0, ts - int(self.entry_bars[g]))
      else:
        self.positions[g].holding_days = 0
    total_asset = self.asset
    self.portfolio_cash[ts] = self.cash
    self.portfolio_assets[ts] = total_asset
    self.portfolio_ror[ts] = (
      (total_asset / self.initial_cash - 1.0) if self.initial_cash > 0 else 0.0
    )

  def rebalance_step(
    self,
    ts: int,
    target_weights: np.ndarray,
    deal_prices: np.ndarray,
    mark_prices: np.ndarray,
    limited: np.ndarray | None = None,
    tolerance: float | None = None,
    universe: set[str] | None = None,
  ):
    """Execute a single rebalancing step to align current holdings with target_weights.

    Execution Flow:
    1. Zero out weights for non-tradable assets (such as benchmark index).
    2. Normalize target weights if sum exceeds 1.0.
    3. Tolerance check: if deviation between current weight and target is within tolerance,
       maintain current position to avoid turnover costs (except for target 0 liquidation).
    4. Pass 1 (Sells): Liquidations and reductions executed first to release cash.
    5. Pass 2 (Buys): Buys scaled by available cash pool if total buy capital exceeds available cash.
    6. Mark-to-market valuation update.
    """
    tol = self.tolerance if tolerance is None else float(tolerance)
    weights = np.maximum(np.nan_to_num(np.asarray(target_weights, dtype=np.float64), nan=0.0), 0.0)
    weights[~self.tradable_mask] = 0.0
    w_sum = float(np.sum(weights))
    if w_sum > 1.0:
      weights = weights / w_sum

    # Update current valuation with deal_prices where valid before computing targets
    for g in self._held_groups:
      if not self.tradable_mask[g]:
        continue
      dp_g = deal_prices[g]
      if not np.isnan(dp_g) and dp_g > 0 and self.positions[g].shares > 0:
        self.positions[g].raw_mark_price = float(dp_g)

    total_equity = self.asset
    target_values = total_equity * weights

    # Apply tolerance band: keep current position value if deviation is within tolerance
    if tol > 0.0 and total_equity > 0.0:
      for g in np.flatnonzero(self.tradable_mask & (weights > 0.0)):
        tgt_w = weights[g]
        dp_g = deal_prices[g]
        if np.isnan(dp_g) or dp_g <= 0:
          continue
        curr_val = self.positions[g].shares * dp_g
        curr_w = curr_val / total_equity
        if abs(tgt_w - curr_w) <= tol:
          target_values[g] = curr_val

    # Pass 1: Sells first to free up cash (only held positions can be sold)
    for g in sorted(self._held_groups):
      if not self.tradable_mask[g]:
        continue
      lim = limited[g] if limited is not None else 0
      if lim == -1:
        continue
      dp_g = deal_prices[g]
      if np.isnan(dp_g) or dp_g <= 0:
        continue
      curr_val = self.positions[g].shares * dp_g
      tgt_val = target_values[g]
      if tgt_val == 0.0 and self.positions[g].shares > 0:
        self._sell_group(g, ts, self.positions[g].shares, dp_g)
      elif curr_val > tgt_val:
        excess_val = curr_val - tgt_val
        shares_to_sell = int(excess_val / dp_g)
        shares_to_sell = shares_to_sell - (shares_to_sell % self.unit)
        if shares_to_sell > 0:
          self._sell_group(g, ts, float(shares_to_sell), dp_g)

    # Pass 2: Buys using available cash (only non-zero target values can exceed curr_val)
    buy_deficits = []
    for g in np.flatnonzero(self.tradable_mask & (target_values > 0.0)):
      if universe is not None and self.dp.objs[g] not in universe:
        continue
      lim = limited[g] if limited is not None else 0
      if lim == 1:
        continue
      dp_g = deal_prices[g]
      if np.isnan(dp_g) or dp_g <= 0:
        continue
      curr_val = self.positions[g].shares * dp_g
      tgt_val = target_values[g]
      if tgt_val > curr_val:
        buy_deficits.append((g, tgt_val - curr_val, dp_g))

    total_deficit = sum(d[1] for d in buy_deficits)
    if total_deficit > 0 and self.cash > 0:
      scale = min(1.0, self.cash / total_deficit)
      for g, deficit, dp_g in buy_deficits:
        self._buy_group(int(g), ts, deficit * scale, dp_g)

    self.update_mark_to_market(ts, mark_prices)


class _HistoricalPanelSelectorDataProvider(SelectorDataProvider):
  """In-memory SelectorDataProvider sliced at current backtest bar for SelectorStrategy."""

  def __init__(
    self,
    panel_slice: Panel,
    fin_only_year: bool = False,
    financial_frame: pl.DataFrame | None = None,
  ):
    super().__init__()
    self.mode = "backtest_slice"
    self._panel = panel_slice
    self._fin_only_year = fin_only_year
    self._financial_frame = financial_frame
    if panel_slice.bars > 0 and "ts" in panel_slice.columns:
      self.date = str(panel_slice.timeline()[-1])[:10]

  @property
  def financial_frame(self) -> pl.DataFrame | None:
    """Preloaded historical financial reports frame (if available)."""
    return self._financial_frame

  def load_financial(
    self,
    objs: list[str],
    fields: list[str],
    only_year: bool = True,
    lastN: int = 12,
  ) -> pl.DataFrame:
    """Load up to `lastN` historical financial reports for `objs` without lookahead bias.

    Filters records strictly by `ts <= current_bar_date` where `ts` is the
    announcement / statutory disclosure date.

    Args:
      objs: List of stock codes to query.
      fields: Financial field names (e.g. ['f001', 'f205']).
      only_year: If True, only select annual reports (report_ts.month == 12).
      lastN: Maximum number of historical records per stock.

    Returns:
      pl.DataFrame with columns ['obj', 'ts', 'report_ts', *fields].
    """
    if not objs or not fields:
      return pl.DataFrame()

    if self._financial_frame is None or len(self._financial_frame) == 0:
      raise ValueError(
        f"In-backtest load_financial requested field(s) {fields}, but no financial data was preloaded in backtest. "
        f"Please declare fin_fields on your Strategy or Selector class (e.g. fin_fields={list(fields)}) or pass via CLI."
      )

    missing = [f for f in fields if f not in self._financial_frame.columns]
    if missing:
      raise ValueError(
        f"In-backtest load_financial requested field(s) {missing}, but only {list(self._financial_frame.columns)} were preloaded. "
        f"Please include {missing} in your Strategy or Selector fin_fields declaration."
      )

    current_date = self._panel.timeline()[-1] if self._panel.bars > 0 else 0
    target_set = set(objs)
    df = self._financial_frame.filter(
      (pl.col("ts") <= current_date) & (pl.col("obj").is_in(target_set))
    )

    if only_year:
      df = df.filter(pl.col("report_ts").dt.month() == 12)

    if len(df) == 0:
      return pl.DataFrame()

    available_fields = [f for f in fields if f in df.columns]
    keep_cols = ["obj", "ts", "report_ts"] + available_fields
    df = df.select([c for c in keep_cols if c in df.columns])

    df = (
      df.sort(["obj", "report_ts"])
      .group_by("obj", maintain_order=True)
      .tail(lastN)
    )
    return df

  def load_kline(
    self,
    objs: list[str],
    lastN: int | list[int] = 100,
    fin_fields: list[str] = [],
    fin_only_year: bool = True,
    baseline: str | None = None,
  ) -> Panel:
    if fin_fields:
      missing = [f for f in fin_fields if f not in self._panel.columns]
      if missing:
        raise ValueError(
          f"In-backtest dynamic selection requested financial field(s) {missing}, but they were not preloaded in backtest data. "
          f"Please declare fin_fields on your Strategy or Selector class (e.g. fin_fields={list(fin_fields)}) or pass via CLI."
        )

    if len(fin_fields) > 0 and fin_only_year and not self._fin_only_year:
      raise ValueError(
        "_HistoricalPanelSelectorDataProvider: load_snapshot/load_kline requested fin_only_year=True with fin_fields, "
        "but backtest data was not loaded with fin_only_year=True (contains quarterly as-of joins). "
        "To align data and prevent silent quarterly/annual mismatch, set fin_only_year=True on your Strategy/Selector or via --fin-only-year."
      )
    n_bars = lastN if isinstance(lastN, int) else int(lastN[0])
    n_bars = min(n_bars, self._panel.bars)
    start_bar = self._panel.bars - n_bars
    return self._panel.slice(start_bar, self._panel.bars, objs=objs)

  def load_snapshot(
    self,
    objs: list[str],
    fin_fields: list[str] = [],
    fin_only_year: bool = True,
  ) -> pl.DataFrame:
    p = self.load_kline(
      objs,
      lastN=1,
      fin_fields=fin_fields,
      fin_only_year=fin_only_year,
    )
    cols = {col: p.last(col) for col in p.columns}
    for base_col in ("open", "high", "low", "close"):
      raw_col = f"raw_{base_col}"
      if raw_col in cols:
        cols[base_col] = cols[raw_col]
    if "report_ts" in cols and "f_ts" not in cols:
      cols["f_ts"] = cols["report_ts"]
    return p.cross_section(**cols)


class PortfolioContext(Context):
  """Strategy context for shared-cash portfolio backtesting."""

  def __init__(self, account: PortfolioAccount):
    self.account = account
    self.dp = account.dp
    self._obj_to_group = account._obj_to_group
    self.universe: set[str] | None = None
    self._universe_mask: np.ndarray | None = None
    self._full_panel: Panel | None = None
    self._in_execute = False
    self._rors_bar: int = -1
    self._rors_cache: np.ndarray | None = None
    self._shares_bar: int = -1
    self._shares_cache: np.ndarray | None = None
    n = len(self.dp.dates)
    self.buy_percents = np.zeros(n, dtype=np.float64)
    self.sell_percents = np.zeros(n, dtype=np.float64)
    self.target_weights = np.full((self.dp.bars, self.dp.groups), np.nan, dtype=np.float64)
    self.target_tolerances = np.full(self.dp.bars, np.nan, dtype=np.float64)
    self._has_target_weights = np.zeros(self.dp.bars, dtype=bool)
    self._group_offsets = np.arange(self.dp.groups, dtype=np.int64) * self.dp.bars

  def set_universe(self, objs: list[str] | set[str] | None):
    """Set the active universe whitelist for buying."""
    if objs is None:
      self.universe = None
      self._universe_mask = None
    else:
      missing = [s for s in objs if s not in self._obj_to_group]
      if missing:
        raise ValueError(
          f"set_universe: {len(missing)} obj(s) not in pre-loaded pool: "
          f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
          "Pass the full candidate universe via -s/--objs (e.g. -s A_STOCKS) when starting the backtest."
        )
      self.universe = set(objs)
      u_mask = np.zeros(self.dp.groups, dtype=bool)
      for s in self.universe:
        u_mask[self._obj_to_group[s]] = True
      self._universe_mask = u_mask

  @property
  def universe_objs(self) -> list[str]:
    """List of objs in the active universe whitelist, or all objs if universe is not set."""
    if self.universe is not None:
      return [s for s in self.dp.objs if s in self.universe]
    return list(self.dp.objs)

  def slice_panel(self, lastN: int = 100) -> Panel:
    """Return a Panel sliced up to current bar (inclusive, no lookahead)."""
    if self._full_panel is None:
      self._full_panel = self.dp.to_panel()
    end_bar = self.dp.i + 1
    n_bars = min(end_bar, max(1, int(lastN)))
    start_bar = end_bar - n_bars
    return self._full_panel.slice(start_bar, end_bar)

  def run_selector(
    self,
    selector: Selector | Callable[[Panel], list[str]],
    lastN: int = 100,
  ) -> list[str]:
    """Execute a Selector on historical data up to current bar without lookahead."""
    sub_panel = self.slice_panel(lastN=lastN)
    with sub_panel.ctx():
      if isinstance(selector, Selector):
        fin_only_year = getattr(self.dp, "fin_only_year", False)
        financial_frame = getattr(self.dp, "financial_frame", None)
        dp_slice = _HistoricalPanelSelectorDataProvider(
          sub_panel, fin_only_year=fin_only_year, financial_frame=financial_frame
        )
        selected = selector.execute(dp_slice, list(sub_panel.objs))
      else:
        selected = selector(sub_panel)
    selected_list = list(selected)
    missing = [s for s in selected_list if s not in self._obj_to_group]
    if missing:
      raise ValueError(
        f"run_selector: {len(missing)} selected obj(s) not in pre-loaded pool: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
        "In-backtest dynamic selection operates only on the pre-loaded pool. "
        "Ensure your Selector uses the passed `stocks` argument in step00 during backtest (`if not self.is_backtest and not stocks:`), "
        "or preload the full candidate pool via -s/--objs."
      )
    return selected_list

  def objs(self) -> list[str]:
    return self.dp.objs

  def index_of(self, obj: str) -> int:
    """Return group index of obj. Raises ValueError if not found."""
    if obj in self._obj_to_group:
      return self._obj_to_group[obj]
    raise ValueError(f"'{obj}' is not in context objs")

  @property
  def bar(self) -> int:
    return self.dp.i

  @property
  def date(self) -> str:
    """Current bar date as YYYY-MM-DD string."""
    if self.dp.bars > 0 and 0 <= self.dp.i < self.dp.bars:
      return str(self.dp.dates[self.dp.i])[:10]
    return ""

  @property
  def current_date(self) -> str:
    return self.date

  @property
  def step(self) -> int:
    return self.dp.i

  @property
  def warmup_bars(self) -> int:
    return int(getattr(self.dp, "warmup_bars", 0))

  @property
  def is_warmup(self) -> bool:
    return self.dp.i < self.warmup_bars

  @property
  def bars(self) -> int:
    return self.dp.bars

  @property
  def n_objs(self) -> int:
    return self.dp.groups

  def panel(self, allow_future: bool = False, sliced: bool | None = None) -> Panel:
    """Return a Panel view of the data provider.

    During strategy execute(), defaults to returning a strictly sliced Panel
    up to the current bar (`sliced=True`) to prevent lookahead bias.
    To explicitly obtain the full-range panel with future bars, pass `allow_future=True`.

    Args:
      allow_future: If True, allows returning full-range panel during execute().
      sliced: If True, returns sliced panel. If None, defaults to True during execute().
    """
    if sliced is True or (sliced is None and self._in_execute and not allow_future):
      return self.slice_panel(lastN=self.dp.i + 1)
    if self._in_execute and allow_future:
      logger.warning(
        "Calling ctx.panel(allow_future=True) during execute() returns the full-range panel with future bars. "
        "Lookahead risk present."
      )
    return self.dp.to_panel()

  def data(self, name: str, /, obj: str | None = None) -> np.ndarray:
    if name in self.dp.data:
      return self.dp.all(name, obj=obj)
    raise ValueError(f"data '{name}' not found")

  def _require_obj(self, obj: str | int | None) -> str | int:
    if obj is not None:
      return obj
    if self.dp.groups == 1:
      return 0
    raise ValueError(
      "PortfolioContext requires 'obj' (symbol or group index) when multiple objs exist."
    )

  def position(self, obj: str | int | None = None) -> Position:
    """Get the current Position object for an obj or group index."""
    return self.account.position(self._require_obj(obj), current_bar=self.dp.i)

  def holding_ror(self, obj: str | int | None = None) -> float:
    """Return current holding rate of return for obj/group as a float (e.g. -0.05 for -5%)."""
    target = self._require_obj(obj)
    if self.account._engine is not None:
      g = self.account._resolve_group_idx(target)
      return float(self.holding_rors()[g])
    return self.account.holding_ror(target)

  def holding_pnl(self, obj: str | int | None = None) -> float:
    """Return current holding profit/loss amount for obj/group."""
    return self.account.holding_pnl(self._require_obj(obj))

  def cost_price(self, obj: str | int | None = None, adjusted: bool = False) -> float:
    """Return acquisition cost price.

    Args:
      obj: Symbol or group index.
      adjusted: If True, returns adjusted equivalent cost (adj_cost_price),
        safe to compare with `ctx.close`. If False, returns raw execution cost price.
    """
    return self.account.cost_price(self._require_obj(obj), adjusted=adjusted)

  def holding_days(self, obj: str | int | None = None) -> int:
    """Return number of bars since position was first acquired."""
    return self.account.holding_days(self._require_obj(obj), current_bar=self.dp.i)

  def holding_shares(self) -> np.ndarray:
    """Return 1D float64 array of currently held shares across all objs."""
    eng = self.account._engine
    if eng is not None:
      cur_bar = self.dp.i
      if self._shares_bar != cur_bar or self._shares_cache is None:
        self._shares_cache = eng.shares()
        self._shares_bar = cur_bar
      return self._shares_cache
    return np.array([p.shares for p in self.account.positions], dtype=np.float64)

  def holding_rors(self) -> np.ndarray:
    """Return 1D float64 array of holding ROR across all objs at current bar."""
    cur_bar = self.dp.i
    eng = self.account._engine
    if eng is not None:
      if self._rors_bar != cur_bar or self._rors_cache is None:
        prices = getattr(self, "_prices_contig", None)
        if prices is None:
          prices = np.ascontiguousarray(self.dp.all("price"), dtype=np.float64)
          self._prices_contig = prices
        self._rors_cache = eng.holding_rors(prices, cur_bar)
        self._rors_bar = cur_bar
      return self._rors_cache
    return np.array(
      [self.account.position(g, current_bar=cur_bar).holding_ror for g in range(self.dp.groups)],
      dtype=np.float64,
    )

  def _resolve_bar_flags(self, flags) -> tuple[np.ndarray, np.ndarray]:
    idx = self._group_offsets + self.dp.i
    if isinstance(flags, str):
      mask = np.zeros(self.dp.groups, dtype=bool)
      g = self.account._resolve_group_idx(flags)
      mask[g] = True
      return idx, mask
    if isinstance(flags, (int, np.integer)) and not isinstance(flags, (bool, np.bool_)):
      mask = np.zeros(self.dp.groups, dtype=bool)
      g = self.account._resolve_group_idx(int(flags))
      mask[g] = True
      return idx, mask
    if isinstance(flags, (list, set, tuple)):
      mask = np.zeros(self.dp.groups, dtype=bool)
      for item in flags:
        if isinstance(item, (str, int, np.integer)) and not isinstance(item, (bool, np.bool_)):
          g = self.account._resolve_group_idx(item)
          mask[g] = True
        else:
          raise TypeError(f"Unsupported flag element type: {type(item).__name__}")
      return idx, mask
    arr = np.asarray(flags, dtype=bool)
    if arr.ndim == 1 and len(arr) == self.dp.groups:
      return idx, arr
    return idx, arr[idx]

  def buy(self, flags, percent: float = 1.0):
    if self.dp.groups == 0:
      return
    idx, mask = self._resolve_bar_flags(flags)
    # Apply universe whitelist guard for buying
    if self._universe_mask is not None:
      mask = mask & self._universe_mask
    active = idx[mask]
    if len(active) > 0:
      self.buy_percents[active] = percent

  def sell(self, flags, percent: float = 1.0):
    if self.dp.groups == 0:
      return
    idx, mask = self._resolve_bar_flags(flags)
    # Sell does NOT filter by universe (supports Runoff natural liquidation)
    active = idx[mask]
    if len(active) > 0:
      self.sell_percents[active] = percent

  def rebalance(self, weights: np.ndarray, tolerance: float | None = None):
    """Set target portfolio weights for the current bar (shape: (groups,) or (groups * bars,))."""
    if tolerance is not None:
      self.target_tolerances[self.dp.i] = float(tolerance)
    arr = np.asarray(weights, dtype=np.float64)
    ts_start = self.account.tradable_start
    if arr.ndim == 1 and len(arr) == self.dp.groups:
      w = arr.copy()
      if ts_start > 0:
        w[:ts_start] = 0.0
    elif ts_start > 0 and arr.ndim == 1 and len(arr) == self.dp.groups - ts_start:
      w = np.zeros(self.dp.groups, dtype=np.float64)
      w[ts_start:] = arr
    elif arr.ndim == 1 and len(arr) == self.dp.groups * self.dp.bars:
      idx = self._group_offsets + self.dp.i
      w = arr[idx].copy()
      if ts_start > 0:
        w[:ts_start] = 0.0
    else:
      raise ValueError(
        f"weights must have shape ({self.dp.groups},) or ({self.dp.groups * self.dp.bars},)"
      )

    # Apply universe whitelist guard: stocks outside universe cannot be bought or increased
    if self._universe_mask is not None:
      outside = ~self._universe_mask
      if np.any(outside & (w > 0.0)):
        shares = self.account.shares_array()
        w[outside & (shares <= 0.0)] = 0.0
        held_outside = outside & (shares > 0.0) & (w > 0.0)
        if np.any(held_outside):
          total_equity = self.account.asset
          if total_equity <= 0.0:
            w[held_outside] = 0.0
          else:
            idx_h = np.flatnonzero(held_outside)
            mark_p = self.account.mark_prices_array()[idx_h]
            curr_w = (shares[idx_h] * mark_p) / total_equity
            w[idx_h] = np.minimum(w[idx_h], curr_w)

    self.target_weights[self.dp.i] = w
    self._has_target_weights[self.dp.i] = True

  def rebalance_objs(self, target_weights: dict[str, float], tolerance: float | None = None):
    """Rebalance portfolio using a mapping of {symbol: weight}.

    Symbols not present in target_weights are assigned a target weight of 0.0.
    Symbols not in the pre-loaded pool raise a ValueError (Fail-Fast).

    Args:
      target_weights: dict mapping stock symbol to target weight (e.g. {"SH600000": 0.05}).
      tolerance: optional tolerance band (e.g. 0.01 for 1%).
    """
    weights = np.zeros(self.dp.groups, dtype=np.float64)
    missing = []
    for sym, weight in target_weights.items():
      g = self._obj_to_group.get(sym)
      if g is not None:
        weights[g] = max(0.0, float(weight))
      else:
        missing.append(sym)
    if missing:
      raise ValueError(
        f"rebalance_objs: {len(missing)} obj(s) not in pre-loaded pool: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
        "Ensure all target symbols are included in the pre-loaded pool via -s/--objs."
      )
    self.rebalance(weights, tolerance=tolerance)


class PortfolioRunner:
  """Runner for multi-stock shared-cash portfolio strategies."""

  def __init__(
    self,
    account: PortfolioAccount,
    /,
    exec_mode: str = "close",
    check_limit: bool = True,
    tolerance: float = 0.0,
  ):
    if exec_mode not in ("close", "next_open"):
      raise ValueError(f"unsupported exec_mode: {exec_mode}, expected 'close' or 'next_open'")
    if tolerance > 0.0:
      account.tolerance = float(tolerance)
    account.exec_mode = exec_mode
    self.ctx = PortfolioContext(account)
    self.exec_mode = exec_mode
    self.check_limit = check_limit
    self.diagnostics: dict | None = None

  def run(self, strategy: Strategy):
    dp = self.ctx.dp
    acc = self.ctx.account
    if dp.groups == 0 or dp.bars == 0:
      return

    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      strategy.prepare(self.ctx)
    dp.reset()

    prices = np.ascontiguousarray(dp.all("price"), dtype=np.float64)
    self.ctx._prices_contig = prices
    if self.exec_mode == "next_open":
      if "raw_open" in dp.data:
        deal_prices_all = np.ascontiguousarray(dp.all("raw_open"), dtype=np.float64)
      elif "open" in dp.data:
        deal_prices_all = np.ascontiguousarray(dp.all("open"), dtype=np.float64)
      else:
        deal_prices_all = prices
    else:
      deal_prices_all = prices

    limited_all = (
      np.ascontiguousarray(np.nan_to_num(dp.all("limited"), nan=0.0), dtype=np.int32)
      if (self.check_limit and "limited" in dp.data)
      else None
    )
    fw_prices_all = (
      np.ascontiguousarray(dp.all("close"), dtype=np.float64)
      if "close" in dp.data
      else None
    )
    div_all = (
      np.ascontiguousarray(dp.all("dividend"), dtype=np.float64)
      if "dividend" in dp.data
      else None
    )
    trans_all = (
      np.ascontiguousarray(dp.all("transfer_shares"), dtype=np.float64)
      if "transfer_shares" in dp.data
      else None
    )
    rs_all = (
      np.ascontiguousarray(dp.all("right_shares"), dtype=np.float64)
      if "right_shares" in dp.data
      else None
    )
    rp_all = (
      np.ascontiguousarray(dp.all("right_price"), dtype=np.float64)
      if "right_price" in dp.data
      else None
    )

    engine = PortfolioEngine(
      dp.groups,
      dp.bars,
      acc.tradable_mask,
      initial_cash=float(acc.initial_cash),
      unit=int(acc.unit),
      slippage=float(acc.slippage),
      tolerance=float(acc.tolerance),
      fees=acc.fee.config,
      dividend=div_all,
      transfer_shares=trans_all,
      right_shares=rs_all,
    )
    acc._engine = engine
    offsets = self.ctx._group_offsets
    ts_list = acc.dates.tolist()
    warmup_b = int(getattr(dp, "warmup_bars", 0))
    is_next_open = self.exec_mode == "next_open"
    has_tw = self.ctx._has_target_weights
    tw_all = self.ctx.target_weights
    tol_all = self.ctx.target_tolerances
    sell_all = self.ctx.sell_percents
    buy_all = self.ctx.buy_percents
    out_actions = acc.actions
    out_action_shares = acc.action_shares
    out_cash = acc.portfolio_cash
    out_assets = acc.portfolio_assets
    out_ror = acc.portfolio_ror

    # Initialize bar 0 mark-to-market
    engine.update_mark_to_market_strided(0, prices, out_cash, out_assets, out_ror)

    bar_ctx = al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)
    self.ctx._in_execute = True
    try:
      with bar_ctx:
        for i in dp:
          bar_ctx.end = i + 1
          bar_ctx.apply()
          engine.apply_corporate_actions(i, div_all, trans_all, rs_all, rp_all)
          strategy.execute(self.ctx)

          sig_i = i - 1 if is_next_open else i
          skip_trade = sig_i < 0 or i < warmup_b
          if not skip_trade and has_tw[sig_i]:
            tw_bar = tw_all[sig_i]
            step_tol = float(tol_all[sig_i])
            tol_val = None if np.isnan(step_tol) else step_tol
          else:
            tw_bar = None
            tol_val = None

          engine.step_bar(
            i,
            sig_i if sig_i >= 0 else 0,
            ts_list[i] if i < len(ts_list) else i,
            skip_trade,
            sell_all,
            buy_all,
            deal_prices_all,
            prices,
            out_actions,
            out_action_shares,
            out_cash,
            out_assets,
            out_ror,
            target_weights_bar=tw_bar,
            tolerance=tol_val,
            limited_all=limited_all,
            universe_mask=self.ctx._universe_mask,
            raw_prices_all=prices,
            fw_prices_all=fw_prices_all,
          )
    finally:
      self.ctx._in_execute = False
      self.ctx._rors_bar = -1
      self.ctx._rors_cache = None
      self.ctx._shares_bar = -1
      self.ctx._shares_cache = None
      acc._hydrate_from_engine()
      acc._engine = None

    acc.pending_signals = self._extract_pending_signals(dp, acc, offsets, prices)

    # Post-execution phase: finalize hook with full bar range
    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      diag = strategy.finalize(self.ctx)
      if diag is not None:
        self.diagnostics = diag
        setattr(strategy, "diagnostics", diag)
        setattr(acc, "diagnostics", diag)

  def _extract_pending_signals(
    self, dp: DataProvider, acc: PortfolioAccount, offsets: np.ndarray, prices: np.ndarray
  ) -> list[dict]:
    """Extract unexecuted signals generated on the final bar when exec_mode == 'next_open'."""
    if self.exec_mode != "next_open" or dp.bars <= 0:
      return []
    last_i = dp.bars - 1
    warmup_b = getattr(dp, "warmup_bars", 0)
    if last_i < warmup_b:
      return []

    objs = dp.objs
    idx_sig = offsets + last_i
    sell_p = self.ctx.sell_percents[idx_sig]
    buy_p = self.ctx.buy_percents[idx_sig]
    tgt_w = self.ctx.target_weights[last_i]
    total_assets = float(acc.portfolio_assets[last_i])

    pending: list[dict] = []
    explicit_sell_groups: set[int] = set()
    for g in range(dp.groups):
      if not acc.tradable_mask[g]:
        continue
      if sell_p[g] > 0 and acc.positions[g].shares > 0:
        ref_p = float(prices[idx_sig[g]])
        est_shares = acc.positions[g].shares * float(sell_p[g])
        if sell_p[g] < 1.0:
          est_shares = float(int(est_shares) - (int(est_shares) % acc.unit))
        if est_shares > 0:
          explicit_sell_groups.add(g)
          pending.append({
            "obj": objs[g] if g < len(objs) else str(g),
            "action": "SELL",
            "ref_price": ref_p,
            "est_shares": float(est_shares),
            "target_weight": None,
            "sell_percent": float(sell_p[g]),
          })

    if not np.all(np.isnan(tgt_w)):
      step_tol = self.ctx.target_tolerances[last_i]
      tol = float(acc.tolerance) if np.isnan(step_tol) else float(step_tol)
      for g in range(dp.groups):
        if not acc.tradable_mask[g] or g in explicit_sell_groups:
          continue
        tw = float(tgt_w[g])
        if np.isnan(tw):
          continue
        ref_p = float(prices[idx_sig[g]])
        pos_shares = float(acc.positions[g].shares)
        cur_w = (pos_shares * ref_p / total_assets) if (total_assets > 0 and ref_p > 0) else 0.0
        if abs(tw - cur_w) <= tol and not (tw == 0.0 and pos_shares > 0):
          continue
        if tw == 0.0 and pos_shares <= 0:
          continue
        action = "BUY" if tw > cur_w else "SELL"
        est_shares = 0.0
        if ref_p > 0 and total_assets > 0:
          delta_val = abs(tw - cur_w) * total_assets
          raw_sh = int(delta_val / ref_p)
          est_shares = float(raw_sh - (raw_sh % acc.unit)) if tw > 0 else pos_shares
        pending.append({
          "obj": objs[g] if g < len(objs) else str(g),
          "action": action,
          "ref_price": ref_p,
          "est_shares": float(est_shares),
          "target_weight": tw,
          "current_weight": float(cur_w),
        })
      return pending

    for g in range(dp.groups):
      if not acc.tradable_mask[g]:
        continue
      if buy_p[g] > 0:
        pending.append({
          "obj": objs[g] if g < len(objs) else str(g),
          "action": "BUY",
          "ref_price": float(prices[idx_sig[g]]),
          "est_shares": None,
          "target_weight": None,
          "buy_percent": float(buy_p[g]),
        })
    return pending


class SelectorStrategy(PortfolioStrategy):
  """Adapter that runs a Selector (or selection callable) periodically during a portfolio backtest."""

  def __init__(
    self,
    selector: Selector | Callable[[Panel], list[str]],
    /,
    rebalance_period: int = 20,
    min_bars: int = 1,
    max_holdings: int | None = None,
  ):
    super().__init__()
    self.selector = selector
    self.rebalance_period = max(1, int(rebalance_period))
    self.min_bars = max(1, int(min_bars))
    self.max_holdings = max_holdings
    self._full_panel: Panel | None = None

  def prepare(self, ctx: Context):
    self._full_panel = ctx.panel()

  def _slice_panel(self, end_bar: int) -> Panel:
    assert self._full_panel is not None
    return self._full_panel.slice(0, end_bar)

  def execute(self, ctx: PortfolioContext):  # type: ignore[override]
    i = ctx.bar
    if i < self.min_bars or (i - self.min_bars) % self.rebalance_period != 0:
      return

    sub_panel = self._slice_panel(i + 1)
    with sub_panel.ctx():
      if isinstance(self.selector, Selector):
        fin_only_year = getattr(ctx.dp, "fin_only_year", False)
        financial_frame = getattr(ctx.dp, "financial_frame", None)
        dp_slice = _HistoricalPanelSelectorDataProvider(
          sub_panel, fin_only_year=fin_only_year, financial_frame=financial_frame
        )
        selected = self.selector.execute(dp_slice, list(sub_panel.objs))
      else:
        selected = self.selector(sub_panel)

    if self.max_holdings is not None and len(selected) > self.max_holdings:
      selected = selected[: self.max_holdings]

    weights = np.zeros(sub_panel.n_objs, dtype=np.float64)
    if len(selected) > 0:
      missing = [s for s in selected if s not in sub_panel._index]
      if missing:
        raise ValueError(
          f"SelectorStrategy: {len(missing)} selected obj(s) not in pre-loaded pool: "
          f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
          "Ensure Selector only selects from the passed `stocks` pool during backtest."
        )
      w = 1.0 / len(selected)
      for s in selected:
        weights[sub_panel.pick(s)] = w

    ctx.rebalance(weights)
