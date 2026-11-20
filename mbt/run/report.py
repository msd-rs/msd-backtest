# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

import datetime
import dataclasses
from typing import Any
from mbt import Account
from mbt import DataProvider
import polars as pl
import numpy as np


def _ffill_nan_zero_1d(arr: np.ndarray) -> np.ndarray:
  """Forward-fill NaNs and replace leading NaNs with 0.0 without Polars overhead."""
  if arr.size == 0 or not np.isnan(arr).any():
    return arr
  mask = np.isnan(arr)
  idx = np.where(~mask, np.arange(arr.size), 0)
  np.maximum.accumulate(idx, out=idx)
  out = arr[idx].copy()
  out[np.isnan(out)] = 0.0
  return out


def _ffill_nan_zero_2d(arr: np.ndarray) -> np.ndarray:
  """Row-wise forward-fill NaNs and replace leading NaNs with 0.0 on 2D (G, T) arrays."""
  if arr.size == 0 or not np.isnan(arr).any():
    return arr
  mask = np.isnan(arr)
  g_n, t_n = arr.shape
  idx = np.where(~mask, np.broadcast_to(np.arange(t_n), (g_n, t_n)), 0)
  np.maximum.accumulate(idx, axis=1, out=idx)
  rows = np.arange(g_n)[:, None]
  out = arr[rows, idx].copy()
  out[np.isnan(out)] = 0.0
  return out


def _calc_shares_1d(actions: np.ndarray, action_shares: np.ndarray) -> np.ndarray:
  """Compute per-bar held shares from actions and action_shares using vectorized cumsum."""
  if actions.size == 0:
    return np.zeros(0, dtype=np.float64)
  delta = np.where(actions == 1, action_shares, np.where(actions == 2, -action_shares, 0.0))
  shares = np.cumsum(delta)
  if shares.min() >= 0.0:
    return shares
  # Clamp sequentially only if cumulative delta ever goes negative
  shares = np.zeros(len(actions), dtype=np.float64)
  curr = 0.0
  for t in range(len(actions)):
    act = actions[t]
    if act == 1:
      curr += action_shares[t]
    elif act == 2:
      curr -= action_shares[t]
    if curr < 0.0:
      curr = 0.0
    shares[t] = curr
  return shares


@dataclasses.dataclass
class Report:
  obj: str
  date: np.ndarray 
  ror: np.ndarray
  ror_hold: np.ndarray
  ror_original: np.ndarray
  actions: np.ndarray
  price: np.ndarray
  close: np.ndarray
  fw_price: np.ndarray
  bw_price: np.ndarray
  alpha: float
  beta: float
  sharpe: float
  information_ratio: float
  ir: float
  sortino: float
  treynor: float
  max_drawdown: float
  calmar: float
  win_rate: float
  avg_win: float
  avg_loss: float
  profit_factor: float
  profit: float
  profit_original: float
  benchmark_profit: float
  fee_rate: float
  trade_count: int
  rebalance_count: int = 0
  diagnostics: dict | None = None
  benchmark_ror: np.ndarray | None = None
  benchmark_obj: str = ""
  trades: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  attribution: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  account: Any = dataclasses.field(default=None, repr=False)
  records: list[Any] = dataclasses.field(default_factory=list, repr=False)
  dp: Any = dataclasses.field(default=None, repr=False)
  pool_size: int = 0
  traded_objs_count: int = 0
  

  def __init__(self, obj: str, dp: DataProvider , account: Account ):
    if len(obj) == 0:
      return
    self.obj = obj
    obj_idx = dp.objs.index(obj)
    s = obj_idx * dp.bars
    e = s + dp.bars
    self._obj_idx = obj_idx
    w = max(0, int(getattr(dp, "warmup_bars", 0)))

    if w > 0 and (e - s) > w:
      self.date = dp.dates[s + w : e]
      assets = dp.slice("assets", s)[w:]
      base_a = assets[0] if (len(assets) > 0 and assets[0] > 0) else account.initial_cash
      self.ror = np.round((assets / base_a) - 1.0, 4)
      self.ror_hold = self.ror.copy()
      self.ror_original = np.round(uniform_ror(dp.slice("bw_price", s)[w:]), 4) if "bw_price" in dp.data else np.zeros_like(self.ror)
      self.actions = dp.slice("actions", s)[w:]
      self.price = dp.slice("price", s)[w:]
      self.close = self.price
      self.fw_price = np.round(dp.slice("fw_price", s)[w:], 3) if "fw_price" in dp.data else self.price
      self.bw_price = np.round(dp.slice("bw_price", s)[w:], 3) if "bw_price" in dp.data else self.price
    else:
      self.date = dp.dates[s:e]
      self.ror = np.round(dp.slice("ror", s), 4)
      self.ror_hold = np.round(dp.slice("ror_hold", s), 4)
      self.ror_original = np.round(uniform_ror(dp.slice("bw_price", s)), 4) if "bw_price" in dp.data else np.zeros_like(self.ror)
      self.actions = dp.slice("actions", s)
      self.price = dp.slice("price", s)
      self.close = self.price # For compatibility with tests
      self.fw_price = np.round(dp.slice("fw_price", s), 3) if "fw_price" in dp.data else self.price
      self.bw_price = np.round(dp.slice("bw_price", s), 3) if "bw_price" in dp.data else self.price

    # Clean NaNs by forward filling and filling leading NaNs with 0
    self.ror = _ffill_nan_zero_1d(self.ror)
    self.ror_hold = _ffill_nan_zero_1d(self.ror_hold)
    self.ror_original = _ffill_nan_zero_1d(self.ror_original)

    self.calculate_metrics(dp, account)
    

  @staticmethod
  def from_dict(obj: str, d: dict) -> "Report":
    r = Report('', None, None) # type: ignore
    r.obj = obj
    r.date = d.get('date', np.array([], dtype='int64'))
    r.ror = np.array(d.get('ror', []), dtype='float64')
    r.ror_hold = np.array(d.get('ror_hold', []), dtype='float64')
    r.ror_original = np.array(d.get('ror_original', []), dtype='float64')
    r.actions = np.array(d.get('actions', []), dtype='int8')
    r.price = np.array(d.get('price', []), dtype='float64')
    r.close = r.price
    r.fw_price = np.array(d.get('fw_price', []), dtype='float64')
    r.bw_price = np.array(d.get('bw_price', []), dtype='float64')
    
    r.alpha = d.get('alpha', 0.0)
    r.beta = d.get('beta', 0.0)
    r.sharpe = d.get('sharpe', 0.0)
    r.information_ratio = d.get('information_ratio', 0.0)
    r.ir = d.get('ir', 0.0)
    r.sortino = d.get('sortino', 0.0)
    r.treynor = d.get('treynor', 0.0)
    r.max_drawdown = d.get('max_drawdown', 0.0)
    r.calmar = d.get('calmar', 0.0)
    r.win_rate = d.get('win_rate', 0.0)
    r.avg_win = d.get('avg_win', 0.0)
    r.avg_loss = d.get('avg_loss', 0.0)
    r.profit_factor = d.get('profit_factor', 0.0)
    r.profit = d.get('profit', 0.0)
    r.profit_original = d.get('profit_original', 0.0)
    r.benchmark_profit = d.get('benchmark_profit', 0.0)
    r.fee_rate = d.get('fee_rate', 0.0)
    r.trade_count = d.get('trade_count', 0)
    r.rebalance_count = d.get('rebalance_count', r.trade_count)
    r.daily_win_rate = d.get('daily_win_rate', r.win_rate)
    r.trade_win_rate = d.get('trade_win_rate', r.win_rate)
    r.trade_avg_win = d.get('trade_avg_win', r.avg_win)
    r.trade_avg_loss = d.get('trade_avg_loss', r.avg_loss)
    r.trade_profit_factor = d.get('trade_profit_factor', r.profit_factor)
    r.trades = d.get('trades', [])
    r.attribution = d.get('attribution', [])
    r.diagnostics = d.get('diagnostics', None)
    b_ror = d.get('benchmark_ror', None)
    r.benchmark_ror = np.array(b_ror, dtype='float64') if b_ror is not None else None
    r.benchmark_obj = d.get('benchmark_obj', '')

    return r    
    


  def calculate_metrics(self, dp: DataProvider, account: Account):
    w = max(0, int(getattr(dp, "warmup_bars", 0)))
    bench_bw = dp.slice("bw_price", 0)
    if w > 0 and len(bench_bw) > w:
      bench_bw = bench_bw[w:]
    benchmark = uniform_ror(bench_bw)
    self.benchmark_ror = np.round(_ffill_nan_zero_1d(benchmark), 4)
    if hasattr(dp, "objs") and len(dp.objs) > 0:
      self.benchmark_obj = str(dp.objs[0])

    # 1. Convert cumulative returns to daily returns
    strategy_ratio = 1.0 + self.ror
    benchmark_ratio = 1.0 + benchmark

    if len(strategy_ratio) > 1:
      strategy_daily = np.diff(strategy_ratio) / strategy_ratio[:-1]
      benchmark_daily = np.diff(benchmark_ratio) / benchmark_ratio[:-1]
    else:
      strategy_daily = np.array([], dtype=np.float64)
      benchmark_daily = np.array([], dtype=np.float64)

    # 2. Portfolio/Strategy metrics calculations
    if len(strategy_daily) >= 2:
      # Sharpe
      mean_strat = np.mean(strategy_daily)
      std_strat = np.std(strategy_daily, ddof=1)
      sharpe = (mean_strat / std_strat * np.sqrt(252)) if std_strat != 0.0 else 0.0

      # Beta
      cov = np.cov(strategy_daily, benchmark_daily)
      var_bench = cov[1, 1]
      beta = (cov[0, 1] / var_bench) if var_bench != 0.0 else 0.0

      # Alpha
      mean_bench = np.mean(benchmark_daily)
      alpha = (mean_strat - beta * mean_bench) * 252

      # Information Ratio (IR)
      active_daily = strategy_daily - benchmark_daily
      mean_active = np.mean(active_daily)
      std_active = np.std(active_daily, ddof=1)
      ir = (mean_active / std_active * np.sqrt(252)) if std_active != 0.0 else 0.0

      # Sortino
      downside_daily = np.minimum(strategy_daily, 0.0)
      std_downside = np.std(downside_daily, ddof=1)
      sortino = (mean_strat / std_downside * np.sqrt(252)) if std_downside != 0.0 else 0.0

      # Treynor
      treynor = (mean_strat * 252 / beta) if beta != 0.0 else 0.0
    else:
      sharpe = 0.0
      beta = 0.0
      alpha = 0.0
      ir = 0.0
      sortino = 0.0
      treynor = 0.0

    # Max Drawdown
    if len(self.ror) > 0:
      equity = 1.0 + self.ror
      peaks = np.maximum.accumulate(equity)
      drawdowns = (equity - peaks) / peaks
      max_drawdown = np.min(drawdowns)
    else:
      max_drawdown = 0.0

    # Calmar
    if max_drawdown != 0.0 and len(strategy_daily) >= 2:
      calmar = (np.mean(strategy_daily) * 252) / abs(max_drawdown)
    else:
      calmar = 0.0

    # 3. Trade-level metrics
    obj_idx = getattr(self, "_obj_idx", None)
    if obj_idx is None:
      obj_idx = dp.objs.index(self.obj)
    s = obj_idx * dp.bars
    e = s + dp.bars
    raw_action_shares = dp.slice("action_shares", s) if "action_shares" in dp.data else np.zeros(dp.bars)
    action_shares = raw_action_shares[w:] if (w > 0 and len(raw_action_shares) > w) else raw_action_shares

    shares = _calc_shares_1d(self.actions, action_shares)

    # Identify holding periods where shares > 0
    holding = shares > 0
    padded = np.concatenate(([False], holding, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    trade_profits = []
    trade_returns = []

    raw_assets = dp.all("assets")[s:e]
    assets = raw_assets[w:] if (w > 0 and len(raw_assets) > w) else raw_assets

    for s_idx, e_idx in zip(starts, ends):
      base_idx = s_idx - 1 if s_idx > 0 else 0
      base_asset = assets[base_idx]

      final_idx = min(e_idx + 1, len(assets) - 1)
      final_asset = assets[final_idx]

      profit = final_asset - base_asset
      trade_profits.append(profit)

      ret = (profit / base_asset) if base_asset != 0.0 else 0.0
      trade_returns.append(ret)

    if trade_profits:
      wins = [p for p in trade_profits if p > 0]
      losses = [p for p in trade_profits if p < 0]

      wins_ret = [r for r in trade_returns if r > 0]
      losses_ret = [r for r in trade_returns if r < 0]

      win_rate = len(wins) / len(trade_profits)
      avg_win = np.mean(wins_ret) if wins_ret else 0.0
      avg_loss = np.mean(losses_ret) if losses_ret else 0.0

      sum_loss = abs(sum(losses))
      profit_factor = sum(wins) / sum_loss if sum_loss != 0.0 else 0.0
    else:
      win_rate = 0.0
      avg_win = 0.0
      avg_loss = 0.0
      profit_factor = 0.0

    # 4. Store all metrics as attributes
    self.alpha = 0.0 if np.isnan(alpha) else np.round(alpha, 4)
    self.beta = 0.0 if np.isnan(beta) else np.round(beta, 4)
    self.sharpe = 0.0 if np.isnan(sharpe) else np.round(sharpe, 4)
    self.information_ratio = 0.0 if np.isnan(ir) else np.round(ir, 4)
    self.ir = self.information_ratio # For compatibility with tests
    self.sortino = 0.0 if np.isnan(sortino) else np.round(sortino, 4)
    self.treynor = 0.0 if np.isnan(treynor) else np.round(treynor, 4)
    self.max_drawdown = 0.0 if np.isnan(max_drawdown) else np.round(max_drawdown, 4)
    self.calmar = 0.0 if np.isnan(calmar) else np.round(calmar, 4)

    self.win_rate = 0.0 if np.isnan(win_rate) else np.round(win_rate, 4)
    self.avg_win = 0.0 if np.isnan(avg_win) else np.round(avg_win, 4)
    self.avg_loss = 0.0 if np.isnan(avg_loss) else np.round(avg_loss, 4)
    self.profit_factor = 0.0 if np.isnan(profit_factor) else np.round(profit_factor, 4)
    self.daily_win_rate = self.win_rate
    self.daily_avg_win = self.avg_win
    self.daily_avg_loss = self.avg_loss
    self.daily_profit_factor = self.profit_factor
    self.trade_win_rate = self.win_rate
    self.trade_avg_win = self.avg_win
    self.trade_avg_loss = self.avg_loss
    self.trade_profit_factor = self.profit_factor
    self.profit = self.ror[-1] if self.ror.size > 0 else 0.0
    self.profit_original = self.ror_original[-1] if self.ror_original.size > 0 else 0.0
    self.benchmark_profit = (
      0.0
      if (len(benchmark) == 0 or np.isnan(benchmark[-1]))
      else float(np.round(benchmark[-1], 4))
    )
    self.fee_rate = np.round(account.deducted / account.initial_cash, 4)
    self.trade_count = len(trade_profits)
    self.rebalance_count = self.trade_count
    

  def metrics(self, with_obj: bool = False) -> dict:
    d = {
      "obj": self.obj,
      "profit": self.profit,
      "profit_original": self.profit_original,
      "benchmark_profit": self.benchmark_profit,
      "alpha": self.alpha,
      "beta": self.beta,
      "sharpe": self.sharpe,
      "information_ratio": self.information_ratio,
      "sortino": self.sortino,
      "treynor": self.treynor,
      "max_drawdown": self.max_drawdown,
      "calmar": self.calmar,
      "win_rate": self.win_rate,
      "avg_win": self.avg_win,
      "avg_loss": self.avg_loss,
      "profit_factor": self.profit_factor,
      "daily_win_rate": getattr(self, "daily_win_rate", self.win_rate),
      "trade_win_rate": getattr(self, "trade_win_rate", self.win_rate),
      "trade_avg_win": getattr(self, "trade_avg_win", self.avg_win),
      "trade_avg_loss": getattr(self, "trade_avg_loss", self.avg_loss),
      "trade_profit_factor": getattr(self, "trade_profit_factor", self.profit_factor),
      "fee_rate": self.fee_rate,
      "trade_count": self.trade_count,
      "rebalance_count": self.rebalance_count,
    }
    if not with_obj:
      del d["obj"]
    return d

  def detailed(self, with_obj: bool = False) -> dict:
    d = {
      "obj": [self.obj] * len(self.date),
      "date": self.date,
      "ror": self.ror,
      "ror_hold": self.ror_hold,
      "ror_original": self.ror_original,
      "actions": self.actions,
    }
    if not with_obj:
      del d["obj"]
    return d

def uniform_ror(a: np.ndarray) -> np.ndarray:
  base = a[0]
  if base == 0:
    return a
  return ((a - base) / base)


def _clean_round4_list(arr: np.ndarray) -> list[float]:
  return np.where(np.isnan(arr), 0.0, np.round(arr, 4)).tolist()


def build_report(
  dp: DataProvider,
  accounts: list[Account],
  diagnostics: dict | None = None,
) -> list[Report]:
  objs = getattr(dp, "objs", [])
  g_count = len(objs)
  bars = dp.bars
  if g_count == 0 or bars == 0:
    reports = [Report(obj, dp, accounts[i]) for i, obj in enumerate(objs)]
    if diagnostics is not None:
      for rep in reports:
        rep.diagnostics = diagnostics
    return reports

  w = max(0, int(getattr(dp, "warmup_bars", 0)))
  slice_start = w if (w > 0 and bars > w) else 0
  eff_bars = bars - slice_start

  dates_2d = dp.dates.reshape(g_count, bars)[:, slice_start:]
  prices_2d = dp.all("price").reshape(g_count, bars)[:, slice_start:]
  actions_2d = dp.all("actions").reshape(g_count, bars)[:, slice_start:]
  action_shares_2d = (
    dp.all("action_shares").reshape(g_count, bars)[:, slice_start:]
    if "action_shares" in dp.data
    else np.zeros((g_count, eff_bars), dtype=np.float64)
  )
  assets_2d = dp.all("assets").reshape(g_count, bars)[:, slice_start:]
  fw_2d = (
    np.round(dp.all("fw_price").reshape(g_count, bars)[:, slice_start:], 3)
    if "fw_price" in dp.data
    else prices_2d
  )
  bw_raw_2d = (
    dp.all("bw_price").reshape(g_count, bars)[:, slice_start:]
    if "bw_price" in dp.data
    else None
  )
  bw_2d = np.round(bw_raw_2d, 3) if bw_raw_2d is not None else prices_2d

  init_cashes = np.array([float(acc.initial_cash) for acc in accounts], dtype=np.float64)
  deducteds = np.array([float(acc.deducted) for acc in accounts], dtype=np.float64)

  if slice_start > 0:
    base_a = np.where((eff_bars > 0) & (assets_2d[:, 0] > 0), assets_2d[:, 0], init_cashes)
    ror_2d = _ffill_nan_zero_2d(np.round((assets_2d / base_a[:, None]) - 1.0, 4))
    ror_hold_2d = ror_2d
  else:
    ror_2d = _ffill_nan_zero_2d(np.round(dp.all("ror").reshape(g_count, bars), 4))
    ror_hold_2d = _ffill_nan_zero_2d(np.round(dp.all("ror_hold").reshape(g_count, bars), 4))

  if bw_raw_2d is not None:
    b0 = bw_raw_2d[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
      raw_orig = np.where(
        b0[:, None] == 0.0,
        bw_raw_2d,
        (bw_raw_2d - b0[:, None]) / b0[:, None],
      )
    ror_original_2d = _ffill_nan_zero_2d(np.round(raw_orig, 4))
  else:
    ror_original_2d = np.zeros((g_count, eff_bars), dtype=np.float64)

  # Compute benchmark metrics once across all objects
  bench_bw = dp.slice("bw_price", 0)[slice_start:] if "bw_price" in dp.data else prices_2d[0]
  benchmark = uniform_ror(bench_bw) if len(bench_bw) > 0 else np.zeros(0, dtype=np.float64)
  benchmark_ror = np.round(_ffill_nan_zero_1d(benchmark), 4)
  benchmark_profit = (
    0.0
    if (len(benchmark) == 0 or np.isnan(benchmark[-1]))
    else float(np.round(benchmark[-1], 4))
  )
  benchmark_obj = str(objs[0])

  if eff_bars >= 3:
    with np.errstate(divide="ignore", invalid="ignore"):
      strategy_ratio = 1.0 + ror_2d
      strategy_daily = np.diff(strategy_ratio, axis=1) / strategy_ratio[:, :-1]
      benchmark_ratio = 1.0 + benchmark
      benchmark_daily = np.diff(benchmark_ratio) / benchmark_ratio[:-1]

      mean_strat = np.mean(strategy_daily, axis=1)
      std_strat = np.std(strategy_daily, axis=1, ddof=1)
      sharpe_arr = np.where(std_strat != 0.0, mean_strat / std_strat * np.sqrt(252.0), 0.0)

      mean_bench = float(np.mean(benchmark_daily))
      var_bench = float(np.var(benchmark_daily, ddof=1))
      n_d = strategy_daily.shape[1]
      cov_sb = (
        np.sum((strategy_daily - mean_strat[:, None]) * (benchmark_daily[None, :] - mean_bench), axis=1)
        / (n_d - 1)
      )
      beta_arr = cov_sb / var_bench if var_bench != 0.0 else np.zeros(g_count, dtype=np.float64)
      alpha_arr = (mean_strat - beta_arr * mean_bench) * 252.0

      active_daily = strategy_daily - benchmark_daily[None, :]
      mean_active = np.mean(active_daily, axis=1)
      std_active = np.std(active_daily, axis=1, ddof=1)
      ir_arr = np.where(std_active != 0.0, mean_active / std_active * np.sqrt(252.0), 0.0)

      downside_daily = np.minimum(strategy_daily, 0.0)
      std_downside = np.std(downside_daily, axis=1, ddof=1)
      sortino_arr = np.where(std_downside != 0.0, mean_strat / std_downside * np.sqrt(252.0), 0.0)
      treynor_arr = np.where(beta_arr != 0.0, mean_strat * 252.0 / beta_arr, 0.0)
  else:
    mean_strat = np.zeros(g_count, dtype=np.float64)
    sharpe_arr = np.zeros(g_count, dtype=np.float64)
    beta_arr = np.zeros(g_count, dtype=np.float64)
    alpha_arr = np.zeros(g_count, dtype=np.float64)
    ir_arr = np.zeros(g_count, dtype=np.float64)
    sortino_arr = np.zeros(g_count, dtype=np.float64)
    treynor_arr = np.zeros(g_count, dtype=np.float64)

  if eff_bars > 0:
    equity = 1.0 + ror_2d
    peaks = np.maximum.accumulate(equity, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
      mdd_arr = np.min((equity - peaks) / peaks, axis=1)
  else:
    mdd_arr = np.zeros(g_count, dtype=np.float64)

  if eff_bars >= 3:
    with np.errstate(divide="ignore", invalid="ignore"):
      calmar_arr = np.where(mdd_arr != 0.0, mean_strat * 252.0 / np.abs(mdd_arr), 0.0)
  else:
    calmar_arr = np.zeros(g_count, dtype=np.float64)

  # Vectorized shares calculation across all groups
  delta_2d = np.where(actions_2d == 1, action_shares_2d, np.where(actions_2d == 2, -action_shares_2d, 0.0))
  shares_2d = np.cumsum(delta_2d, axis=1)
  neg_rows = np.flatnonzero(shares_2d.min(axis=1) < 0.0) if eff_bars > 0 else np.empty(0, dtype=np.int64)
  for g in neg_rows:
    shares_2d[g] = _calc_shares_1d(actions_2d[g], action_shares_2d[g])

  holding_2d = shares_2d > 0
  win_rate_arr = np.zeros(g_count, dtype=np.float64)
  avg_win_arr = np.zeros(g_count, dtype=np.float64)
  avg_loss_arr = np.zeros(g_count, dtype=np.float64)
  pf_arr = np.zeros(g_count, dtype=np.float64)
  trade_counts = [0] * g_count

  if eff_bars > 0 and holding_2d.any():
    padded_2d = np.zeros((g_count, eff_bars + 2), dtype=np.int8)
    padded_2d[:, 1:-1] = holding_2d
    diff_2d = np.diff(padded_2d, axis=1)
    s_rows, s_cols = np.where(diff_2d == 1)
    _, e_cols = np.where(diff_2d == -1)
    e_cols = e_cols - 1

    if s_rows.size > 0:
      base_cols = np.maximum(s_cols - 1, 0)
      final_cols = np.minimum(e_cols + 1, eff_bars - 1)
      base_a = assets_2d[s_rows, base_cols]
      profits = assets_2d[s_rows, final_cols] - base_a
      with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(base_a != 0.0, profits / base_a, 0.0)

      tc_arr = np.bincount(s_rows, minlength=g_count)
      trade_counts = tc_arr.tolist()
      win_pos = profits > 0
      loss_pos = profits < 0
      win_ret = rets > 0
      loss_ret = rets < 0

      win_cnt = np.bincount(s_rows, weights=win_pos.astype(np.float64), minlength=g_count)
      win_ret_cnt = np.bincount(s_rows, weights=win_ret.astype(np.float64), minlength=g_count)
      loss_ret_cnt = np.bincount(s_rows, weights=loss_ret.astype(np.float64), minlength=g_count)
      win_ret_sum = np.bincount(s_rows, weights=np.where(win_ret, rets, 0.0), minlength=g_count)
      loss_ret_sum = np.bincount(s_rows, weights=np.where(loss_ret, rets, 0.0), minlength=g_count)
      win_prof_sum = np.bincount(s_rows, weights=np.where(win_pos, profits, 0.0), minlength=g_count)
      loss_prof_sum = np.bincount(s_rows, weights=np.where(loss_pos, -profits, 0.0), minlength=g_count)

      with np.errstate(divide="ignore", invalid="ignore"):
        win_rate_arr = np.where(tc_arr > 0, win_cnt / tc_arr, 0.0)
        avg_win_arr = np.where(win_ret_cnt > 0, win_ret_sum / win_ret_cnt, 0.0)
        avg_loss_arr = np.where(loss_ret_cnt > 0, loss_ret_sum / loss_ret_cnt, 0.0)
        pf_arr = np.where(loss_prof_sum != 0.0, win_prof_sum / loss_prof_sum, 0.0)

  alpha_list = _clean_round4_list(alpha_arr)
  beta_list = _clean_round4_list(beta_arr)
  sharpe_list = _clean_round4_list(sharpe_arr)
  ir_list = _clean_round4_list(ir_arr)
  sortino_list = _clean_round4_list(sortino_arr)
  treynor_list = _clean_round4_list(treynor_arr)
  mdd_list = _clean_round4_list(mdd_arr)
  calmar_list = _clean_round4_list(calmar_arr)
  wr_list = _clean_round4_list(win_rate_arr)
  aw_list = _clean_round4_list(avg_win_arr)
  al_list = _clean_round4_list(avg_loss_arr)
  pf_list = _clean_round4_list(pf_arr)
  with np.errstate(divide="ignore", invalid="ignore"):
    fee_rate_list = _clean_round4_list(np.where(init_cashes > 0.0, deducteds / init_cashes, 0.0))
  profit_list = (ror_2d[:, -1].tolist() if eff_bars > 0 else [0.0] * g_count)
  profit_orig_list = (ror_original_2d[:, -1].tolist() if eff_bars > 0 else [0.0] * g_count)

  reports: list[Report] = []
  for i, obj in enumerate(objs):
    r = Report.__new__(Report)
    r.obj = obj
    r.date = dates_2d[i]
    r.ror = ror_2d[i]
    r.ror_hold = ror_hold_2d[i].copy() if slice_start > 0 else ror_hold_2d[i]
    r.ror_original = ror_original_2d[i]
    r.actions = actions_2d[i]
    r.price = prices_2d[i]
    r.close = r.price
    r.fw_price = fw_2d[i]
    r.bw_price = bw_2d[i]

    r.alpha = alpha_list[i]
    r.beta = beta_list[i]
    r.sharpe = sharpe_list[i]
    r.information_ratio = ir_list[i]
    r.ir = ir_list[i]
    r.sortino = sortino_list[i]
    r.treynor = treynor_list[i]
    r.max_drawdown = mdd_list[i]
    r.calmar = calmar_list[i]

    wr = wr_list[i]
    aw = aw_list[i]
    al_v = al_list[i]
    pf = pf_list[i]
    r.win_rate = wr
    r.avg_win = aw
    r.avg_loss = al_v
    r.profit_factor = pf
    r.daily_win_rate = wr
    r.daily_avg_win = aw
    r.daily_avg_loss = al_v
    r.daily_profit_factor = pf
    r.trade_win_rate = wr
    r.trade_avg_win = aw
    r.trade_avg_loss = al_v
    r.trade_profit_factor = pf

    r.profit = float(profit_list[i])
    r.profit_original = float(profit_orig_list[i])
    r.benchmark_profit = benchmark_profit
    r.fee_rate = fee_rate_list[i]
    tc = trade_counts[i]
    r.trade_count = tc
    r.rebalance_count = tc
    r.diagnostics = diagnostics
    r.benchmark_ror = benchmark_ror
    r.benchmark_obj = benchmark_obj
    reports.append(r)

  return reports


def _format_date(bar: int, ts: int, dates: np.ndarray | None = None) -> str:
  if dates is not None and 0 <= bar < len(dates):
    d = dates[bar]
    if isinstance(d, np.datetime64):
      return str(d)[:10]
    elif isinstance(d, (int, np.integer)):
      if d > 100000000:
        try:
          return datetime.datetime.fromtimestamp(int(d), tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        except Exception:
          pass
      return str(d)
    return str(d)[:10]
  if ts > 100000000:
    try:
      return datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    except Exception:
      pass
  return str(ts)


def extract_portfolio_trades(
  account=None,
  dp: DataProvider | None = None,
  records: list | None = None,
) -> list[dict]:
  """Extract holding cycles (调仓记录) from a PortfolioAccount or records list."""
  if records is None and account is not None:
    records = getattr(account, "records", None)
  if not records:
    return []

  dp_obj = dp or getattr(account, "dp", None)
  bars = getattr(account, "bars", getattr(dp_obj, "bars", 0))
  dates = dp_obj.dates[:bars] if dp_obj and hasattr(dp_obj, "dates") else None

  trades_by_obj: dict[str, list] = {}
  for r in records:
    trades_by_obj.setdefault(r.obj, []).append(r)

  cycles = []
  positions = getattr(account, "positions", None)

  for obj, recs in trades_by_obj.items():
    g = recs[0].group
    in_pos = False
    entry_bar = 0
    entry_ts = 0
    total_buy_amount = 0.0
    total_sell_amount = 0.0
    total_fees = 0.0
    curr_shares = 0.0

    for r in recs:
      if r.action == "BUY":
        if not in_pos or curr_shares <= 0:
          in_pos = True
          entry_bar = r.bar
          entry_ts = r.ts
          total_buy_amount = 0.0
          total_sell_amount = 0.0
          total_fees = 0.0
          curr_shares = 0.0
        total_buy_amount += r.amount
        total_fees += r.fee
        curr_shares += r.shares
      elif r.action == "SELL":
        total_sell_amount += r.amount
        total_fees += r.fee
        curr_shares = max(0.0, curr_shares - r.shares)

        if curr_shares <= 0:
          in_pos = False
          profit_amount = total_sell_amount - total_buy_amount - total_fees
          cost = total_buy_amount
          profit_rate = (profit_amount / cost) if cost > 0 else 0.0
          cycles.append({
            "obj": obj,
            "entry_bar": entry_bar,
            "entry_date": _format_date(entry_bar, entry_ts, dates),
            "exit_bar": r.bar,
            "exit_date": _format_date(r.bar, r.ts, dates),
            "profit_rate": profit_rate,
            "profit_amount": profit_amount,
          })
          total_buy_amount = 0.0
          total_sell_amount = 0.0
          total_fees = 0.0

    if in_pos and curr_shares > 0:
      if positions and g < len(positions) and positions[g].shares > 0:
        mark_val = positions[g].market_value + positions[g].dividends
      else:
        pos_price = recs[-1].deal_price if recs else 0.0
        mark_val = curr_shares * pos_price
      profit_amount = total_sell_amount + mark_val - total_buy_amount - total_fees
      cost = total_buy_amount
      profit_rate = (profit_amount / cost) if cost > 0 else 0.0
      cycles.append({
        "obj": obj,
        "entry_bar": entry_bar,
        "entry_date": _format_date(entry_bar, entry_ts, dates),
        "exit_bar": bars - 1 if bars > 0 else entry_bar,
        "exit_date": "Holding",
        "profit_rate": profit_rate,
        "profit_amount": profit_amount,
      })

  cycles.sort(key=lambda x: (x["entry_bar"], x["obj"]))
  return cycles


def compute_stock_attribution(
  trades: list[dict], initial_cash: float = 1000000.0
) -> list[dict]:
  """Aggregate trades by stock obj and calculate contribution metrics."""
  if not trades:
    return []

  by_obj: dict[str, list[dict]] = {}
  for t in trades:
    by_obj.setdefault(t["obj"], []).append(t)

  attrib = []
  for obj, recs in by_obj.items():
    cnt = len(recs)
    total_profit = sum(float(r.get("profit_amount", 0.0)) for r in recs)
    wins = [r for r in recs if float(r.get("profit_amount", 0.0)) > 0]
    win_rate = len(wins) / cnt if cnt > 0 else 0.0
    avg_ret = float(np.mean([float(r.get("profit_rate", 0.0)) for r in recs])) if cnt > 0 else 0.0
    capital_contrib = (total_profit / initial_cash) if initial_cash > 0 else 0.0
    attrib.append({
      "obj": obj,
      "trade_count": cnt,
      "win_rate": round(win_rate, 4),
      "total_profit": round(total_profit, 2),
      "capital_contribution": round(capital_contrib, 4),
      "avg_return": round(avg_ret, 4),
    })

  attrib.sort(key=lambda x: x["total_profit"], reverse=True)
  return attrib


def build_portfolio_report(
  dp: DataProvider,
  account,
  obj_name: str = "PORTFOLIO",
  diagnostics: dict | None = None,
) -> list[Report]:
  """Build a Report for a shared-cash PortfolioAccount."""
  r = Report("", None, None)  # type: ignore
  r.obj = obj_name
  r.account = account
  r.diagnostics = diagnostics or getattr(account, "diagnostics", None)
  r.records = getattr(account, "records", [])
  r.dp = dp
  bars = dp.bars
  w = max(0, int(getattr(dp, "warmup_bars", 0)))

  if w > 0 and bars > w:
    r.date = dp.dates[w:bars]
    base_asset = (
      account.portfolio_assets[w]
      if account.portfolio_assets[w] > 0
      else account.initial_cash
    )
    ror = np.round((account.portfolio_assets[w:] / base_asset) - 1.0, 4)
    r.records = [rec for rec in r.records if rec.bar >= w]
    eff_bars = bars - w
  else:
    r.date = dp.dates[:bars] if bars > 0 else np.array([], dtype="int64")
    ror = np.round(account.portfolio_ror, 4)
    base_asset = account.initial_cash
    eff_bars = bars

  r.ror = (
    pl.Series("", ror)
    .fill_nan(None)
    .fill_null(strategy="forward")
    .fill_null(0.0)
    .to_numpy()
  )
  r.ror_hold = r.ror.copy()

  # Compute equal-weight original buy-and-hold ror across tradable stocks
  tradable_mask = getattr(account, "tradable_mask", None)
  if tradable_mask is not None:
    tradable_indices = [g for g in range(dp.groups) if tradable_mask[g]]
  else:
    start_g = getattr(account, "tradable_start", 0)
    tradable_indices = list(range(start_g, dp.groups))

  bench_idx = getattr(account, "benchmark_idx", None)
  if bench_idx is None:
    bench_obj = getattr(account, "benchmark_obj", None)
    if bench_obj is not None and bench_obj in dp.objs:
      bench_idx = dp.objs.index(bench_obj)
    elif tradable_mask is not None and not tradable_mask[0]:
      bench_idx = 0
    else:
      bench_idx = 0

  if len(tradable_indices) > 0 and "bw_price" in dp.data and bars > w:
    bw_mat = dp.data["bw_price"].reshape(dp.groups, bars)[tradable_indices, w:]
    valid_mask = (~np.isnan(bw_mat)) & (bw_mat > 0)
    has_any = valid_mask.any(axis=1)
    first_idx = np.argmax(valid_mask, axis=1)
    # Only include stocks already tradable near start to avoid survivor/IPO extreme skew
    keep = has_any & (first_idx <= max(5, int(0.1 * eff_bars)))
    if np.any(keep):
      kept_p = bw_mat[keep]
      kept_v = valid_mask[keep]
      base_p = kept_p[np.arange(kept_p.shape[0]), first_idx[keep]][:, None]
      ratio = np.where(kept_v, kept_p / base_p, np.nan)
      portfolio_nav = np.nanmean(ratio, axis=0)
      r.ror_original = np.round(portfolio_nav - 1.0, 4)
    else:
      r.ror_original = np.zeros(eff_bars, dtype=np.float64)
  elif "bw_price" in dp.data and bars > w:
    r.ror_original = np.round(uniform_ror(dp.slice("bw_price", 0)[w:]), 4)
  else:
    r.ror_original = np.zeros(eff_bars, dtype=np.float64)

  r.ror_original = (
    pl.Series("", r.ror_original)
    .fill_nan(None)
    .fill_null(strategy="forward")
    .fill_null(0.0)
    .to_numpy()
  )

  r.pool_size = len(tradable_indices)
  if hasattr(account, "records"):
    r.traded_objs_count = len(set(rec.obj for rec in account.records))
  else:
    r.traded_objs_count = 0

  # Aggregate portfolio-level actions per bar
  all_actions = account.actions.reshape(dp.groups, bars)[:, w:]
  if len(tradable_indices) > 0:
    has_buy = (all_actions[tradable_indices] == 1).any(axis=0)
    has_sell = (all_actions[tradable_indices] == 2).any(axis=0)
  else:
    has_buy = (all_actions == 1).any(axis=0)
    has_sell = (all_actions == 2).any(axis=0)
  r.actions = np.where(has_buy, 1, np.where(has_sell, 2, 0)).astype(np.int32)

  nav = (
    account.portfolio_assets[w:] / base_asset
    if base_asset > 0
    else np.ones(eff_bars, dtype=np.float64)
  )
  r.price = np.round(nav, 4)
  r.close = r.price
  r.fw_price = np.round(nav, 3)
  r.bw_price = np.round(nav, 3)

  # Compute portfolio metrics against benchmark
  if "bw_price" in dp.data and bars > w and bench_idx is not None and bench_idx < dp.groups:
    benchmark = uniform_ror(dp.slice("bw_price", bench_idx * bars)[w:])
  elif "bw_price" in dp.data and bars > w and dp.groups > 0:
    benchmark = uniform_ror(dp.slice("bw_price", 0)[w:])
  else:
    benchmark = np.zeros(eff_bars, dtype=np.float64)

  r.benchmark_ror = np.round(
    pl.Series("", benchmark).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).to_numpy(),
    4,
  )
  if bench_idx is not None and hasattr(dp, "objs") and bench_idx < len(dp.objs):
    r.benchmark_obj = str(dp.objs[bench_idx])
  elif getattr(account, "benchmark_obj", None):
    r.benchmark_obj = str(account.benchmark_obj)

  strategy_ratio = 1.0 + r.ror
  benchmark_ratio = 1.0 + benchmark

  if len(strategy_ratio) > 1:
    strategy_daily = np.diff(strategy_ratio) / strategy_ratio[:-1]
    benchmark_daily = np.diff(benchmark_ratio) / benchmark_ratio[:-1]
  else:
    strategy_daily = np.array([], dtype=np.float64)
    benchmark_daily = np.array([], dtype=np.float64)

  if len(strategy_daily) >= 2:
    mean_strat = np.mean(strategy_daily)
    std_strat = np.std(strategy_daily, ddof=1)
    sharpe = (mean_strat / std_strat * np.sqrt(252)) if std_strat != 0.0 else 0.0

    cov = np.cov(strategy_daily, benchmark_daily)
    var_bench = cov[1, 1]
    beta = (cov[0, 1] / var_bench) if var_bench != 0.0 else 0.0

    mean_bench = np.mean(benchmark_daily)
    alpha = (mean_strat - beta * mean_bench) * 252

    active_daily = strategy_daily - benchmark_daily
    mean_active = np.mean(active_daily)
    std_active = np.std(active_daily, ddof=1)
    ir = (mean_active / std_active * np.sqrt(252)) if std_active != 0.0 else 0.0

    downside_daily = np.minimum(strategy_daily, 0.0)
    std_downside = np.std(downside_daily, ddof=1)
    sortino = (mean_strat / std_downside * np.sqrt(252)) if std_downside != 0.0 else 0.0

    treynor = (mean_strat * 252 / beta) if beta != 0.0 else 0.0
  else:
    sharpe = beta = alpha = ir = sortino = treynor = 0.0

  if len(r.ror) > 0:
    equity = 1.0 + r.ror
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks
    max_drawdown = float(np.min(drawdowns))
  else:
    max_drawdown = 0.0

  calmar = (
    (np.mean(strategy_daily) * 252) / abs(max_drawdown)
    if max_drawdown != 0.0 and len(strategy_daily) >= 2
    else 0.0
  )

  # Bar-level / daily statistics
  trade_bars = int(np.sum((has_buy | has_sell)))
  wins_ret = strategy_daily[strategy_daily > 0]
  losses_ret = strategy_daily[strategy_daily < 0]
  active_days = len(wins_ret) + len(losses_ret)
  daily_win_rate = (len(wins_ret) / active_days) if active_days > 0 else 0.0
  daily_avg_win = float(np.mean(wins_ret)) if len(wins_ret) > 0 else 0.0
  daily_avg_loss = float(np.mean(losses_ret)) if len(losses_ret) > 0 else 0.0
  daily_sum_loss = float(abs(np.sum(losses_ret)))
  daily_profit_factor = float(np.sum(wins_ret) / daily_sum_loss) if daily_sum_loss != 0.0 else 0.0

  # Trade-level holding cycle statistics and attribution
  trades = extract_portfolio_trades(account=account, dp=dp)
  initial_cash = getattr(account, "initial_cash", 1000000.0)
  attrib = compute_stock_attribution(trades, initial_cash=initial_cash)
  r.trades = trades
  r.attribution = attrib

  if trades:
    trade_wins = [float(t["profit_rate"]) for t in trades if float(t["profit_rate"]) > 0]
    trade_losses = [float(t["profit_rate"]) for t in trades if float(t["profit_rate"]) < 0]
    trade_win_rate = (len(trade_wins) / len(trades)) if trades else 0.0
    trade_avg_win = float(np.mean(trade_wins)) if trade_wins else 0.0
    trade_avg_loss = float(np.mean(trade_losses)) if trade_losses else 0.0
    trade_sum_loss = float(abs(sum(trade_losses)))
    trade_profit_factor = (
      float(sum(trade_wins) / trade_sum_loss) if trade_sum_loss != 0.0 else 0.0
    )
  else:
    trade_win_rate = trade_avg_win = trade_avg_loss = trade_profit_factor = 0.0

  r.alpha = 0.0 if np.isnan(alpha) else float(np.round(alpha, 4))
  r.beta = 0.0 if np.isnan(beta) else float(np.round(beta, 4))
  r.sharpe = 0.0 if np.isnan(sharpe) else float(np.round(sharpe, 4))
  r.information_ratio = 0.0 if np.isnan(ir) else float(np.round(ir, 4))
  r.ir = r.information_ratio
  r.sortino = 0.0 if np.isnan(sortino) else float(np.round(sortino, 4))
  r.treynor = 0.0 if np.isnan(treynor) else float(np.round(treynor, 4))
  r.max_drawdown = 0.0 if np.isnan(max_drawdown) else float(np.round(max_drawdown, 4))
  r.calmar = 0.0 if np.isnan(calmar) else float(np.round(calmar, 4))

  r.daily_win_rate = 0.0 if np.isnan(daily_win_rate) else float(np.round(daily_win_rate, 4))
  r.daily_avg_win = 0.0 if np.isnan(daily_avg_win) else float(np.round(daily_avg_win, 4))
  r.daily_avg_loss = 0.0 if np.isnan(daily_avg_loss) else float(np.round(daily_avg_loss, 4))
  r.daily_profit_factor = 0.0 if np.isnan(daily_profit_factor) else float(np.round(daily_profit_factor, 4))

  r.trade_win_rate = float(np.round(trade_win_rate, 4))
  r.trade_avg_win = float(np.round(trade_avg_win, 4))
  r.trade_avg_loss = float(np.round(trade_avg_loss, 4))
  r.trade_profit_factor = float(np.round(trade_profit_factor, 4))

  r.win_rate = r.trade_win_rate if trades else r.daily_win_rate
  r.avg_win = r.trade_avg_win if trades else r.daily_avg_win
  r.avg_loss = r.trade_avg_loss if trades else r.daily_avg_loss
  r.profit_factor = r.trade_profit_factor if trades else r.daily_profit_factor

  r.profit = float(r.ror[-1]) if r.ror.size > 0 else 0.0
  r.profit_original = float(r.ror_original[-1]) if r.ror_original.size > 0 else 0.0
  r.benchmark_profit = (
    0.0
    if (len(benchmark) == 0 or np.isnan(benchmark[-1]))
    else float(np.round(benchmark[-1], 4))
  )
  r.fee_rate = float(np.round(account.deducted / account.initial_cash, 4)) if account.initial_cash > 0 else 0.0
  records = getattr(account, "records", [])
  r.trade_count = len(records) if len(records) > 0 else trade_bars
  r.rebalance_count = trade_bars

  return [r]


def build_json_report(reports: list[Report]) -> dict:
  """Helper to build a JSON report from DataProvider"""

  data = {}
  for report in reports:
    d = {}
    d.update(report.metrics())
    d.update(report.detailed())
    if hasattr(report, "trades") and report.trades:
      d["trades"] = report.trades
    if hasattr(report, "attribution") and report.attribution:
      d["attribution"] = report.attribution
    if getattr(report, "diagnostics", None):
      d["diagnostics"] = report.diagnostics
    data[report.obj] = d
  return data


SUMMARY_METRICS = [
  "profit",
  "profit_original",
  "benchmark_profit",
  "alpha",
  "beta",
  "sharpe",
  "information_ratio",
  "sortino",
  "treynor",
  "max_drawdown",
  "calmar",
  "win_rate",
  "avg_win",
  "avg_loss",
  "profit_factor",
  "fee_rate",
  "trade_count",
  "rebalance_count",
]


def build_markdown_summary(
  reports: list[Report], top_k: int = 10, account=None, holdout: str | int | None = None
) -> str:
  """Build a compact Markdown statistical summary of backtest reports for humans and AI Agents."""
  total = len(reports)
  if total == 0:
    return "# Backtest Summary\n\nNo reports generated.\n"

  active = sum(1 for r in reports if r.trade_count > 0)
  pos_profit = sum(1 for r in reports if r.profit > 0)
  beat_hold = sum(1 for r in reports if r.profit > r.profit_original)
  bench_ret = float(getattr(reports[0], "benchmark_profit", 0.0))
  beat_bench = sum(
    1 for r in reports if r.profit > float(getattr(r, "benchmark_profit", 0.0))
  )

  is_portfolio = (total == 1 and reports[0].obj == "PORTFOLIO") or (getattr(reports[0], "records", None) is not None)
  pool_note = " (Pool Buy&Hold)" if is_portfolio else ""
  exec_mode = getattr(account, "exec_mode", "close") if account else "close"
  exec_info = f" | **Exec Mode**: `{exec_mode}`" if exec_mode != "close" else ""

  if is_portfolio:
    r0 = reports[0]
    pool_sz = getattr(r0, "pool_size", total)
    traded_cnt = getattr(r0, "traded_objs_count", active)
    objs_line = f"- **Mode**: Portfolio (Shared Cash){exec_info} | **Pool Size**: {pool_sz} | **Active (Traded Objs)**: {traded_cnt} ({traded_cnt / pool_sz * 100:.2f}%)" if pool_sz > 0 else f"- **Total Objs**: {total}{exec_info} | **Active (Traded)**: {active}"
  else:
    objs_line = f"- **Total Objs**: {total}{exec_info} | **Active (Traded)**: {active} ({active / total * 100:.2f}%)"

  lines = [
    "# Backtest Summary",
    "",
    objs_line,
    f"- **Positive Profit**: {pos_profit} ({pos_profit / total * 100:.2f}%) | **Beat Buy&Hold{pool_note}**: {beat_hold} ({beat_hold / total * 100:.2f}%)",
    f"- **Benchmark Return**: {bench_ret:.4f} ({bench_ret * 100:.2f}%) | **Beat Benchmark**: {beat_bench} ({beat_bench / total * 100:.2f}%)",
  ]

  if not is_portfolio:
    no_cash_signals = sum(
      int(np.count_nonzero(r.actions == 5))
      for r in reports
      if getattr(r, "actions", None) is not None and len(r.actions) > 0
    )
    buy_fills = sum(
      int(np.count_nonzero(r.actions == 1))
      for r in reports
      if getattr(r, "actions", None) is not None and len(r.actions) > 0
    )
    if no_cash_signals > 0 and (buy_fills == 0 or no_cash_signals > buy_fills):
      init_c = float(getattr(account, "initial_cash", 0.0)) if account is not None else 0.0
      unit_sz = int(getattr(account, "unit", 100)) if account is not None else 100
      cash_hint = f" (initial_cash={init_c:,.0f}, unit={unit_sz})" if init_c > 0 else f" (unit={unit_sz})"
      lines.extend([
        "",
        "> [!WARNING]",
        f"> **{no_cash_signals:,} buy signals rejected (`BUY_NO_CASH`)** vs {buy_fills:,} filled buys{cash_hint}. "
        "Order budget may be smaller than 1 lot (`unit * price`); increase `-c/--cash` or `buy_percent`.",
      ])

  if total == 1:
    r = reports[0]
    lines.extend([
      "",
      "## Strategy Metrics",
      "",
      "| Metric | Value |",
      "| :--- | ---: |",
      f"| profit | {r.profit:+.2%} |",
      f"| benchmark_profit | {bench_ret:+.2%} |",
      f"| profit_original{pool_note} | {r.profit_original:+.2%} |",
      f"| alpha | {r.alpha:.4f} |",
      f"| beta | {r.beta:.4f} |",
      f"| sharpe | {r.sharpe:.4f} |",
      f"| sortino | {r.sortino:.4f} |",
      f"| information_ratio | {r.information_ratio:.4f} |",
      f"| max_drawdown | {r.max_drawdown:.2%} |",
      f"| calmar | {r.calmar:.4f} |",
      f"| trade_win_rate (Single-Trade) | {getattr(r, 'trade_win_rate', r.win_rate):.2%} |",
      f"| daily_win_rate (Bar-Level) | {getattr(r, 'daily_win_rate', r.win_rate):.2%} |",
      f"| trade_avg_win | {getattr(r, 'trade_avg_win', r.avg_win):+.2%} |",
      f"| trade_avg_loss | {getattr(r, 'trade_avg_loss', r.avg_loss):+.2%} |",
      f"| profit_factor | {r.profit_factor:.4f} |",
      f"| fee_rate | {r.fee_rate:.2%} |",
      f"| trade_count | {r.trade_count} |",
      f"| rebalance_count | {getattr(r, 'rebalance_count', r.trade_count)} |",
    ])
  else:
    lines.extend([
      "",
      "## Metrics Statistics",
      "",
      "| Metric | Mean | Std | Min | 25% | 50% (Median) | 75% | Max |",
      "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for m in SUMMARY_METRICS:
      vals = np.array([float(getattr(r, m, 0.0)) for r in reports], dtype=np.float64)
      vals = vals[~np.isnan(vals)]
      if len(vals) == 0:
        lines.append(f"| {m} | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |")
        continue
      mean_v = float(np.mean(vals))
      std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
      min_v = float(np.min(vals))
      p25, p50, p75 = [float(x) for x in np.percentile(vals, [25, 50, 75])]
      max_v = float(np.max(vals))
      lines.append(
        f"| {m} | {mean_v:.4f} | {std_v:.4f} | {min_v:.4f} | {p25:.4f} | {p50:.4f} | {p75:.4f} | {max_v:.4f} |"
      )

  def _format_obj_table(title: str, subset: list[Report]):
    lines.extend([
      "",
      f"## {title}",
      "",
      "| obj | profit | benchmark_profit | profit_original | alpha | sharpe | max_drawdown | win_rate | fee_rate | trade_count |",
      "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in subset:
      bench_p = float(getattr(r, "benchmark_profit", 0.0))
      lines.append(
        f"| {r.obj} | {r.profit:.4f} | {bench_p:.4f} | {r.profit_original:.4f} | {r.alpha:.4f} | {r.sharpe:.4f} | {r.max_drawdown:.4f} | {r.win_rate:.4f} | {r.fee_rate:.4f} | {r.trade_count} |"
      )

  k = min(max(1, top_k), total)

  if k > 1:
    sorted_desc = sorted(reports, key=lambda x: x.profit, reverse=True)
    _format_obj_table(f"Top {k} Objs (by profit)", sorted_desc[:k])
    sorted_asc = sorted(reports, key=lambda x: x.profit)
    _format_obj_table(f"Bottom {k} Objs (by profit)", sorted_asc[:k])

  # Holdout / Out-of-sample split statistics
  if holdout is not None and len(reports) > 0:
    split_info = compute_holdout_split(reports[0], holdout)
    if split_info:
      lines.extend([
        "",
        "## Sample Split Validation (Holdout Guard)",
        "",
        f"- **Holdout Split Date**: {split_info['split_date']}",
        f"- **In-Sample (IS)**: {split_info['is_range']} | Profit: {split_info['is_profit']:+.2%} | Sharpe: {split_info['is_sharpe']:.2f} | MaxDD: {split_info['is_max_dd']:.2%}",
        f"- **Out-of-Sample (OOS)**: {split_info['oos_range']} | Profit: {split_info['oos_profit']:+.2%} | Sharpe: {split_info['oos_sharpe']:.2f} | MaxDD: {split_info['oos_max_dd']:.2%}",
        f"- **Overfit Ratio (OOS/IS Sharpe)**: {split_info['overfit_ratio']:.2f}",
      ])

  # 调仓记录 (Rebalance Records) for Portfolio strategies
  acc = account
  if acc is None and len(reports) > 0:
    acc = getattr(reports[0], "account", None)

  records = getattr(acc, "records", None) if acc else None
  if records is None and len(reports) > 0:
    records = getattr(reports[0], "records", None)

  is_portfolio = (total == 1 and reports[0].obj == "PORTFOLIO") or (records is not None)
  if is_portfolio and (acc is not None or records is not None):
    dp_ref = getattr(acc, "dp", None) if acc else getattr(reports[0], "dp", None)
    trades = getattr(reports[0], "trades", None)
    if not trades:
      trades = extract_portfolio_trades(account=acc, dp=dp_ref, records=records)

    attrib = getattr(reports[0], "attribution", None)
    if not attrib and trades:
      init_c = getattr(acc, "initial_cash", 1000000.0) if acc else 1000000.0
      attrib = compute_stock_attribution(trades, initial_cash=init_c)

    if attrib:
      lines.extend([
        "",
        "## Stock Attribution",
        "",
        "| obj | Trade Count | Win Rate | Total Profit | Capital Contribution | Avg Return |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
      ])
      display_attrib = attrib if len(attrib) <= 10 else (attrib[:5] + attrib[-5:])
      for at in display_attrib:
        lines.append(
          f"| {at['obj']} | {at['trade_count']} | {at['win_rate']:.2%} | {at['total_profit']:+,.2f} | {at['capital_contribution']:+.2%} | {at['avg_return']:+.2%} |"
        )

    lines.extend([
      "",
      "## Rebalance Records",
      "",
      "| obj | Entry | Exit | Hold profit |",
      "| :--- | :--- | :--- | ---: |",
    ])
    if trades:
      for t in trades:
        rate_str = f"{t['profit_rate']:+.2%}" if abs(t['profit_rate']) >= 1e-9 else "0.00%"
        amt_str = f"{t['profit_amount']:+,.2f}" if abs(t['profit_amount']) >= 1e-9 else "0.00"
        profit_str = f"{rate_str} ({amt_str})"
        lines.append(
          f"| {t['obj']} | {t['entry_date']} | {t['exit_date']} | {profit_str} |"
        )
    else:
      lines.append("| - | - | - | - |")

  # Strategy Diagnostics
  diag = None
  if len(reports) > 0:
    diag = getattr(reports[0], "diagnostics", None)
  if not diag and account is not None:
    diag = getattr(account, "diagnostics", None)

  if diag and isinstance(diag, dict):
    lines.extend([
      "",
      "## Strategy Diagnostics",
      "",
    ])
    for k, v in diag.items():
      if isinstance(v, dict):
        lines.append(f"### {k}")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("| :--- | ---: |")
        for sub_k, sub_v in v.items():
          lines.append(f"| {sub_k} | {sub_v} |")
        lines.append("")
      elif isinstance(v, (list, tuple)):
        lines.append(f"### {k}")
        lines.append("")
        for item in v:
          lines.append(f"- {item}")
        lines.append("")
      else:
        lines.append(f"- **{k}**: {v}")

  return "\n".join(lines) + "\n"


def compute_holdout_split(report: Report, holdout: str | int) -> dict | None:
  """Split report returns into In-Sample (IS) and Out-of-Sample (OOS)."""
  if len(report.date) < 4:
    return None

  dates = report.date
  split_idx = -1
  holdout_str = str(holdout).strip()
  if holdout_str.endswith("d") or holdout_str.endswith("D"):
    days = int(holdout_str[:-1])
    split_idx = max(2, len(dates) - days)
  elif holdout_str.endswith("y") or holdout_str.endswith("Y"):
    days = int(float(holdout_str[:-1]) * 250)
    split_idx = max(2, len(dates) - days)
  else:
    try:
      d_target = str(holdout)[:10]
      if len(dates) > 0 and hasattr(dates, "dtype") and np.issubdtype(dates.dtype, np.datetime64):
        target_val = np.datetime64(d_target).astype(dates.dtype)
        split_idx = int(np.searchsorted(dates, target_val))
      elif len(dates) > 0 and isinstance(dates[0], (str, bytes)):
        split_idx = int(np.searchsorted(dates, d_target))
      elif len(dates) > 0:
        v0 = float(dates[0])
        if isinstance(holdout, (int, np.integer)) or (holdout_str.isdigit() and len(holdout_str) < 8):
          target_val = int(holdout_str)
        else:
          dt = datetime.datetime.fromisoformat(d_target).replace(tzinfo=datetime.timezone.utc)
          if v0 > 1e17:
            target_val = int(dt.timestamp() * 1e9)
          elif v0 > 1e14:
            target_val = int(dt.timestamp() * 1e6)
          elif v0 > 1e11:
            target_val = int(dt.timestamp() * 1000)
          elif v0 > 1e8:
            target_val = int(dt.timestamp())
          else:
            target_val = int(d_target.replace("-", ""))
        split_idx = int(np.searchsorted(dates, target_val))
    except Exception as e:
      raise ValueError(
        f"Invalid --holdout value '{holdout}'. Expected a date string (YYYY-MM-DD), "
        "bar count (e.g. '60d'), or year duration (e.g. '1y')."
      ) from e

  if split_idx <= 1 or split_idx >= len(dates) - 1:
    return None

  equity = 1.0 + report.ror
  is_equity = equity[:split_idx] / equity[0]
  oos_equity = equity[split_idx:] / equity[split_idx]

  is_profit = float(is_equity[-1] - 1.0)
  oos_profit = float(oos_equity[-1] - 1.0)

  def _calc_sharpe(eq):
    if len(eq) < 3:
      return 0.0
    daily = np.diff(eq) / eq[:-1]
    std = np.std(daily, ddof=1)
    return float(np.mean(daily) / std * np.sqrt(252)) if std > 0 else 0.0

  def _calc_max_dd(eq):
    peaks = np.maximum.accumulate(eq)
    dds = (eq - peaks) / peaks
    return float(np.min(dds)) if len(dds) > 0 else 0.0

  is_sharpe = _calc_sharpe(is_equity)
  oos_sharpe = _calc_sharpe(oos_equity)
  is_dd = _calc_max_dd(is_equity)
  oos_dd = _calc_max_dd(oos_equity)

  def _fmt_d(val):
    if isinstance(val, (int, np.integer)):
      s = str(val)
      if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
      for threshold, scale in ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3), (1e8, 1.0)):
        if val > threshold:
          return datetime.datetime.fromtimestamp(
            val / scale, tz=datetime.timezone.utc
          ).strftime("%Y-%m-%d")
    return str(val)[:10]

  split_date_str = _fmt_d(dates[split_idx])
  is_start_str = _fmt_d(dates[0])
  is_end_str = _fmt_d(dates[split_idx - 1])
  oos_start_str = split_date_str
  oos_end_str = _fmt_d(dates[-1])

  ratio = (oos_sharpe / is_sharpe) if is_sharpe > 0 else 0.0

  return {
    "split_date": split_date_str,
    "is_range": f"{is_start_str} ~ {is_end_str}",
    "oos_range": f"{oos_start_str} ~ {oos_end_str}",
    "is_profit": is_profit,
    "oos_profit": oos_profit,
    "is_sharpe": is_sharpe,
    "oos_sharpe": oos_sharpe,
    "is_max_dd": is_dd,
    "oos_max_dd": oos_dd,
    "overfit_ratio": ratio,
  }


def build_multi_start_summary(runs: list[dict]) -> str:
  """Build a summary table comparing multiple backtest runs across different start dates."""
  if not runs:
    return "# Multi-Start Comparison Summary\n\nNo backtest runs executed.\n"

  lines = [
    "# Multi-Start Backtest Comparison Summary",
    "",
    "| Start Date | End Date | Profit | Benchmark | Beat Bench | Max Drawdown | Sharpe | Win Rate | Trades | Rebalance |",
    "| :--- | :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
  ]

  profits = []
  sharpes = []
  drawdowns = []
  beat_count = 0

  for r_info in runs:
    rep: Report = r_info["report"]
    start_d = str(r_info.get("start", rep.date[0] if len(rep.date) > 0 else "-"))
    end_d = str(r_info.get("end", rep.date[-1] if len(rep.date) > 0 else "-"))
    bench_p = float(getattr(rep, "benchmark_profit", 0.0))
    beat = rep.profit > bench_p
    if beat:
      beat_count += 1
    profits.append(rep.profit)
    sharpes.append(rep.sharpe)
    drawdowns.append(rep.max_drawdown)

    beat_str = "YES" if beat else "NO"
    trades_cnt = getattr(rep, "trade_count", 0)
    rebal_cnt = getattr(rep, "rebalance_count", 0)

    lines.append(
      f"| {start_d} | {end_d} | {rep.profit:+.2%} | {bench_p:+.2%} | {beat_str} | {rep.max_drawdown:.2%} | {rep.sharpe:.2f} | {rep.win_rate:.1%} | {trades_cnt} | {rebal_cnt} |"
    )

  total = len(runs)
  avg_profit = float(np.mean(profits)) if profits else 0.0
  avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
  worst_dd = float(np.min(drawdowns)) if drawdowns else 0.0

  lines.extend([
    "",
    f"- **Evaluated Starts**: {total} | **Beat Benchmark**: {beat_count}/{total} ({beat_count / total * 100:.1f}%)",
    f"- **Average Strategy Profit**: {avg_profit:+.2%} | **Average Sharpe**: {avg_sharpe:.2f} | **Worst Max Drawdown**: {worst_dd:.2%}",
    "",
  ])

  return "\n".join(lines) + "\n"




  