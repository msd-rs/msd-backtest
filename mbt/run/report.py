import datetime
import dataclasses
from mbt import Account
from mbt import DataProvider
import polars as pl
import numpy as np


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
  

  def __init__(self, obj: str, dp: DataProvider , account: Account ):
    if len(obj) == 0:
      return
    self.obj = obj
    s, e = dp.obj_indices(obj)
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
    self.ror = pl.Series("", self.ror).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).to_numpy()
    self.ror_hold = pl.Series("", self.ror_hold).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).to_numpy()
    self.ror_original = pl.Series("", self.ror_original).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).to_numpy()

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

    return r    
    


  def calculate_metrics(self, dp: DataProvider, account: Account):
    w = max(0, int(getattr(dp, "warmup_bars", 0)))
    bench_bw = dp.slice("bw_price", 0)
    if w > 0 and len(bench_bw) > w:
      bench_bw = bench_bw[w:]
    benchmark = uniform_ror(bench_bw)

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
    action_shares = dp.slice("action_shares", self.obj) if "action_shares" in dp.data else np.zeros_like(self.price)

    # Track shares held at each bar
    shares = np.zeros(len(self.actions))
    curr_shares = 0.0
    for t in range(len(self.actions)):
      act = self.actions[t]
      sh = action_shares[t]
      if act == 1:    # ACTION_BUY
        curr_shares += sh
      elif act == 2:  # ACTION_SELL
        curr_shares -= sh
      shares[t] = max(curr_shares, 0.0)

    # Identify holding periods where shares > 0
    holding = shares > 0
    padded = np.concatenate(([False], holding, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    trade_profits = []
    trade_returns = []

    s = dp.objs.index(self.obj) * dp.bars
    e = s + dp.bars
    assets = dp.all("assets")[s:e]

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


def build_report(dp: DataProvider, accounts: list[Account]) -> list[Report]:
  reports = [Report(obj, dp, accounts[i]) for i, obj in enumerate(dp.objs)]

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
      pos_price = 0.0
      if positions and g < len(positions):
        pos_price = positions[g].price
      elif recs:
        pos_price = recs[-1].deal_price
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
  dp: DataProvider, account, obj_name: str = "PORTFOLIO"
) -> list[Report]:
  """Build a Report for a shared-cash PortfolioAccount."""
  r = Report("", None, None)  # type: ignore
  r.obj = obj_name
  r.account = account
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
  start_g = getattr(account, "tradable_start", 0)
  if dp.groups > start_g and "bw_price" in dp.data:
    orig_rors = [
      uniform_ror(dp.slice("bw_price", g * bars)[w:])
      for g in range(start_g, dp.groups)
    ]
    r.ror_original = np.round(np.nanmean(orig_rors, axis=0), 4)
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

  # Aggregate portfolio-level actions per bar
  all_actions = account.actions.reshape(dp.groups, bars)[:, w:]
  has_buy = (all_actions[start_g:] == 1).any(axis=0)
  has_sell = (all_actions[start_g:] == 2).any(axis=0)
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

  # Compute portfolio metrics against benchmark (group 0)
  if "bw_price" in dp.data and bars > w:
    benchmark = uniform_ror(dp.slice("bw_price", 0)[w:])
  else:
    benchmark = np.zeros(eff_bars, dtype=np.float64)

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

  lines = [
    "# Backtest Summary",
    "",
    f"- **Total Objs**: {total} | **Active (Traded)**: {active} ({active / total * 100:.2f}%)",
    f"- **Positive Profit**: {pos_profit} ({pos_profit / total * 100:.2f}%) | **Beat Buy&Hold{pool_note}**: {beat_hold} ({beat_hold / total * 100:.2f}%)",
    f"- **Benchmark Return**: {bench_ret:.4f} ({bench_ret * 100:.2f}%) | **Beat Benchmark**: {beat_bench} ({beat_bench / total * 100:.2f}%)",
  ]

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
        dt = datetime.datetime.fromisoformat(d_target)
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
    except Exception:
      split_idx = len(dates) // 2

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
      if val > 1e11:
        return datetime.date.fromtimestamp(val / 1000).isoformat()
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




  