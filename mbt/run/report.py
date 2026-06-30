from mbt import Account
from mbt import DataProvider
import pandas as pd
import numpy as np


class Report:

  def __init__(self, symbol: str, dp: DataProvider, account: Account):
    self.symbol = symbol
    s, e = dp.symbol_indices(symbol)
    self.date = dp.dates[s:e]
    

    self.ror = np.round(dp.slice("ror", s), 4)
    self.ror_hold = np.round(dp.slice("ror_hold", s), 4)
    self.ror_original = np.round(uniform_ror(dp.slice("bw_price", s)), 4)
    
    # Clean NaNs by forward filling and filling leading NaNs with 0
    self.ror = pd.Series(self.ror).ffill().fillna(0.0).values
    self.ror_hold = pd.Series(self.ror_hold).ffill().fillna(0.0).values
    self.ror_original = pd.Series(self.ror_original).ffill().fillna(0.0).values

    self.actions = dp.slice("actions", s)
    self.price = dp.slice("price", s)
    self.close = self.price # For compatibility with tests
    self.fw_price = np.round(dp.slice("fw_price", s), 3)
    self.bw_price = np.round(dp.slice("bw_price", s), 3)
    
    self.calcaute_metrics(dp, account)
    


  def calcaute_metrics(self, dp: DataProvider, account: Account):
    bechmark = uniform_ror(dp.slice("bw_price", 0))

    # 1. Convert cumulative returns to daily returns
    strategy_ratio = 1.0 + self.ror
    benchmark_ratio = 1.0 + bechmark

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
    action_shares = dp.slice("action_shares", self.symbol) if "action_shares" in dp.data else np.zeros_like(self.price)

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

    s = dp.symbols.index(self.symbol) * dp.bars
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
    self.profit = self.ror[-1] if self.ror.size > 0 else 0.0
    self.profit_original = self.ror_original[-1] if self.ror_original.size > 0 else 0.0
    self.fee_rate = np.round(account.deducted / account.initial_cash, 4)
    self.trade_count = len(trade_profits)
    

  def metrics(self, with_symbol: bool = False) -> dict:
    d = {
      "symbol": self.symbol,
      "profit": self.profit,
      "profit_original": self.profit_original,
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
      "fee_rate": self.fee_rate,
      "trade_count": self.trade_count,
    }
    if not with_symbol:
      del d["symbol"]
    return d

  def detailed(self, with_symbol: bool = False) -> dict:
    d = {
      "symbol": [self.symbol] * len(self.date),
      "date": self.date,
      "ror": self.ror,
      "ror_hold": self.ror_hold,
      "ror_original": self.ror_original,
      "actions": self.actions,
    }
    if not with_symbol:
      del d["symbol"]
    return d

def uniform_ror(a: np.ndarray) -> np.ndarray:
  base = a[0]
  if base == 0:
    return a
  return ((a - base) / base)


def build_report(dp: DataProvider, accounts: list[Account]) -> list[Report]:
  reports = [Report(symbol, dp, accounts[i]) for i, symbol in enumerate(dp.symbols)]

  return reports


def build_json_report(reports: list[Report]) -> dict:
  """Helper to build a JSON report from DataProvider"""

  data = {}
  for report in reports:
    d = {}
    d.update(report.metrics())
    d.update(report.detailed())
    d['price'] = report.price
    d['fw_price'] = report.fw_price
    d['bw_price'] = report.bw_price
    data[report.symbol] = d
  return data
