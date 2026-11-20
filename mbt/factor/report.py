# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Factor evaluation report models and Markdown / JSON / CSV / XLSX exporters."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import orjson
import polars as pl

from mbt.factor.metrics import ICSummary, QuantileAnalysis

logger = logging.getLogger("factor")


def _fmt_pct(v: float | None, digits: int = 2) -> str:
  if v is None or not np.isfinite(v):
    return "N/A"
  return f"{v * 100.0:.{digits}f}%"


def _fmt_num(v: float | None, digits: int = 4) -> str:
  if v is None or not np.isfinite(v):
    return "N/A"
  return f"{v:.{digits}f}"


def _fmt_bps(v: float | None, digits: int = 1) -> str:
  if v is None or not np.isfinite(v):
    return "N/A"
  return f"{v * 10000.0:.{digits}f}bp"


def _date_strings(dates: np.ndarray) -> list[str]:
  return [str(d)[:10] for d in dates]


def _sanitize_json(obj: Any) -> Any:
  """Recursively convert numpy types and non-finite floats into JSON-safe values."""
  if isinstance(obj, dict):
    return {str(k): _sanitize_json(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_sanitize_json(v) for v in obj]
  if isinstance(obj, np.ndarray):
    return [_sanitize_json(v) for v in obj.tolist()]
  if isinstance(obj, (np.floating, float)):
    fv = float(obj)
    return None if not np.isfinite(fv) else round(fv, 8)
  if isinstance(obj, (np.integer, int)):
    return int(obj)
  if isinstance(obj, (np.bool_, bool)):
    return bool(obj)
  if isinstance(obj, np.datetime64):
    return str(obj)[:10]
  return obj


@dataclasses.dataclass
class FactorReport:
  """Structured evaluation report for a single alpha factor."""

  name: str
  dates: np.ndarray
  n_objs: int
  bars: int
  start_date: str
  end_date: str
  periods: tuple[int, ...]
  primary_period: int
  quantiles: int
  entry: str
  winsorize: str
  standardize: str
  neutralize: tuple[str, ...]
  cost_bps: float
  direction: int
  benchmark: str
  ic_summaries: dict[int, ICSummary]
  ic_series: dict[int, np.ndarray]
  rank_ic_series: dict[int, np.ndarray]
  quantile_analyses: dict[int, QuantileAnalysis]
  turnover: dict[str, Any]
  annual_breakdown: list[dict[str, Any]]
  holdout_breakdown: dict[str, Any] | None
  industry_breakdown: list[dict[str, Any]]
  diagnostics: dict[str, Any]
  expression: str = ""
  author: str = ""

  @property
  def primary_ic(self) -> ICSummary:
    return self.ic_summaries[self.primary_period]

  @property
  def primary_quantile(self) -> QuantileAnalysis:
    return self.quantile_analyses[self.primary_period]

  def summary_row(self) -> dict[str, Any]:
    """Compact one-row metric summary for leaderboards and assertions."""
    ic = self.primary_ic
    qa = self.primary_quantile
    hl = self.turnover.get("half_life_days")
    return {
      "factor": self.name,
      "period": self.primary_period,
      "rank_ic_mean": ic.rank_ic_mean,
      "rank_icir": ic.rank_icir,
      "nw_t_stat": ic.nw_t_stat,
      "ic_mean": ic.ic_mean,
      "top_ann_excess": qa.long_excess_market.ann_return,
      "ls_ann_return": qa.long_short.ann_return,
      "ls_sharpe": qa.long_short.sharpe,
      "monotonicity": qa.monotonicity,
      "top_daily_turnover": self.turnover.get("top_daily_turnover", 0.0),
      "half_life_days": hl,
      "avg_coverage": self.diagnostics.get("coverage_mean", 0.0),
      "mktcap_corr": self.diagnostics.get("mktcap_corr"),
      "causality": self.diagnostics.get("causality_check", {}).get("status", "UNKNOWN"),
    }

  def to_dict(self, include_series: bool = True) -> dict[str, Any]:
    """Serialize the report to a JSON-compatible dictionary."""
    payload: dict[str, Any] = {
      "meta": {
        "name": self.name,
        "expression": self.expression,
        "author": self.author,
        "direction": self.direction,
        "n_objs": self.n_objs,
        "bars": self.bars,
        "start_date": self.start_date,
        "end_date": self.end_date,
        "periods": list(self.periods),
        "primary_period": self.primary_period,
        "quantiles": self.quantiles,
        "entry": self.entry,
        "winsorize": self.winsorize,
        "standardize": self.standardize,
        "neutralize": list(self.neutralize),
        "cost_bps": self.cost_bps,
        "benchmark": self.benchmark,
      },
      "summary": self.summary_row(),
      "ic_by_period": {str(p): s.to_dict() for p, s in self.ic_summaries.items()},
      "quantile_by_period": {
        str(p): q.to_dict(include_series=include_series)
        for p, q in self.quantile_analyses.items()
      },
      "turnover": self.turnover,
      "annual_breakdown": self.annual_breakdown,
      "holdout_breakdown": self.holdout_breakdown,
      "industry_breakdown": self.industry_breakdown,
      "diagnostics": self.diagnostics,
    }
    if include_series:
      payload["dates"] = _date_strings(self.dates)
      payload["ic_series"] = {str(p): s.tolist() for p, s in self.ic_series.items()}
      payload["rank_ic_series"] = {str(p): s.tolist() for p, s in self.rank_ic_series.items()}
    return _sanitize_json(payload)

  def to_markdown(self) -> str:
    """Render a concise, Agent-friendly Markdown diagnostic report."""
    lines: list[str] = []
    lines.append(f"# Factor Evaluation Report: `{self.name}`")
    lines.append("")

    neut_str = ",".join(self.neutralize) if self.neutralize else "none"
    dir_str = "+1 (ascending)" if self.direction >= 0 else "-1 (descending, auto-flipped for quantiles)"
    lines.append(
      f"- **Range**: `{self.start_date}` ~ `{self.end_date}` ({self.bars} bars, {self.n_objs} stocks) | "
      f"**Entry**: `{self.entry}` | **Quantiles**: `{self.quantiles}` | **Direction**: `{dir_str}`"
    )
    lines.append(
      f"- **Preprocess**: `winsorize={self.winsorize}`, `neutralize={neut_str}`, "
      f"`standardize={self.standardize}`, `cost_bps={self.cost_bps:g}` | "
      f"**Causality**: `{self.diagnostics.get('causality_check', {}).get('status', 'UNKNOWN')}`"
    )
    if self.expression:
      lines.append(f"- **Expression**: `{self.expression}`")
    lines.append("")

    # 1. Multi-period IC / RankIC table
    lines.append("## 1. IC / RankIC Analysis by Horizon")
    lines.append(
      "| Period | Valid Bars | RankIC Mean | RankIC Std | RankICIR | Ann RankICIR | "
      "NW t-stat | IC Mean | IC>0 | |RankIC|>2% |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p in self.periods:
      s = self.ic_summaries[p]
      mark = "*" if p == self.primary_period else ""
      lines.append(
        f"| {p}d{mark} | {s.valid_bars} | {_fmt_num(s.rank_ic_mean, 4)} | "
        f"{_fmt_num(s.rank_ic_std, 4)} | {_fmt_num(s.rank_icir, 3)} | "
        f"{_fmt_num(s.ann_rank_icir, 2)} | {_fmt_num(s.nw_t_stat, 2)} | "
        f"{_fmt_num(s.ic_mean, 4)} | {_fmt_pct(s.rank_ic_positive_ratio, 1)} | "
        f"{_fmt_pct(s.rank_ic_sig_ratio, 1)} |"
      )
    lines.append("")

    # 2. Primary Period Quantile Breakdown
    qa = self.primary_quantile
    lines.append(
      f"## 2. Quantile Portfolio Performance (Primary Horizon: {self.primary_period}d, "
      f"Monotonicity: `{_fmt_num(qa.monotonicity, 3)}`)"
    )
    lines.append(
      "| Bucket | Mean Period Ret | Ann Return | Ann Vol | Sharpe | Max Drawdown | Total Return | Daily Turnover |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for qm in qa.quantiles:
      lines.append(
        f"| {qm.label} | {_fmt_bps(qm.mean_period_ret)} | {_fmt_pct(qm.ann_return)} | "
        f"{_fmt_pct(qm.ann_vol)} | {_fmt_num(qm.sharpe, 2)} | {_fmt_pct(qm.max_drawdown)} | "
        f"{_fmt_pct(qm.total_return)} | {_fmt_pct(qm.daily_turnover, 2)} |"
      )
    for sp in [qa.long_excess_market, qa.long_short]:
      lines.append(
        f"| **{sp.label}** | {_fmt_bps(sp.mean_period_ret)} | **{_fmt_pct(sp.ann_return)}** | "
        f"{_fmt_pct(sp.ann_vol)} | **{_fmt_num(sp.sharpe, 2)}** | {_fmt_pct(sp.max_drawdown)} | "
        f"{_fmt_pct(sp.total_return)} | {_fmt_pct(sp.daily_turnover, 2)} |"
      )
    if qa.long_excess_benchmark is not None:
      sb = qa.long_excess_benchmark
      lines.append(
        f"| **{sb.label} ({self.benchmark})** | - | **{_fmt_pct(sb.ann_return)}** | "
        f"{_fmt_pct(sb.ann_vol)} | **{_fmt_num(sb.sharpe, 2)}** | {_fmt_pct(sb.max_drawdown)} | "
        f"{_fmt_pct(sb.total_return)} | {_fmt_pct(sb.daily_turnover, 2)} |"
      )
    lines.append("")

    # 3. Turnover & Autocorrelation Decay
    ac = self.turnover.get("autocorrelation", {})
    ac_parts = [f"Lag {k}d: `{_fmt_num(v, 3)}`" for k, v in sorted(ac.items())]
    hl = self.turnover.get("half_life_days")
    hl_str = f"`{hl:.1f}d`" if hl is not None and np.isfinite(hl) else "`N/A`"
    lines.append("## 3. Turnover & Signal Decay")
    lines.append(
      f"- **Top Quantile Daily Turnover**: `{_fmt_pct(self.turnover.get('top_daily_turnover'), 2)}` | "
      f"**Bottom Daily Turnover**: `{_fmt_pct(self.turnover.get('bottom_daily_turnover'), 2)}` | "
      f"**Est. Half-Life**: {hl_str}"
    )
    if ac_parts:
      lines.append(f"- **Rank Autocorrelation**: {', '.join(ac_parts)}")
    lines.append("")

    # 4. Annual Stability
    if self.annual_breakdown:
      lines.append(f"## 4. Annual Stability ({self.primary_period}d)")
      lines.append("| Year | Bars | RankIC Mean | RankICIR | RankIC>0 | Top Excess Return |")
      lines.append("|---|---:|---:|---:|---:|---:|")
      for row in self.annual_breakdown:
        lines.append(
          f"| {row['year']} | {row['bars']} | {_fmt_num(row['rank_ic_mean'], 4)} | "
          f"{_fmt_num(row['rank_icir'], 3)} | {_fmt_pct(row['rank_ic_positive_ratio'], 1)} | "
          f"{_fmt_pct(row['top_excess_return'], 2)} |"
        )
      lines.append("")

    # 5. Holdout IS vs OOS
    if self.holdout_breakdown:
      hb = self.holdout_breakdown
      ins = hb["in_sample"]
      oos = hb["out_of_sample"]
      ret = hb.get("rank_ic_retention")
      ret_str = f"`{ret:.2f}x`" if ret is not None and np.isfinite(ret) else "`N/A`"
      lines.append(
        f"## 5. Holdout Validation (Split: `{hb['split_date']}`, OOS RankIC Retention: {ret_str})"
      )
      lines.append("| Segment | Range | Bars | RankIC Mean | RankICIR | Top Ann Excess | LS Ann Return | Monotonicity |")
      lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
      for seg_name, seg in [("In-Sample", ins), ("Out-of-Sample", oos)]:
        lines.append(
          f"| {seg_name} | {seg['start']} ~ {seg['end']} | {seg['bars']} | "
          f"{_fmt_num(seg['rank_ic_mean'], 4)} | {_fmt_num(seg['rank_icir'], 3)} | "
          f"{_fmt_pct(seg['top_ann_excess'])} | {_fmt_pct(seg['ls_ann_return'])} | "
          f"{_fmt_num(seg['monotonicity'], 3)} |"
        )
      lines.append("")

    # 6. Industry Breakdown (optional)
    if self.industry_breakdown:
      lines.append(f"## 6. Industry RankIC Breakdown ({self.primary_period}d, Top/Bottom)")
      lines.append("| Industry | Stocks | Avg Valid | RankIC Mean | RankICIR |")
      lines.append("|---|---:|---:|---:|---:|")
      show_rows = self.industry_breakdown
      if len(show_rows) > 12:
        show_rows = show_rows[:6] + show_rows[-6:]
      for row in show_rows:
        lines.append(
          f"| {row['industry']} | {row['n_stocks']} | {row['avg_valid_stocks']:.1f} | "
          f"{_fmt_num(row['rank_ic_mean'], 4)} | {_fmt_num(row['rank_icir'], 3)} |"
        )
      lines.append("")

    # 7. Coverage & Diagnostics
    diag = self.diagnostics
    lines.append("## Diagnostics & Health Checks")
    mktcap_corr = diag.get("mktcap_corr")
    mktcap_str = _fmt_num(mktcap_corr, 3) if mktcap_corr is not None else "N/A"
    lines.append(
      f"- **Coverage**: mean `{_fmt_pct(diag.get('coverage_mean'), 1)}`, "
      f"min `{_fmt_pct(diag.get('coverage_min'), 1)}` | "
      f"**Constant CS Days**: `{diag.get('constant_cs_days', 0)}` | "
      f"**Suspended**: `{_fmt_pct(diag.get('suspended_ratio'), 1)}` | "
      f"**Limit-Up Filtered**: `{_fmt_pct(diag.get('limit_filtered_ratio'), 2)}` | "
      f"**Log-MktCap Corr**: `{mktcap_str}`"
    )
    warnings_list = diag.get("warnings", [])
    if warnings_list:
      for w in warnings_list:
        lines.append(f"- **Warning**: {w}")
    else:
      lines.append("- **Alerts**: None (no major structural anomalies detected).")
    lines.append("")
    return "\n".join(lines)


@dataclasses.dataclass
class MultiFactorReport:
  """Aggregated evaluation report for multiple factors with pairwise correlation matrix."""

  reports: list[FactorReport]
  correlation_matrix: dict[str, dict[str, float]]
  primary_period: int

  def ranked_reports(self) -> list[FactorReport]:
    """Return reports sorted descending by absolute primary RankICIR."""
    return sorted(
      self.reports,
      key=lambda r: abs(r.primary_ic.rank_icir),
      reverse=True,
    )

  def to_dict(self, include_series: bool = True) -> dict[str, Any]:
    ranked = self.ranked_reports()
    return _sanitize_json({
      "primary_period": self.primary_period,
      "leaderboard": [r.summary_row() for r in ranked],
      "correlation_matrix": self.correlation_matrix,
      "factors": {r.name: r.to_dict(include_series=include_series) for r in self.reports},
    })

  def to_markdown(self) -> str:
    ranked = self.ranked_reports()
    lines: list[str] = []
    lines.append(f"# Multi-Factor Evaluation Summary ({len(ranked)} Factors, Primary Horizon: {self.primary_period}d)")
    lines.append("")
    lines.append("## 1. Factor Leaderboard (Sorted by |RankICIR|)")
    lines.append(
      "| Rank | Factor | RankIC | RankICIR | NW t-stat | Top Ann Excess | "
      "LS Ann Ret | Monotonicity | Daily Turn | Coverage | MktCap Corr |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx, r in enumerate(ranked, 1):
      row = r.summary_row()
      lines.append(
        f"| {idx} | `{row['factor']}` | {_fmt_num(row['rank_ic_mean'], 4)} | "
        f"{_fmt_num(row['rank_icir'], 3)} | {_fmt_num(row['nw_t_stat'], 2)} | "
        f"{_fmt_pct(row['top_ann_excess'])} | {_fmt_pct(row['ls_ann_return'])} | "
        f"{_fmt_num(row['monotonicity'], 3)} | {_fmt_pct(row['top_daily_turnover'], 2)} | "
        f"{_fmt_pct(row['avg_coverage'], 1)} | {_fmt_num(row['mktcap_corr'], 3)} |"
      )
    lines.append("")

    # Pairwise correlation matrix
    if len(ranked) > 1 and self.correlation_matrix:
      names = [r.name for r in ranked]
      lines.append("## 2. Pairwise Cross-Sectional Rank Correlation Matrix")
      header = "| Factor | " + " | ".join(f"`{n}`" for n in names) + " |"
      sep = "|---|" + "|".join(["---:"] * len(names)) + "|"
      lines.append(header)
      lines.append(sep)
      for ni in names:
        vals = [
          _fmt_num(self.correlation_matrix.get(ni, {}).get(nj), 3)
          for nj in names
        ]
        lines.append(f"| `{ni}` | " + " | ".join(vals) + " |")
      lines.append("")

    # Detailed sections: show all if <= 3 factors, otherwise show top factor detail
    detail_reports = ranked if len(ranked) <= 3 else ranked[:1]
    for r in detail_reports:
      lines.append("---")
      lines.append("")
      lines.append(r.to_markdown())

    return "\n".join(lines)


def _daily_frame(report: FactorReport) -> pl.DataFrame:
  p = report.primary_period
  qa = report.primary_quantile
  cols: dict[str, Any] = {
    "date": _date_strings(report.dates),
    f"ic_{p}d": report.ic_series[p],
    f"rank_ic_{p}d": report.rank_ic_series[p],
    "long_excess_nav": qa.long_excess_nav,
    "long_short_nav": qa.long_short_nav,
    "market_nav": qa.market_nav,
  }
  for q_label, q_nav in qa.quantile_navs.items():
    cols[f"{q_label}_nav"] = q_nav
  return pl.DataFrame(cols)


def export_report(
  report: FactorReport | MultiFactorReport,
  output_path: str,
) -> None:
  """Export a factor report to ``.json``, ``.md``, ``.csv``, ``.xlsx``, or ``.png``."""
  ext = output_path.rsplit(".", 1)[-1].lower() if "." in output_path else ""
  if ext == "json":
    payload = report.to_dict(include_series=True)
    with open(output_path, "wb") as f:
      f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY))
  elif ext == "md":
    with open(output_path, "w", encoding="utf-8") as f:
      f.write(report.to_markdown())
  elif ext == "csv":
    if isinstance(report, FactorReport):
      _daily_frame(report).write_csv(output_path)
    else:
      pl.DataFrame([r.summary_row() for r in report.ranked_reports()]).write_csv(output_path)
  elif ext == "xlsx":
    import xlsxwriter

    wb = xlsxwriter.Workbook(output_path, options={"nan_inf_to_errors": True})
    if isinstance(report, MultiFactorReport):
      ranked = report.ranked_reports()
      pl.DataFrame([r.summary_row() for r in ranked]).write_excel(
        workbook=wb, worksheet="leaderboard"
      )
      if report.correlation_matrix:
        names = [r.name for r in ranked]
        corr_rows = []
        for ni in names:
          row: dict[str, Any] = {"factor": ni}
          for nj in names:
            row[nj] = report.correlation_matrix.get(ni, {}).get(nj)
          corr_rows.append(row)
        pl.DataFrame(corr_rows).write_excel(workbook=wb, worksheet="correlation")
      primary_rep = ranked[0]
    else:
      primary_rep = report

    pl.DataFrame([primary_rep.ic_summaries[p].to_dict() for p in primary_rep.periods]).write_excel(
      workbook=wb, worksheet="ic_summary"
    )
    qa = primary_rep.primary_quantile
    pl.DataFrame(
      [q.to_dict() for q in qa.quantiles]
      + [qa.long_excess_market.to_dict(), qa.long_short.to_dict()]
    ).write_excel(workbook=wb, worksheet="quantiles")
    _daily_frame(primary_rep).write_excel(workbook=wb, worksheet="daily_series")
    if primary_rep.annual_breakdown:
      pl.DataFrame(primary_rep.annual_breakdown).write_excel(workbook=wb, worksheet="annual")
    if primary_rep.industry_breakdown:
      pl.DataFrame(primary_rep.industry_breakdown).write_excel(workbook=wb, worksheet="industry")
    wb.close()
  elif ext == "png":
    from mbt.factor.overview import render_factor_overview

    render_factor_overview(report, output_path)
  else:
    raise ValueError(
      f"Unsupported factor report output format '{ext}' for path '{output_path}'. "
      "Supported extensions: .json, .md, .csv, .xlsx, .png."
    )
  logger.info(f"factor report saved to: {output_path}")
