# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Strategy result view components for select, timing, and portfolio modes."""

from __future__ import annotations

import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from mbt.gui.chart import EquityCurvePanel
from mbt.gui.dpi import scaled, scaled_font


def _format_cell(val: Any, fmt: str | None = None) -> str:
  if val is None:
    return ""
  if isinstance(val, float):
    if val != val:  # NaN
      return ""
    if fmt == "pct":
      return f"{val:+.2%}"
    if fmt == "pct_plain":
      return f"{val:.2%}"
    if fmt == "money":
      return f"{val:,.2f}"
    if fmt == "money_signed":
      return f"{val:+,.2f}"
    if fmt == "price":
      return f"{val:.3f}"
    if fmt == "f4":
      return f"{val:.4f}"
    if fmt == "int":
      return f"{int(round(val))}"
    return f"{val:.4g}"
  return str(val)


def _sort_key(val: Any) -> tuple[int, Any]:
  if val is None or val == "":
    return (2, "")
  if isinstance(val, (int, float)):
    if val != val:
      return (2, "")
    return (0, float(val))
  s = str(val).strip()
  try:
    if s.endswith("%"):
      return (0, float(s[:-1]))
    return (0, float(s.replace(",", "")))
  except ValueError:
    return (1, s.lower())


class SortableTable(ttk.Frame):
  """Reusable sortable, filterable Treeview table with CSV export."""

  def __init__(
    self,
    parent: tk.Widget,
    on_select: Callable[[dict[str, Any]], None] | None = None,
    title: str = "",
  ):
    super().__init__(parent)
    self._on_select = on_select
    self._columns: list[dict[str, Any]] = []
    self._raw_rows: list[dict[str, Any]] = []
    self._filtered_rows: list[dict[str, Any]] = []
    self._row_by_iid: dict[str, dict[str, Any]] = {}
    self._sort_col: str | None = None
    self._sort_desc = False
    self._filter_var = tk.StringVar()

    self._build_ui(title)

  def _build_ui(self, title: str) -> None:
    bar = ttk.Frame(self)
    bar.pack(fill=tk.X, padx=6, pady=(4, 4))

    if title:
      ttk.Label(bar, text=title, font=scaled_font(self, 12, bold=True)).pack(
        side=tk.LEFT, padx=(0, 10)
      )

    self._count_lbl = ttk.Label(bar, text="0 条记录", foreground="#475569")
    self._count_lbl.pack(side=tk.LEFT)

    ttk.Button(bar, text="导出 CSV", command=self._export_csv).pack(side=tk.RIGHT)
    search_entry = ttk.Entry(bar, textvariable=self._filter_var, width=22)
    search_entry.pack(side=tk.RIGHT, padx=(4, 8))
    ttk.Label(bar, text="过滤:").pack(side=tk.RIGHT)
    self._filter_var.trace_add("write", lambda *_a: self._apply_filter())

    tree_wrap = ttk.Frame(self)
    tree_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
    tree_wrap.rowconfigure(0, weight=1)
    tree_wrap.columnconfigure(0, weight=1)

    self._tree = ttk.Treeview(tree_wrap, show="headings", selectmode="browse")
    ysb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self._tree.yview)
    xsb = ttk.Scrollbar(tree_wrap, orient=tk.HORIZONTAL, command=self._tree.xview)
    self._tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

    self._tree.grid(row=0, column=0, sticky=tk.NSEW)
    ysb.grid(row=0, column=1, sticky=tk.NS)
    xsb.grid(row=1, column=0, sticky=tk.EW)

    self._tree.tag_configure("buy", foreground="#DC2626")
    self._tree.tag_configure("sell", foreground="#059669")
    self._tree.bind("<<TreeviewSelect>>", self._handle_select)

  def set_data(
    self,
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    tag_fn: Callable[[dict[str, Any]], tuple[str, ...]] | None = None,
  ) -> None:
    """Populate columns and rows.

    Each column spec dict supports: ``key``, ``label``, ``width``, ``anchor``, ``fmt``.
    """
    self._columns = columns
    self._raw_rows = list(rows)
    self._tag_fn = tag_fn
    self._sort_col = None
    self._sort_desc = False

    col_ids = [c["key"] for c in columns]
    self._tree["columns"] = col_ids
    for c in columns:
      key = c["key"]
      self._tree.heading(
        key,
        text=c.get("label", key),
        command=lambda k=key: self._sort_by(k),
      )
      self._tree.column(
        key,
        width=scaled(c.get("width", 105), self),
        minwidth=scaled(60, self),
        anchor=c.get("anchor", tk.E if c.get("fmt") else tk.W),
        stretch=c.get("stretch", True),
      )

    self._apply_filter()

  def _apply_filter(self) -> None:
    q = self._filter_var.get().strip().lower()
    if not q:
      self._filtered_rows = list(self._raw_rows)
    else:
      keys = [c["key"] for c in self._columns]
      self._filtered_rows = [
        r
        for r in self._raw_rows
        if any(q in str(r.get(k, "")).lower() for k in keys)
      ]
    if self._sort_col:
      col_key = self._sort_col
      self._filtered_rows.sort(
        key=lambda r: _sort_key(r.get(col_key)), reverse=self._sort_desc
      )
    self._render_rows()

  def _sort_by(self, key: str) -> None:
    if self._sort_col == key:
      self._sort_desc = not self._sort_desc
    else:
      self._sort_col = key
      self._sort_desc = True
    self._apply_filter()

  def _render_rows(self) -> None:
    for item in self._tree.get_children():
      self._tree.delete(item)
    self._row_by_iid.clear()

    for idx, row in enumerate(self._filtered_rows):
      vals = [_format_cell(row.get(c["key"]), c.get("fmt")) for c in self._columns]
      tags = self._tag_fn(row) if getattr(self, "_tag_fn", None) else ()
      iid = f"r{idx}"
      self._tree.insert("", tk.END, iid=iid, values=vals, tags=tags)
      self._row_by_iid[iid] = row

    total = len(self._raw_rows)
    shown = len(self._filtered_rows)
    self._count_lbl.configure(
      text=f"{shown} / {total} 条" if shown != total else f"{total} 条记录"
    )

  def _handle_select(self, _event) -> None:
    if self._on_select is None:
      return
    sel = self._tree.selection()
    if not sel:
      return
    row = self._row_by_iid.get(sel[0])
    if row is not None:
      self._on_select(row)

  def _export_csv(self) -> None:
    if not self._filtered_rows:
      messagebox.showinfo("提示", "当前没有可导出的数据。", parent=self)
      return
    path = filedialog.asksaveasfilename(
      parent=self,
      title="导出表格为 CSV",
      defaultextension=".csv",
      filetypes=[("CSV", "*.csv")],
    )
    if not path:
      return
    keys = [c["key"] for c in self._columns]
    headers = [c.get("label", c["key"]) for c in self._columns]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(headers)
      for row in self._filtered_rows:
        writer.writerow([row.get(k, "") for k in keys])


class SelectResultView(ttk.Frame):
  """Result view for stock selector runs: summary bar + stocks/factors table."""

  def __init__(self, parent: tk.Widget):
    super().__init__(parent)
    self._summary_var = tk.StringVar(value="")
    ttk.Label(
      self,
      textvariable=self._summary_var,
      padding=(8, 6),
      font=scaled_font(self, 13, bold=True),
    ).pack(fill=tk.X)
    self._table = SortableTable(self, title="选股结果与因子明细")
    self._table.pack(fill=tk.BOTH, expand=True)

  def show_result(self, res: dict[str, Any]) -> None:
    """Populate selector result table."""
    stocks = res.get("stocks") or []
    cols = res.get("columns") or ["obj"]
    raw_rows = res.get("rows") or []
    sel_date = res.get("select_date") or "-"
    elapsed = float(res.get("elapsed", 0.0))
    name = res.get("strategy_name", "Selector")

    self._summary_var.set(
      f"[{name}]  选股日期: {sel_date}  |  入选股票: {len(stocks)} 只  |  耗时: {elapsed:.2f}s"
    )

    col_specs: list[dict[str, Any]] = []
    sample_row = raw_rows[0] if raw_rows else []
    for idx, col_name in enumerate(cols):
      is_num = idx < len(sample_row) and isinstance(sample_row[idx], (int, float))
      col_specs.append({
        "key": col_name,
        "label": col_name,
        "width": 115 if col_name != "obj" else 110,
        "anchor": tk.E if is_num else tk.W,
        "fmt": None,
      })

    dict_rows = [
      {col_name: row[i] if i < len(row) else None for i, col_name in enumerate(cols)}
      for row in raw_rows
    ]
    self._table.set_data(col_specs, dict_rows)


class TimingResultView(ttk.Frame):
  """Result view for timing strategies: equity curve on top, stock metrics table below."""

  def __init__(self, parent: tk.Widget):
    super().__init__(parent)
    self._res: dict[str, Any] | None = None
    self._reports_by_obj: dict[str, dict[str, Any]] = {}

    header = ttk.Frame(self, padding=(8, 4))
    header.pack(fill=tk.X)
    self._summary_var = tk.StringVar(value="")
    ttk.Label(header, textvariable=self._summary_var, font=scaled_font(self, 13, bold=True)).pack(
      side=tk.LEFT
    )
    self._reset_curve_btn = ttk.Button(
      header, text="显示等权汇总曲线", command=self._show_summary_curve
    )
    self._reset_curve_btn.pack(side=tk.RIGHT)

    paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
    paned.pack(fill=tk.BOTH, expand=True)

    self._chart = EquityCurvePanel(paned)
    paned.add(self._chart, weight=3)

    self._table = SortableTable(
      paned,
      on_select=self._on_row_selected,
      title="各标的策略指标（点击行切换上方曲线）",
    )
    paned.add(self._table, weight=2)

  def show_result(self, res: dict[str, Any]) -> None:
    """Render timing backtest reports and default curve."""
    self._res = res
    reports = res.get("reports") or []
    self._reports_by_obj = {r["obj"]: r for r in reports}
    elapsed = float(res.get("elapsed", 0.0))
    name = res.get("strategy_name", "Strategy")

    n = len(reports)
    pos = sum(1 for r in reports if float(r["metrics"].get("profit", 0.0)) > 0)
    beat_bh = sum(
      1
      for r in reports
      if float(r["metrics"].get("profit", 0.0)) > float(r["metrics"].get("profit_original", 0.0))
    )
    self._summary_var.set(
      f"[{name}]  标的数: {n}  |  正收益: {pos}/{n}  |  跑赢买入持有: {beat_bh}/{n}  |  耗时: {elapsed:.2f}s"
    )

    if res.get("summary_curve"):
      self._reset_curve_btn.state(["!disabled"])
    else:
      self._reset_curve_btn.state(["disabled"])

    columns = [
      {"key": "obj", "label": "标的", "width": 95, "anchor": tk.W},
      {"key": "profit", "label": "策略收益", "width": 92, "fmt": "pct"},
      {"key": "profit_original", "label": "买入持有", "width": 92, "fmt": "pct"},
      {"key": "benchmark_profit", "label": "基准收益", "width": 92, "fmt": "pct"},
      {"key": "alpha", "label": "Alpha", "width": 80, "fmt": "f4"},
      {"key": "beta", "label": "Beta", "width": 72, "fmt": "f4"},
      {"key": "sharpe", "label": "Sharpe", "width": 78, "fmt": "f4"},
      {"key": "sortino", "label": "Sortino", "width": 78, "fmt": "f4"},
      {"key": "max_drawdown", "label": "最大回撤", "width": 90, "fmt": "pct"},
      {"key": "calmar", "label": "Calmar", "width": 78, "fmt": "f4"},
      {"key": "win_rate", "label": "胜率", "width": 76, "fmt": "pct_plain"},
      {"key": "profit_factor", "label": "盈亏比", "width": 76, "fmt": "f4"},
      {"key": "fee_rate", "label": "费率", "width": 74, "fmt": "pct_plain"},
      {"key": "trade_count", "label": "交易次数", "width": 78, "fmt": "int"},
    ]
    rows = [r["metrics"] for r in reports]
    self._table.set_data(columns, rows)

    if res.get("summary_curve"):
      self._show_summary_curve()
    elif reports:
      self._show_obj_curve(reports[0])
    else:
      self._chart.clear()

  def _show_summary_curve(self) -> None:
    if not self._res or not self._res.get("summary_curve"):
      return
    sc = self._res["summary_curve"]
    self._chart.set_series(
      dates=sc["dates"],
      ror=sc["ror"],
      ror_original=sc["ror_original"],
      benchmark_ror=sc["benchmark_ror"],
      title=sc["label"],
      orig_label="等权买入持有",
      bench_label=f"基准 ({sc.get('benchmark_obj', '')})",
    )

  def _show_obj_curve(self, rep: dict[str, Any]) -> None:
    self._chart.set_series(
      dates=rep["dates"],
      ror=rep["ror"],
      ror_original=rep["ror_original"],
      benchmark_ror=rep["benchmark_ror"],
      title=f"{rep['obj']} 收益曲线",
      orig_label="买入持有",
      bench_label=f"基准 ({rep.get('benchmark_obj', '')})",
    )

  def _on_row_selected(self, row: dict[str, Any]) -> None:
    obj = row.get("obj")
    if obj is None:
      return
    rep = self._reports_by_obj.get(str(obj))
    if rep is not None:
      self._show_obj_curve(rep)


class PortfolioResultView(ttk.Frame):
  """Result view for shared-cash portfolio strategies."""

  def __init__(self, parent: tk.Widget):
    super().__init__(parent)
    self._kpi_var = tk.StringVar(value="")

    kpi_lbl = ttk.Label(
      self,
      textvariable=self._kpi_var,
      padding=(8, 6),
      font=scaled_font(self, 13, bold=True),
    )
    kpi_lbl.pack(fill=tk.X)

    paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
    paned.pack(fill=tk.BOTH, expand=True)

    self._chart = EquityCurvePanel(paned)
    paned.add(self._chart, weight=3)

    nb = ttk.Notebook(paned)
    paned.add(nb, weight=2)

    self._records_table = SortableTable(nb, title="逐笔调仓委托成交记录")
    nb.add(self._records_table, text="调仓列表")

    self._trades_table = SortableTable(nb, title="持仓盈亏周期")
    nb.add(self._trades_table, text="持仓盈亏周期")

    self._metrics_frame = ttk.Frame(nb)
    nb.add(self._metrics_frame, text="组合指标与归因")
    self._build_metrics_tab()

  def _build_metrics_tab(self) -> None:
    split = ttk.PanedWindow(self._metrics_frame, orient=tk.HORIZONTAL)
    split.pack(fill=tk.BOTH, expand=True)

    self._metrics_table = SortableTable(split, title="核心指标与诊断")
    split.add(self._metrics_table, weight=1)

    self._attrib_table = SortableTable(split, title="个股收益归因")
    split.add(self._attrib_table, weight=2)

  def show_result(self, res: dict[str, Any]) -> None:
    """Render portfolio equity curve, KPI summary, rebalance records, and trades."""
    m = res.get("metrics") or {}
    name = res.get("strategy_name", "PortfolioStrategy")
    elapsed = float(res.get("elapsed", 0.0))
    bench_obj = res.get("benchmark_obj", "基准")

    self._kpi_var.set(
      f"[{name}]  "
      f"累计收益: {m.get('profit', 0.0):+.2%}  |  "
      f"基准({bench_obj}): {m.get('benchmark_profit', 0.0):+.2%}  |  "
      f"Sharpe: {m.get('sharpe', 0.0):.2f}  |  "
      f"最大回撤: {m.get('max_drawdown', 0.0):.2%}  |  "
      f"胜率: {m.get('win_rate', 0.0):.1%}  |  "
      f"调仓/成交: {len(res.get('rebalance_records') or [])}笔  |  "
      f"耗时: {elapsed:.2f}s"
    )

    ror_data = res.get("ror")
    self._chart.set_series(
      dates=res.get("dates") or [],
      ror=ror_data if ror_data is not None else [],
      ror_original=res.get("ror_original"),
      benchmark_ror=res.get("benchmark_ror"),
      title=f"{name} 组合净值收益率",
      orig_label="池内等权买入持有",
      bench_label=f"基准 ({bench_obj})",
    )

    # Tab 1: Rebalance records
    rec_cols = [
      {"key": "date", "label": "日期", "width": 100, "anchor": tk.W},
      {"key": "obj", "label": "标的", "width": 95, "anchor": tk.W},
      {"key": "action", "label": "方向", "width": 65, "anchor": tk.CENTER},
      {"key": "shares", "label": "成交股数", "width": 85, "fmt": "int"},
      {"key": "deal_price", "label": "成交价", "width": 85, "fmt": "price"},
      {"key": "amount", "label": "成交金额", "width": 105, "fmt": "money"},
      {"key": "fee", "label": "手续费", "width": 80, "fmt": "money"},
      {"key": "pos_shares", "label": "持仓股数", "width": 85, "fmt": "int"},
      {"key": "cash", "label": "可用现金", "width": 110, "fmt": "money"},
    ]
    self._records_table.set_data(
      rec_cols,
      res.get("rebalance_records") or [],
      tag_fn=lambda r: ("buy",) if r.get("action") == "BUY" else ("sell",) if r.get("action") == "SELL" else (),
    )

    # Tab 2: Closed/holding position cycles
    trade_cols = [
      {"key": "obj", "label": "标的", "width": 100, "anchor": tk.W},
      {"key": "entry_date", "label": "买入日期", "width": 105, "anchor": tk.W},
      {"key": "exit_date", "label": "卖出日期", "width": 105, "anchor": tk.W},
      {"key": "profit_rate", "label": "持仓收益率", "width": 105, "fmt": "pct"},
      {"key": "profit_amount", "label": "持仓盈亏额", "width": 115, "fmt": "money_signed"},
    ]
    self._trades_table.set_data(
      trade_cols,
      res.get("trades") or [],
      tag_fn=lambda r: ("buy",) if float(r.get("profit_amount", 0.0)) > 0 else ("sell",) if float(r.get("profit_amount", 0.0)) < 0 else (),
    )

    # Tab 3: Core metrics + diagnostics + attribution
    metric_rows = [{"item": k, "value": _format_cell(v)} for k, v in m.items()]
    diag = res.get("diagnostics")
    if isinstance(diag, dict):
      for k, v in diag.items():
        metric_rows.append({"item": f"[diag] {k}", "value": str(v)})
    self._metrics_table.set_data(
      [
        {"key": "item", "label": "指标项", "width": 160, "anchor": tk.W},
        {"key": "value", "label": "数值", "width": 130, "anchor": tk.E},
      ],
      metric_rows,
    )

    attrib_cols = [
      {"key": "obj", "label": "标的", "width": 95, "anchor": tk.W},
      {"key": "trade_count", "label": "交易轮次", "width": 80, "fmt": "int"},
      {"key": "win_rate", "label": "胜率", "width": 80, "fmt": "pct_plain"},
      {"key": "total_profit", "label": "累计盈亏", "width": 105, "fmt": "money_signed"},
      {"key": "capital_contribution", "label": "资金贡献", "width": 90, "fmt": "pct"},
      {"key": "avg_return", "label": "单次均收益", "width": 95, "fmt": "pct"},
    ]
    self._attrib_table.set_data(attrib_cols, res.get("attribution") or [])
