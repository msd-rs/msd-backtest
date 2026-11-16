# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Modal dialogs for managing strategies and shortcuts in the mbt GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from mbt.gui.config import (
  RUN_MODE_LABELS,
  RUN_MODE_SELECT,
  ShortcutConfig,
  VALID_RUN_MODES,
)
from mbt.gui.dpi import scaled
from mbt.gui.inspect_strategy import ParamSpec, StrategyMetadata


class RenameDialog(tk.Toplevel):
  """Simple modal dialog for renaming a strategy."""

  def __init__(self, parent: tk.Widget, initial_name: str, title: str = "重命名策略"):
    super().__init__(parent)
    self.title(title)
    self.resizable(False, False)
    self.transient(parent)
    self.result: str | None = None

    self._name_var = tk.StringVar(value=initial_name)

    body = ttk.Frame(self, padding=14)
    body.pack(fill=tk.BOTH, expand=True)

    ttk.Label(body, text="策略显示名称:").pack(anchor=tk.W, pady=(0, 6))
    entry = ttk.Entry(body, textvariable=self._name_var, width=38)
    entry.pack(fill=tk.X, pady=(0, 12))
    entry.select_range(0, tk.END)
    entry.focus_set()

    btns = ttk.Frame(body)
    btns.pack(fill=tk.X)
    ttk.Button(btns, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=(6, 0))
    ttk.Button(btns, text="确定", command=self._on_ok).pack(side=tk.RIGHT)

    self.bind("<Return>", lambda _e: self._on_ok())
    self.bind("<Escape>", lambda _e: self.destroy())
    self.grab_set()
    self._center_on_parent(parent)

  def _center_on_parent(self, parent: tk.Widget) -> None:
    self.update_idletasks()
    px = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
    py = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)
    self.geometry(f"+{px}+{py}")

  def _on_ok(self) -> None:
    val = self._name_var.get().strip()
    if not val:
      messagebox.showwarning("提示", "名称不能为空。", parent=self)
      return
    self.result = val
    self.destroy()


class ShortcutDialog(tk.Toplevel):
  """Modal editor for creating or editing a strategy run shortcut."""

  def __init__(
    self,
    parent: tk.Widget,
    shortcut: ShortcutConfig,
    metadata: StrategyMetadata | None,
    title: str = "编辑快捷路径",
  ):
    super().__init__(parent)
    self.title(title)
    w = min(scaled(620, parent), max(600, self.winfo_screenwidth() - 80))
    h = min(scaled(640, parent), max(520, self.winfo_screenheight() - 80))
    self.geometry(f"{w}x{h}")
    self.minsize(min(scaled(540, parent), w), min(scaled(500, parent), h))
    self.transient(parent)

    self._orig = shortcut
    self._metadata = metadata
    self.result: ShortcutConfig | None = None

    self._name_var = tk.StringVar(value=shortcut.name)
    self._mode_var = tk.StringVar(value=shortcut.mode)
    self._objs_var = tk.StringVar(value=shortcut.objs)
    self._use_internal_sel_var = tk.BooleanVar(value=shortcut.use_internal_selector)
    self._begin_var = tk.StringVar(value=shortcut.begin)
    self._end_var = tk.StringVar(value=shortcut.end)
    self._select_date_var = tk.StringVar(value=shortcut.select_date)
    self._benchmark_var = tk.StringVar(value=shortcut.benchmark)
    self._cash_var = tk.StringVar(value=shortcut.cash)
    self._exec_mode_var = tk.StringVar(value=shortcut.exec_mode)
    self._warmup_var = tk.StringVar(value=shortcut.warmup_bars)
    self._msd_var = tk.StringVar(value=shortcut.msd_host)
    self._extra_var = tk.StringVar(value=shortcut.extra_args)

    self._param_vars: dict[str, tk.StringVar] = {
      k: tk.StringVar(value=v) for k, v in shortcut.strategy_args.items()
    }

    self._build_ui()
    self._on_mode_changed()
    self.grab_set()
    self._center_on_parent(parent)

  def _center_on_parent(self, parent: tk.Widget) -> None:
    self.update_idletasks()
    px = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
    py = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)
    self.geometry(f"+{px}+{py}")

  def _build_ui(self) -> None:
    root = ttk.Frame(self, padding=12)
    root.pack(fill=tk.BOTH, expand=True)

    # Top: Name & Mode
    basic = ttk.LabelFrame(root, text="基本设置", padding=10)
    basic.pack(fill=tk.X, pady=(0, 8))
    basic.columnconfigure(1, weight=1)

    ttk.Label(basic, text="快捷路径名称:").grid(row=0, column=0, sticky=tk.W, pady=3, padx=(0, 8))
    name_entry = ttk.Entry(basic, textvariable=self._name_var)
    name_entry.grid(row=0, column=1, sticky=tk.EW, pady=3)
    name_entry.focus_set()

    ttk.Label(basic, text="运行模式:").grid(row=1, column=0, sticky=tk.W, pady=3, padx=(0, 8))
    mode_row = ttk.Frame(basic)
    mode_row.grid(row=1, column=1, sticky=tk.W, pady=3)
    for m in VALID_RUN_MODES:
      ttk.Radiobutton(
        mode_row,
        text=f"{RUN_MODE_LABELS[m]} ({m})",
        value=m,
        variable=self._mode_var,
        command=self._on_mode_changed,
      ).pack(side=tk.LEFT, padx=(0, 10))

    if self._metadata is not None:
      detected_parts = []
      if self._metadata.selector_name:
        detected_parts.append(f"Selector: {self._metadata.selector_name}")
      if self._metadata.strategy_name:
        kind = "PortfolioStrategy" if self._metadata.has_portfolio_strategy else "Strategy"
        detected_parts.append(f"{kind}: {self._metadata.strategy_name}")
      ttk.Label(
        basic,
        text="文件检测: " + (" | ".join(detected_parts) if detected_parts else "未知"),
        foreground="#475569",
      ).grid(row=2, column=1, sticky=tk.W, pady=(2, 0))

    # Run parameters
    run_box = ttk.LabelFrame(root, text="运行参数", padding=10)
    run_box.pack(fill=tk.X, pady=(0, 8))
    run_box.columnconfigure(1, weight=1)

    self._objs_lbl = ttk.Label(run_box, text="标的池 (-s):")
    self._objs_lbl.grid(row=0, column=0, sticky=tk.W, pady=3, padx=(0, 8))
    objs_row = ttk.Frame(run_box)
    objs_row.grid(row=0, column=1, sticky=tk.EW, pady=3)
    objs_row.columnconfigure(0, weight=1)
    ttk.Entry(objs_row, textvariable=self._objs_var).grid(row=0, column=0, sticky=tk.EW)

    preset_btn = ttk.Menubutton(objs_row, text="预设/文件")
    preset_menu = tk.Menu(preset_btn, tearoff=False)
    for preset in ("A_STOCKS_EXCLUDE_ST", "A_STOCKS", "FOUNDS", "SH600000 SZ000001"):
      preset_menu.add_command(
        label=preset, command=lambda p=preset: self._objs_var.set(p)
      )
    preset_menu.add_separator()
    if self._metadata is not None and self._metadata.has_selector:
      sel_title = self._metadata.selector_name or "Selector"
      preset_menu.add_command(
        label=f"使用内置选股器 ({sel_title})",
        command=self._enable_internal_selector,
      )
    preset_menu.add_command(label="选择标的/选股器文件...", command=self._pick_objs_file)
    preset_btn["menu"] = preset_menu
    preset_btn.grid(row=0, column=1, padx=(6, 0))

    has_sel = self._metadata is not None and self._metadata.has_selector
    sel_name = (self._metadata.selector_name or "Selector") if has_sel else "未检测到 Selector"
    self._internal_sel_chk = ttk.Checkbutton(
      run_box,
      text=f"使用本策略内置选股器 ({sel_name}) 生成初始股票池",
      variable=self._use_internal_sel_var,
      command=self._update_objs_label,
    )
    if self._metadata is not None and not has_sel:
      self._use_internal_sel_var.set(False)
      self._internal_sel_chk.state(["disabled"])

    # Mode-dependent frames
    self._select_frame = ttk.Frame(run_box)
    self._select_frame.columnconfigure(1, weight=1)
    ttk.Label(self._select_frame, text="选股日期 (-d):").grid(
      row=0, column=0, sticky=tk.W, pady=3, padx=(0, 8)
    )
    ttk.Entry(self._select_frame, textvariable=self._select_date_var).grid(
      row=0, column=1, sticky=tk.EW, pady=3
    )
    ttk.Label(self._select_frame, text="留空默认最新交易日，格式 YYYY-MM-DD", foreground="#64748B").grid(
      row=1, column=1, sticky=tk.W
    )

    self._bt_frame = ttk.Frame(run_box)
    self._bt_frame.columnconfigure(1, weight=1)
    self._bt_frame.columnconfigure(3, weight=1)

    ttk.Label(self._bt_frame, text="开始日期 (-b):").grid(row=0, column=0, sticky=tk.W, pady=3, padx=(0, 6))
    ttk.Entry(self._bt_frame, textvariable=self._begin_var).grid(row=0, column=1, sticky=tk.EW, pady=3, padx=(0, 12))
    ttk.Label(self._bt_frame, text="结束日期 (-e):").grid(row=0, column=2, sticky=tk.W, pady=3, padx=(0, 6))
    ttk.Entry(self._bt_frame, textvariable=self._end_var).grid(row=0, column=3, sticky=tk.EW, pady=3)

    ttk.Label(self._bt_frame, text="基准 (-B):").grid(row=1, column=0, sticky=tk.W, pady=3, padx=(0, 6))
    ttk.Entry(self._bt_frame, textvariable=self._benchmark_var).grid(row=1, column=1, sticky=tk.EW, pady=3, padx=(0, 12))
    ttk.Label(self._bt_frame, text="初始资金 (-c):").grid(row=1, column=2, sticky=tk.W, pady=3, padx=(0, 6))
    ttk.Entry(self._bt_frame, textvariable=self._cash_var).grid(row=1, column=3, sticky=tk.EW, pady=3)

    ttk.Label(self._bt_frame, text="撮合方式:").grid(row=2, column=0, sticky=tk.W, pady=3, padx=(0, 6))
    exec_cb = ttk.Combobox(
      self._bt_frame,
      textvariable=self._exec_mode_var,
      values=("", "close", "next_open"),
      state="readonly",
    )
    exec_cb.grid(row=2, column=1, sticky=tk.EW, pady=3, padx=(0, 12))
    ttk.Label(self._bt_frame, text="预热 Bar:").grid(row=2, column=2, sticky=tk.W, pady=3, padx=(0, 6))
    ttk.Entry(self._bt_frame, textvariable=self._warmup_var).grid(row=2, column=3, sticky=tk.EW, pady=3)

    ttk.Label(run_box, text="MSD 服务 (-m):").grid(row=3, column=0, sticky=tk.W, pady=3, padx=(0, 8))
    ttk.Entry(run_box, textvariable=self._msd_var).grid(row=3, column=1, sticky=tk.EW, pady=3)

    # Custom __init__ parameters (scrollable)
    self._params_box = ttk.LabelFrame(root, text="策略自定义参数 (-a key=value)", padding=8)
    self._params_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

    canvas = tk.Canvas(self._params_box, highlightthickness=0, height=scaled(150, self))
    scrollbar = ttk.Scrollbar(self._params_box, orient=tk.VERTICAL, command=canvas.yview)
    self._params_inner = ttk.Frame(canvas)
    self._params_inner.bind(
      "<Configure>",
      lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    self._params_win = canvas.create_window((0, 0), window=self._params_inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind(
      "<Configure>",
      lambda e: canvas.itemconfigure(self._params_win, width=e.width),
    )
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Extra key=value fallback
    extra_box = ttk.Frame(root)
    extra_box.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(extra_box, text="额外参数 (-a):").pack(side=tk.LEFT, padx=(0, 8))
    ttk.Entry(extra_box, textvariable=self._extra_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Bottom buttons
    btns = ttk.Frame(root)
    btns.pack(fill=tk.X)
    ttk.Button(btns, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=(6, 0))
    ttk.Button(btns, text="保存", command=self._on_save).pack(side=tk.RIGHT)

    self.bind("<Escape>", lambda _e: self.destroy())

  def _enable_internal_selector(self) -> None:
    if self._mode_var.get() == RUN_MODE_SELECT:
      return
    self._use_internal_sel_var.set(True)
    self._update_objs_label()

  def _update_objs_label(self) -> None:
    if self._mode_var.get() != RUN_MODE_SELECT and self._use_internal_sel_var.get():
      self._objs_lbl.configure(text="候选池 (可选):")
    else:
      self._objs_lbl.configure(text="标的池 (-s):")

  def _pick_objs_file(self) -> None:
    path = filedialog.askopenfilename(
      parent=self,
      title="选择股票池文件或 Selector 文件",
      filetypes=[("Python / Text", "*.py *.txt"), ("All Files", "*.*")],
    )
    if path:
      self._objs_var.set(path)

  def _on_mode_changed(self) -> None:
    mode = self._mode_var.get()
    self._internal_sel_chk.grid_forget()
    self._select_frame.grid_forget()
    self._bt_frame.grid_forget()
    if mode == RUN_MODE_SELECT:
      self._select_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)
    else:
      self._internal_sel_chk.grid(row=1, column=1, sticky=tk.W, pady=(2, 2))
      self._bt_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)
    self._update_objs_label()
    self._rebuild_param_fields(mode)

  def _rebuild_param_fields(self, mode: str) -> None:
    for w in self._params_inner.winfo_children():
      w.destroy()

    specs: list[ParamSpec] = []
    if self._metadata is not None:
      specs = self._metadata.params_for_mode(mode)

    known_names = {s.name for s in specs}
    for k, var in self._param_vars.items():
      if k not in known_names:
        specs.append(ParamSpec(name=k, default_str=var.get(), type_label="saved"))

    if not specs:
      ttk.Label(
        self._params_inner,
        text="该策略未声明自定义 __init__ 参数，可在下方“额外参数”填写系统参数。",
        foreground="#64748B",
      ).pack(anchor=tk.W, pady=6)
      return

    self._params_inner.columnconfigure(1, weight=1)
    for row_idx, spec in enumerate(specs):
      if spec.name not in self._param_vars:
        self._param_vars[spec.name] = tk.StringVar(value=spec.default_str)
      var = self._param_vars[spec.name]
      label_text = f"{spec.name} ({spec.type_label})" if spec.type_label else spec.name
      ttk.Label(self._params_inner, text=label_text).grid(
        row=row_idx, column=0, sticky=tk.W, padx=(2, 10), pady=3
      )
      ttk.Entry(self._params_inner, textvariable=var).grid(
        row=row_idx, column=1, sticky=tk.EW, padx=(0, 6), pady=3
      )

  def _on_save(self) -> None:
    name = self._name_var.get().strip()
    if not name:
      messagebox.showwarning("提示", "快捷路径名称不能为空。", parent=self)
      return
    mode = self._mode_var.get()
    if mode not in VALID_RUN_MODES:
      messagebox.showwarning("提示", f"无效的运行模式: {mode}", parent=self)
      return

    # Keep only params currently shown or non-empty
    active_names = set()
    if self._metadata is not None:
      active_names.update(p.name for p in self._metadata.params_for_mode(mode))
    strategy_args = {
      k: v.get().strip()
      for k, v in self._param_vars.items()
      if (k in active_names or k in self._orig.strategy_args) and v.get().strip() != ""
    }

    use_internal = (
      bool(self._use_internal_sel_var.get()) if mode != RUN_MODE_SELECT else False
    )
    if use_internal and self._metadata is not None and not self._metadata.has_selector:
      messagebox.showwarning("提示", "当前策略文件未定义 Selector 子类。", parent=self)
      return

    self.result = ShortcutConfig(
      id=self._orig.id,
      name=name,
      mode=mode,
      objs=self._objs_var.get().strip(),
      use_internal_selector=use_internal,
      begin=self._begin_var.get().strip(),
      end=self._end_var.get().strip(),
      select_date=self._select_date_var.get().strip(),
      benchmark=self._benchmark_var.get().strip() or "SH000300",
      cash=self._cash_var.get().strip(),
      exec_mode=self._exec_mode_var.get().strip(),
      warmup_bars=self._warmup_var.get().strip(),
      msd_host=self._msd_var.get().strip(),
      strategy_args=strategy_args,
      extra_args=self._extra_var.get().strip(),
    )
    self.destroy()
