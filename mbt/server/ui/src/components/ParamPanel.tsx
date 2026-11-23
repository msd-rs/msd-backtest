import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Check, Link2, Play, RotateCcw, Save, SaveAll, SlidersHorizontal, Square } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SelectNative } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  RUN_MODE_LABELS,
  type DraftParams,
  type ParamSpec,
  type RunMode,
  type ShortcutRecord,
  type StrategyRecord,
  type SystemDefaults,
} from "@/types/api";

interface ParamPanelProps {
  strategy: StrategyRecord;
  shortcut: ShortcutRecord | null;
  defaults: SystemDefaults;
  draft: DraftParams;
  isDirty: boolean;
  isRunning: boolean;
  onChangeDraft: (updater: (prev: DraftParams) => DraftParams) => void;
  onResetDraft: () => void;
  onRunDraft: () => void;
  onStopRun: () => void;
  onSaveShortcut: () => Promise<void>;
  onSaveAsShortcut: (name: string) => Promise<void>;
}

const OBJS_PRESETS = ["A_STOCKS_EXCLUDE_ST", "A_STOCKS", "FOUNDS", "SH600000 SZ000001"];

export function ParamPanel({
  strategy,
  shortcut,
  defaults,
  draft,
  isDirty,
  isRunning,
  onChangeDraft,
  onResetDraft,
  onRunDraft,
  onStopRun,
  onSaveShortcut,
  onSaveAsShortcut,
}: ParamPanelProps) {
  const panelId = useId();
  const headerRef = useRef<HTMLElement | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const [paramsOpen, setParamsOpen] = useState(false);
  const [saveAsOpen, setSaveAsOpen] = useState(false);
  const [newScName, setNewScName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveErr, setSaveErr] = useState("");
  const [copiedLink, setCopiedLink] = useState(false);

  const supportsPopover =
    typeof HTMLElement !== "undefined" &&
    Object.hasOwn(HTMLElement.prototype, "popover");

  useEffect(() => {
    if (!supportsPopover || !paramsOpen) return;
    if (typeof CSS !== "undefined" && CSS.supports("top: anchor(bottom)")) return;
    const syncPosition = () => {
      const hdr = headerRef.current;
      const pop = popoverRef.current;
      if (!hdr || !pop) return;
      const rect = hdr.getBoundingClientRect();
      pop.style.setProperty("--param-popover-top", `${Math.round(rect.bottom)}px`);
      pop.style.setProperty("--param-popover-left", `${Math.round(rect.left)}px`);
      pop.style.setProperty("--param-popover-width", `${Math.round(rect.width)}px`);
    };
    syncPosition();
    window.addEventListener("resize", syncPosition);
    const obs = headerRef.current ? new ResizeObserver(syncPosition) : null;
    if (obs && headerRef.current) obs.observe(headerRef.current);
    return () => {
      window.removeEventListener("resize", syncPosition);
      obs?.disconnect();
    };
  }, [supportsPopover, paramsOpen]);

  useEffect(() => {
    if (supportsPopover || !paramsOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setParamsOpen(false);
    };
    const onPointerDown = (e: PointerEvent) => {
      if (!headerRef.current?.contains(e.target as Node)) {
        setParamsOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("pointerdown", onPointerDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("pointerdown", onPointerDown);
    };
  }, [supportsPopover, paramsOpen]);

  const meta = strategy.metadata;
  const availableModes: RunMode[] = useMemo(() => {
    const list = meta?.available_modes?.length ? meta.available_modes : [strategy.default_mode];
    return list.includes(draft.mode) ? list : [draft.mode, ...list];
  }, [meta, strategy.default_mode, draft.mode]);

  const customParams: ParamSpec[] = useMemo(() => {
    if (!meta) return [];
    if (draft.mode === "select") return meta.selector_params || [];
    if (draft.mode === "factor") return meta.factor_params || [];
    const merged: ParamSpec[] = [];
    const seen = new Set<string>();
    for (const p of [...(meta.strategy_params || []), ...(meta.selector_params || [])]) {
      if (!seen.has(p.name)) {
        seen.add(p.name);
        merged.push(p);
      }
    }
    return merged;
  }, [meta, draft.mode]);

  function setField<K extends keyof DraftParams>(key: K, val: DraftParams[K]) {
    onChangeDraft((prev) => ({ ...prev, [key]: val }));
  }

  function setStrategyArg(name: string, val: string) {
    onChangeDraft((prev) => ({
      ...prev,
      strategy_args: { ...prev.strategy_args, [name]: val },
    }));
  }

  async function handleSave() {
    setSaving(true);
    setSaveErr("");
    try {
      if (shortcut) {
        await onSaveShortcut();
      } else {
        setNewScName(`${strategy.name} - 方案`);
        setSaveAsOpen(true);
      }
    } catch (err) {
      setSaveErr(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleConfirmSaveAs(e: React.FormEvent) {
    e.preventDefault();
    if (!newScName.trim()) return;
    setSaving(true);
    setSaveErr("");
    try {
      await onSaveAsShortcut(newScName.trim());
      setSaveAsOpen(false);
    } catch (err) {
      setSaveErr(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleCopyLink() {
    const base = window.location.origin;
    const path = shortcut
      ? `/strategies/${encodeURIComponent(strategy.id)}/shortcuts/${encodeURIComponent(shortcut.id)}`
      : `/strategies/${encodeURIComponent(strategy.id)}`;
    const link = `${base}${path}`;
    try {
      await navigator.clipboard.writeText(link);
      setCopiedLink(true);
      window.setTimeout(() => setCopiedLink(false), 1800);
    } catch {
      window.prompt("复制策略链接", link);
    }
  }

  return (
    <section ref={headerRef} className="mbt-param-anchor relative shrink-0 border-b bg-card">
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h2 className="truncate text-sm font-semibold">
              {strategy.name}
              {shortcut ? ` / ${shortcut.name}` : " (默认参数)"}
            </h2>
            <button
              type="button"
              onClick={handleCopyLink}
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
              title="复制当前策略页面链接"
              aria-label="复制当前策略页面链接"
            >
              {copiedLink ? (
                <>
                  <Check className="h-3 w-3 text-emerald-500" aria-hidden="true" />
                  <span className="text-emerald-600 dark:text-emerald-400">已复制</span>
                </>
              ) : (
                <Link2 className="h-3 w-3" aria-hidden="true" />
              )}
            </button>
            <Badge variant={draft.mode}>{RUN_MODE_LABELS[draft.mode]}</Badge>
            {isDirty && (
              <Badge variant="outline" className="border-amber-500/50 text-amber-600 dark:text-amber-400">
                未保存草稿
              </Badge>
            )}
            {meta?.author && (
              <span className="text-[11px] text-muted-foreground">作者: {meta.author}</span>
            )}
          </div>
          <p className="truncate text-[11px] text-muted-foreground" title={strategy.path}>
            {strategy.path}
            {meta?.docstring ? ` · ${meta.docstring}` : ""}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {isRunning ? (
            <Button variant="destructive" size="sm" onClick={onStopRun}>
              <Square className="h-3.5 w-3.5" aria-hidden="true" />
              停止
            </Button>
          ) : (
            <Button size="sm" onClick={onRunDraft} disabled={!strategy.file_exists}>
              <Play className="h-3.5 w-3.5" aria-hidden="true" />
              运行
            </Button>
          )}

          <Button
            variant="outline"
            size="sm"
            onClick={handleSave}
            disabled={saving || (!shortcut && !isDirty)}
          >
            <Save className="h-3.5 w-3.5" aria-hidden="true" />
            {shortcut ? "保存" : "保存为快捷方式"}
          </Button>

          {shortcut && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setNewScName(`${shortcut.name} (新方案)`);
                setSaveAsOpen(true);
              }}
            >
              <SaveAll className="h-3.5 w-3.5" aria-hidden="true" />
              另存为
            </Button>
          )}

          <Button variant="ghost" size="sm" onClick={onResetDraft} disabled={!isDirty}>
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            重置
          </Button>

          <Button
            variant={paramsOpen ? "secondary" : "outline"}
            size="sm"
            aria-expanded={paramsOpen}
            aria-controls={panelId}
            popoverTarget={supportsPopover ? panelId : undefined}
            onClick={supportsPopover ? undefined : () => setParamsOpen((v) => !v)}
          >
            <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
            {paramsOpen ? "收起参数" : "展开参数"}
          </Button>
        </div>
      </div>

      {saveErr && (
        <div role="alert" className="mx-4 mb-2 rounded-md bg-destructive/10 px-3 py-1.5 text-xs text-destructive">
          {saveErr}
        </div>
      )}

      {(() => {
        const paramFieldsContent = (
          <div className="space-y-3 px-4 py-3">
            <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4 lg:grid-cols-6">
            <div className="space-y-1">
              <Label htmlFor="param-mode">运行模式</Label>
              <SelectNative
                id="param-mode"
                value={draft.mode}
                onChange={(e) => {
                  const nextMode = e.target.value as RunMode;
                  onChangeDraft((prev) => {
                    const hasSelector = Boolean(meta?.has_selector);
                    if ((nextMode === "timing" || nextMode === "portfolio") && hasSelector) {
                      const trimmed = prev.objs.trim();
                      return {
                        ...prev,
                        mode: nextMode,
                        use_internal_selector: true,
                        objs: trimmed === "A_STOCKS_EXCLUDE_ST" ? "" : prev.objs,
                      };
                    }
                    return {
                      ...prev,
                      mode: nextMode,
                      use_internal_selector: false,
                      objs: prev.objs.trim() ? prev.objs : "A_STOCKS_EXCLUDE_ST",
                    };
                  });
                }}
              >
                {availableModes.map((m) => (
                  <option key={m} value={m}>
                    {RUN_MODE_LABELS[m]} ({m})
                  </option>
                ))}
              </SelectNative>
            </div>

            <div className="space-y-1 md:col-span-2 lg:col-span-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="param-objs">
                  {draft.use_internal_selector && draft.mode !== "select"
                    ? "候选池 (可选)"
                    : "标的池 (-s)"}
                </Label>
                <div className="flex items-center gap-1">
                  {OBJS_PRESETS.map((p) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => setField("objs", p)}
                      className="rounded border px-1.5 py-0.5 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
                    >
                      {p === "SH600000 SZ000001" ? "示例股" : p}
                    </button>
                  ))}
                </div>
              </div>
              <Input
                id="param-objs"
                value={draft.objs}
                onChange={(e) => setField("objs", e.target.value)}
                placeholder="如 A_STOCKS_EXCLUDE_ST 或 SH600000 SZ000001"
                autoCorrect="off"
                autoCapitalize="none"
                spellCheck={false}
              />
            </div>

            <div className="space-y-1 lg:col-span-2">
              <Label htmlFor="param-msd">MSD 服务 (-m)</Label>
              <Input
                id="param-msd"
                value={draft.msd_host}
                onChange={(e) => setField("msd_host", e.target.value)}
                placeholder={`默认: ${defaults.msd_host}`}
                autoCorrect="off"
                spellCheck={false}
              />
            </div>

            {draft.mode === "select" ? (
              <div className="space-y-1">
                <Label htmlFor="param-select-date">选股日期 (-d)</Label>
                <Input
                  id="param-select-date"
                  value={draft.select_date}
                  onChange={(e) => setField("select_date", e.target.value)}
                  placeholder={`默认: ${defaults.select_date || "最新交易日"}`}
                />
              </div>
            ) : (
              <>
                <div className="space-y-1">
                  <Label htmlFor="param-begin">开始日期 (-b)</Label>
                  <Input
                    id="param-begin"
                    value={draft.begin}
                    onChange={(e) => setField("begin", e.target.value)}
                    placeholder={`默认: ${defaults.begin}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-end">结束日期 (-e)</Label>
                  <Input
                    id="param-end"
                    value={draft.end}
                    onChange={(e) => setField("end", e.target.value)}
                    placeholder={`默认: ${defaults.end}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-benchmark">基准 (-B)</Label>
                  <Input
                    id="param-benchmark"
                    value={draft.benchmark}
                    onChange={(e) => setField("benchmark", e.target.value)}
                    placeholder={`默认: ${defaults.benchmark}`}
                    autoCorrect="off"
                    spellCheck={false}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-holdout">样本外 (--holdout)</Label>
                  <Input
                    id="param-holdout"
                    value={draft.holdout}
                    onChange={(e) => setField("holdout", e.target.value)}
                    placeholder={defaults.holdout ? `默认: ${defaults.holdout}` : "如 2025-01-01"}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-warmup">预热 Bar</Label>
                  <Input
                    id="param-warmup"
                    value={draft.warmup_bars}
                    onChange={(e) => setField("warmup_bars", e.target.value)}
                    placeholder={`默认: ${defaults.warmup_bars}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fin-fields">财务字段</Label>
                  <Input
                    id="param-fin-fields"
                    value={draft.fin_fields}
                    onChange={(e) => setField("fin_fields", e.target.value)}
                    placeholder={defaults.fin_fields ? `默认: ${defaults.fin_fields}` : "如 f001,f003"}
                  />
                </div>
              </>
            )}

            {(draft.mode === "timing" || draft.mode === "portfolio") && (
              <>
                <div className="space-y-1">
                  <Label htmlFor="param-cash">初始资金 (-c)</Label>
                  <Input
                    id="param-cash"
                    value={draft.cash}
                    onChange={(e) => setField("cash", e.target.value)}
                    placeholder={`默认: ${defaults.cash}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-exec">撮合方式</Label>
                  <SelectNative
                    id="param-exec"
                    value={draft.exec_mode}
                    onChange={(e) => setField("exec_mode", e.target.value)}
                  >
                    <option value="">默认 ({defaults.exec_mode})</option>
                    <option value="close">close</option>
                    <option value="next_open">next_open</option>
                  </SelectNative>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fee">费率配置 (-f)</Label>
                  <Input
                    id="param-fee"
                    value={draft.fee}
                    onChange={(e) => setField("fee", e.target.value)}
                    placeholder={defaults.fee || "默认费率"}
                  />
                </div>
                {meta?.has_selector && (
                  <div className="flex items-center gap-2 pt-5 md:col-span-2">
                    <Switch
                      id="param-internal-sel"
                      checked={draft.use_internal_selector}
                      onCheckedChange={(v) => setField("use_internal_selector", v)}
                    />
                    <Label htmlFor="param-internal-sel">
                      使用内置选股器 ({meta.selector_name || "Selector"})
                    </Label>
                  </div>
                )}
              </>
            )}

            {draft.mode === "factor" && (
              <>
                <div className="space-y-1">
                  <Label htmlFor="param-fp">持有期 (--periods)</Label>
                  <Input
                    id="param-fp"
                    value={draft.factor_periods}
                    onChange={(e) => setField("factor_periods", e.target.value)}
                    placeholder={`默认: ${defaults.factor_periods}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fq">分层数 (--quantiles)</Label>
                  <Input
                    id="param-fq"
                    value={draft.factor_quantiles}
                    onChange={(e) => setField("factor_quantiles", e.target.value)}
                    placeholder={`默认: ${defaults.factor_quantiles}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fentry">对齐 (--entry)</Label>
                  <SelectNative
                    id="param-fentry"
                    value={draft.factor_entry}
                    onChange={(e) => setField("factor_entry", e.target.value)}
                  >
                    <option value="">默认 ({defaults.factor_entry})</option>
                    <option value="next_open">next_open</option>
                    <option value="open_close">open_close</option>
                    <option value="close">close</option>
                  </SelectNative>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fwin">去极值</Label>
                  <SelectNative
                    id="param-fwin"
                    value={draft.factor_winsorize}
                    onChange={(e) => setField("factor_winsorize", e.target.value)}
                  >
                    <option value="">默认 ({defaults.factor_winsorize})</option>
                    <option value="none">none</option>
                    <option value="mad">mad</option>
                    <option value="sigma">sigma</option>
                    <option value="percentile">percentile</option>
                  </SelectNative>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fstd">标准化</Label>
                  <SelectNative
                    id="param-fstd"
                    value={draft.factor_standardize}
                    onChange={(e) => setField("factor_standardize", e.target.value)}
                  >
                    <option value="">默认 ({defaults.factor_standardize})</option>
                    <option value="none">none</option>
                    <option value="zscore">zscore</option>
                    <option value="rank">rank</option>
                  </SelectNative>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fneu">中性化</Label>
                  <Input
                    id="param-fneu"
                    value={draft.factor_neutralize}
                    onChange={(e) => setField("factor_neutralize", e.target.value)}
                    placeholder={`默认: ${defaults.factor_neutralize}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fcost">成本 bps</Label>
                  <Input
                    id="param-fcost"
                    value={draft.factor_cost_bps}
                    onChange={(e) => setField("factor_cost_bps", e.target.value)}
                    placeholder={`默认: ${defaults.factor_cost_bps}`}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="param-fdir">方向</Label>
                  <SelectNative
                    id="param-fdir"
                    value={draft.factor_direction}
                    onChange={(e) => setField("factor_direction", e.target.value)}
                  >
                    <option value="">默认 (代码定义)</option>
                    <option value="1">+1 (越大越好)</option>
                    <option value="-1">-1 (越小越好)</option>
                  </SelectNative>
                </div>
                <div className="flex items-center gap-2 pt-5">
                  <Switch
                    id="param-f-ind"
                    checked={draft.factor_by_industry}
                    onCheckedChange={(v) => setField("factor_by_industry", v)}
                  />
                  <Label htmlFor="param-f-ind">分行业 RankIC</Label>
                </div>
                <div className="flex items-center gap-2 pt-5">
                  <Switch
                    id="param-f-cau"
                    checked={draft.factor_causality_check}
                    onCheckedChange={(v) => setField("factor_causality_check", v)}
                  />
                  <Label htmlFor="param-f-cau">因果探针</Label>
                </div>
              </>
            )}
          </div>

          {(customParams.length > 0 || draft.mode !== "factor" || !meta?.is_expression_file) && (
            <div className="grid grid-cols-2 gap-2.5 border-t pt-2.5 md:grid-cols-4 lg:grid-cols-6">
              {customParams.map((p) => (
                <div key={p.name} className="space-y-1">
                  <Label htmlFor={`cust-${p.name}`}>
                    {p.name}
                    {p.type_label ? (
                      <span className="ml-1 font-normal text-muted-foreground">({p.type_label})</span>
                    ) : null}
                  </Label>
                  <Input
                    id={`cust-${p.name}`}
                    value={draft.strategy_args[p.name] ?? ""}
                    onChange={(e) => setStrategyArg(p.name, e.target.value)}
                    placeholder={p.default_str ? `默认: ${p.default_str}` : "留空使用默认"}
                    autoCorrect="off"
                    spellCheck={false}
                  />
                </div>
              ))}
              <div className="space-y-1 md:col-span-2">
                <Label htmlFor="param-extra">额外参数 (-a key=value)</Label>
                <Input
                  id="param-extra"
                  value={draft.extra_args}
                  onChange={(e) => setField("extra_args", e.target.value)}
                  placeholder="如 commission=0.0002 slippage=0.001"
                  autoCorrect="off"
                  spellCheck={false}
                />
              </div>
            </div>
          )}
            <div className="flex items-center justify-between border-t pt-2 text-[11px] text-muted-foreground">
              <span>参数修改立即生效于当前草稿；点击外部或按 Esc 收起</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
                popoverTarget={supportsPopover ? panelId : undefined}
                popoverTargetAction={supportsPopover ? "hide" : undefined}
                onClick={supportsPopover ? undefined : () => setParamsOpen(false)}
              >
                收起参数
              </Button>
            </div>
          </div>
        );

        if (supportsPopover) {
          return (
            <div
              id={panelId}
              ref={popoverRef}
              popover="auto"
              role="region"
              aria-label="策略运行参数"
              onToggle={(e) => {
                const ev = e.nativeEvent as unknown as { newState?: string };
                if (ev.newState) {
                  setParamsOpen(ev.newState === "open");
                }
              }}
              className="mbt-param-popover overflow-y-auto rounded-b-lg border border-t-0 bg-popover text-popover-foreground shadow-xl"
            >
              {paramFieldsContent}
            </div>
          );
        }

        return paramsOpen ? (
          <div
            id={panelId}
            role="region"
            aria-label="策略运行参数"
            className="absolute inset-x-0 top-full z-40 max-h-[68dvh] overflow-y-auto overscroll-contain rounded-b-lg border border-t-0 bg-popover text-popover-foreground shadow-xl"
          >
            {paramFieldsContent}
          </div>
        ) : null;
      })()}

      <Dialog open={saveAsOpen} onOpenChange={setSaveAsOpen}>
        <DialogContent className="w-[min(92vw,420px)]" aria-labelledby="save-as-title">
          <DialogHeader>
            <DialogTitle id="save-as-title">保存为新快捷方式</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleConfirmSaveAs} className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="new-shortcut-name">快捷方式名称</Label>
              <Input
                id="new-shortcut-name"
                value={newScName}
                onChange={(e) => setNewScName(e.target.value)}
                autoFocus
                required
              />
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setSaveAsOpen(false)}>
                取消
              </Button>
              <Button type="submit" disabled={saving}>
                {saving ? "保存中..." : "保存"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </section>
  );
}
