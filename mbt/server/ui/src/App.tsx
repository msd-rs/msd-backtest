import { useCallback, useEffect, useMemo, useState } from "react";
import { AddStrategyDialog } from "@/components/AddStrategyDialog";
import { HeaderBar } from "@/components/HeaderBar";
import { ParamPanel } from "@/components/ParamPanel";
import { ResultPane } from "@/components/ResultPane";
import { StrategyTree, type SelectionTarget } from "@/components/StrategyTree";
import { SystemConfigDialog } from "@/components/SystemConfigDialog";
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
import { api } from "@/lib/api";
import type {
  DraftParams,
  RunMode,
  RunSummaryRecord,
  ServerConfigData,
  ShortcutRecord,
  StrategyRecord,
  SystemDefaults,
} from "@/types/api";

function buildDraftFromSelection(
  strat: StrategyRecord,
  sc: ShortcutRecord | null,
): DraftParams {
  const p = sc?.params ?? {};
  const mode: RunMode = sc?.mode ?? strat.default_mode;
  const rawArgs = (p.strategy_args as Record<string, unknown> | undefined) ?? {};
  const strategyArgs: Record<string, string> = {};
  for (const [k, v] of Object.entries(rawArgs)) {
    strategyArgs[k] = v === null || v === undefined ? "" : String(v);
  }
  const s = (k: string) => {
    const val = p[k];
    return val === null || val === undefined ? "" : String(val);
  };
  const b = (k: string, fallback: boolean) => {
    const val = p[k];
    return typeof val === "boolean" ? val : fallback;
  };
  return {
    mode,
    objs: s("objs"),
    use_internal_selector: b("use_internal_selector", false),
    begin: s("begin"),
    end: s("end"),
    select_date: s("select_date"),
    benchmark: s("benchmark"),
    cash: s("cash"),
    fee: s("fee"),
    exec_mode: s("exec_mode"),
    warmup_bars: s("warmup_bars"),
    holdout: s("holdout"),
    msd_host: s("msd_host"),
    fin_fields: s("fin_fields"),
    fin_only_year: b("fin_only_year", false),
    fin_lookback_days: s("fin_lookback_days"),
    strategy_args: strategyArgs,
    extra_args: s("extra_args"),
    factor_periods: s("factor_periods"),
    factor_primary_period: s("factor_primary_period"),
    factor_quantiles: s("factor_quantiles"),
    factor_entry: s("factor_entry"),
    factor_exclude_limit: b("factor_exclude_limit", true),
    factor_min_listed_days: s("factor_min_listed_days"),
    factor_winsorize: s("factor_winsorize"),
    factor_standardize: s("factor_standardize"),
    factor_neutralize: s("factor_neutralize"),
    factor_cost_bps: s("factor_cost_bps"),
    factor_min_obs: s("factor_min_obs"),
    factor_by_industry: b("factor_by_industry", false),
    factor_industry_category: s("factor_industry_category"),
    factor_causality_check: b("factor_causality_check", true),
    factor_direction: s("factor_direction"),
  };
}

function draftToShortcutParams(draft: DraftParams): Record<string, unknown> {
  const cleanArgs: Record<string, string> = {};
  for (const [k, v] of Object.entries(draft.strategy_args)) {
    if (v.trim() !== "") cleanArgs[k] = v.trim();
  }
  return {
    objs: draft.objs.trim(),
    use_internal_selector: draft.use_internal_selector,
    begin: draft.begin.trim(),
    end: draft.end.trim(),
    select_date: draft.select_date.trim(),
    benchmark: draft.benchmark.trim(),
    cash: draft.cash.trim(),
    fee: draft.fee.trim(),
    exec_mode: draft.exec_mode.trim(),
    warmup_bars: draft.warmup_bars.trim(),
    holdout: draft.holdout.trim(),
    msd_host: draft.msd_host.trim(),
    fin_fields: draft.fin_fields.trim(),
    fin_only_year: draft.fin_only_year,
    fin_lookback_days: draft.fin_lookback_days.trim(),
    strategy_args: cleanArgs,
    extra_args: draft.extra_args.trim(),
    factor_periods: draft.factor_periods.trim(),
    factor_primary_period: draft.factor_primary_period.trim(),
    factor_quantiles: draft.factor_quantiles.trim(),
    factor_entry: draft.factor_entry.trim(),
    factor_exclude_limit: draft.factor_exclude_limit,
    factor_min_listed_days: draft.factor_min_listed_days.trim(),
    factor_winsorize: draft.factor_winsorize.trim(),
    factor_standardize: draft.factor_standardize.trim(),
    factor_neutralize: draft.factor_neutralize.trim(),
    factor_cost_bps: draft.factor_cost_bps.trim(),
    factor_min_obs: draft.factor_min_obs.trim(),
    factor_by_industry: draft.factor_by_industry,
    factor_industry_category: draft.factor_industry_category.trim(),
    factor_causality_check: draft.factor_causality_check,
    factor_direction: draft.factor_direction ? Number(draft.factor_direction) : null,
  };
}

export function App() {
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem("mbt-theme");
    if (saved === "dark") return true;
    if (saved === "light") return false;
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  const [healthText, setHealthText] = useState("");
  const [healthOk, setHealthOk] = useState(false);
  const [config, setConfig] = useState<ServerConfigData | null>(null);
  const [strategies, setStrategies] = useState<StrategyRecord[]>([]);
  const [selection, setSelection] = useState<SelectionTarget>(null);

  const [draft, setDraft] = useState<DraftParams | null>(null);
  const [baseDraftJson, setBaseDraftJson] = useState("");
  const [currentRun, setCurrentRun] = useState<RunSummaryRecord | null>(null);

  const [addOpen, setAddOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [refreshingList, setRefreshingList] = useState(false);
  const [renameTarget, setRenameTarget] = useState<StrategyRecord | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [toastError, setToastError] = useState("");

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(darkMode ? "dark" : "light");
    localStorage.setItem("mbt-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  const reloadAll = useCallback(async () => {
    const [h, cfg, strats] = await Promise.all([
      api.getHealth().catch(() => ""),
      api.getConfig(),
      api.listStrategies(),
    ]);
    setHealthText(h);
    setHealthOk(h.startsWith("OK, msd-backtest"));
    setConfig(cfg);
    setStrategies(strats);
    return strats;
  }, []);

  useEffect(() => {
    reloadAll()
      .then((strats) => {
        if (strats.length > 0) {
          const first = strats[0];
          if (first.shortcuts.length > 0) {
            setSelection({ kind: "shortcut", strategyId: first.id, shortcutId: first.shortcuts[0].id });
          } else {
            setSelection({ kind: "strategy", strategyId: first.id });
          }
        }
      })
      .catch((err) => setToastError(err instanceof Error ? err.message : String(err)));
  }, [reloadAll]);

  const selectedStrategy = useMemo(() => {
    if (!selection) return null;
    return strategies.find((s) => s.id === selection.strategyId) || null;
  }, [strategies, selection]);

  const selectedShortcut = useMemo(() => {
    if (!selectedStrategy || selection?.kind !== "shortcut") return null;
    return selectedStrategy.shortcuts.find((sc) => sc.id === selection.shortcutId) || null;
  }, [selectedStrategy, selection]);

  useEffect(() => {
    if (!selectedStrategy) {
      setDraft(null);
      setBaseDraftJson("");
      setCurrentRun(null);
      return;
    }
    const nextDraft = buildDraftFromSelection(selectedStrategy, selectedShortcut);
    setDraft(nextDraft);
    setBaseDraftJson(JSON.stringify(nextDraft));
    const active = selectedShortcut ? selectedShortcut.active_run : selectedStrategy.active_run;
    const latest = selectedShortcut ? selectedShortcut.latest_run : selectedStrategy.latest_run;
    setCurrentRun(active || latest || null);
  }, [selectedStrategy?.id, selectedShortcut?.id, selectedShortcut?.updated_at]);

  // Poll active run until completion
  useEffect(() => {
    if (!currentRun || currentRun.status !== "running") return;
    const timer = setInterval(async () => {
      try {
        const updated = await api.getRun(currentRun.id);
        setCurrentRun(updated);
        if (updated.status !== "running") {
          const strats = await api.listStrategies();
          setStrategies(strats);
        }
      } catch {
        // ignore transient poll error
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [currentRun?.id, currentRun?.status]);

  const isDirty = Boolean(draft && JSON.stringify(draft) !== baseDraftJson);

  async function handleRegisterStrategy(payload: { path: string; name?: string; notes?: string }) {
    const rec = await api.registerStrategy(payload);
    const strats = await api.listStrategies();
    setStrategies(strats);
    setSelection({ kind: "strategy", strategyId: rec.id });
  }

  async function handleSaveSystemDefaults(defaults: SystemDefaults) {
    const updated = await api.updateConfig({
      server: config?.server,
      defaults,
    });
    setConfig(updated);
  }

  async function handleRunTarget(explicit?: { strategyId?: string; shortcutId?: string }) {
    setToastError("");
    try {
      let runRec: RunSummaryRecord;
      if (explicit?.shortcutId) {
        runRec = await api.startRun({ shortcut_id: explicit.shortcutId });
      } else if (explicit?.strategyId) {
        runRec = await api.startRun({ strategy_id: explicit.strategyId });
      } else if (selectedStrategy && draft) {
        const overrides = draftToShortcutParams(draft);
        runRec = await api.startRun({
          strategy_id: selectedShortcut ? undefined : selectedStrategy.id,
          shortcut_id: selectedShortcut ? selectedShortcut.id : undefined,
          mode: draft.mode,
          overrides,
        });
      } else {
        return;
      }
      setCurrentRun(runRec);
      const strats = await api.listStrategies();
      setStrategies(strats);
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleStopRun() {
    if (!currentRun || currentRun.status !== "running") return;
    try {
      const stopped = await api.cancelRun(currentRun.id);
      setCurrentRun(stopped);
      const strats = await api.listStrategies();
      setStrategies(strats);
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleSaveShortcut() {
    if (!selectedShortcut || !draft) return;
    const updated = await api.updateShortcut(selectedShortcut.id, {
      mode: draft.mode,
      params: draftToShortcutParams(draft),
    });
    const strats = await api.listStrategies();
    setStrategies(strats);
    setBaseDraftJson(JSON.stringify(draft));
    if (updated.latest_run && currentRun?.id === updated.latest_run.id) {
      setCurrentRun(updated.latest_run);
    }
  }

  async function handleSaveAsShortcut(name: string) {
    if (!selectedStrategy || !draft) return;
    const created = await api.createShortcut(selectedStrategy.id, {
      name,
      mode: draft.mode,
      params: draftToShortcutParams(draft),
    });
    const strats = await api.listStrategies();
    setStrategies(strats);
    setSelection({
      kind: "shortcut",
      strategyId: selectedStrategy.id,
      shortcutId: created.id,
    });
  }

  async function handleAddShortcutForStrategy(strat: StrategyRecord) {
    try {
      const sc = await api.createShortcut(strat.id, {
        name: `方案 ${strat.shortcuts.length + 1}`,
        mode: strat.default_mode,
        params: {},
      });
      const strats = await api.listStrategies();
      setStrategies(strats);
      setSelection({ kind: "shortcut", strategyId: strat.id, shortcutId: sc.id });
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleRefreshList() {
    setRefreshingList(true);
    try {
      await reloadAll();
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    } finally {
      setRefreshingList(false);
    }
  }

  async function handleRefreshStrategy(strategyId: string) {
    try {
      await api.refreshStrategy(strategyId);
      const strats = await api.listStrategies();
      setStrategies(strats);
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDeleteStrategy(strat: StrategyRecord) {
    if (!window.confirm(`确定移除策略「${strat.name}」及其全部快捷方式与缓存结果吗？（不删除磁盘文件）`)) {
      return;
    }
    try {
      await api.deleteStrategy(strat.id);
      const strats = await api.listStrategies();
      setStrategies(strats);
      if (selection?.strategyId === strat.id) {
        setSelection(strats[0] ? { kind: "strategy", strategyId: strats[0].id } : null);
      }
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDuplicateShortcut(sc: ShortcutRecord) {
    try {
      const copy = await api.duplicateShortcut(sc.id);
      const strats = await api.listStrategies();
      setStrategies(strats);
      setSelection({ kind: "shortcut", strategyId: sc.strategy_id, shortcutId: copy.id });
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleDeleteShortcut(sc: ShortcutRecord) {
    if (!window.confirm(`确定删除快捷方式「${sc.name}」吗？`)) return;
    try {
      await api.deleteShortcut(sc.id);
      const strats = await api.listStrategies();
      setStrategies(strats);
      if (selection?.kind === "shortcut" && selection.shortcutId === sc.id) {
        setSelection({ kind: "strategy", strategyId: sc.strategy_id });
      }
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleConfirmRename(e: React.FormEvent) {
    e.preventDefault();
    if (!renameTarget || !renameValue.trim()) return;
    try {
      await api.updateStrategy(renameTarget.id, { name: renameValue.trim() });
      const strats = await api.listStrategies();
      setStrategies(strats);
      setRenameTarget(null);
    } catch (err) {
      setToastError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="flex h-dvh w-full flex-col overflow-hidden bg-background text-foreground">
      <HeaderBar
        healthText={healthText}
        healthOk={healthOk}
        darkMode={darkMode}
        onToggleDark={() => setDarkMode((d) => !d)}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      {toastError && (
        <div
          role="alert"
          className="flex items-center justify-between border-b bg-destructive/15 px-4 py-1.5 text-xs text-destructive"
        >
          <span>{toastError}</span>
          <button
            type="button"
            onClick={() => setToastError("")}
            className="ml-4 font-semibold hover:underline cursor-pointer"
          >
            关闭
          </button>
        </div>
      )}

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <StrategyTree
          strategies={strategies}
          selection={selection}
          refreshingList={refreshingList}
          onSelect={setSelection}
          onRefreshList={handleRefreshList}
          onOpenAddDialog={() => setAddOpen(true)}
          onRefreshStrategy={handleRefreshStrategy}
          onRenameStrategy={(s) => {
            setRenameTarget(s);
            setRenameValue(s.name);
          }}
          onDeleteStrategy={handleDeleteStrategy}
          onAddShortcut={handleAddShortcutForStrategy}
          onDuplicateShortcut={handleDuplicateShortcut}
          onDeleteShortcut={handleDeleteShortcut}
          onQuickRun={handleRunTarget}
        />

        <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
          {selectedStrategy && draft && config ? (
            <ParamPanel
              strategy={selectedStrategy}
              shortcut={selectedShortcut}
              defaults={config.defaults}
              draft={draft}
              isDirty={isDirty}
              isRunning={currentRun?.status === "running"}
              onChangeDraft={(fn) => setDraft((prev) => (prev ? fn(prev) : prev))}
              onResetDraft={() => {
                if (selectedStrategy) {
                  const reset = buildDraftFromSelection(selectedStrategy, selectedShortcut);
                  setDraft(reset);
                }
              }}
              onRunDraft={() => handleRunTarget()}
              onStopRun={handleStopRun}
              onSaveShortcut={handleSaveShortcut}
              onSaveAsShortcut={handleSaveAsShortcut}
            />
          ) : (
            <div className="border-b bg-card px-4 py-3 text-xs text-muted-foreground">
              请从左侧选择或添加一个策略文件
            </div>
          )}

          <ResultPane
            run={currentRun}
            darkMode={darkMode}
            onRefreshRun={async () => {
              if (!currentRun) return;
              const updated = await api.getRun(currentRun.id).catch(() => null);
              if (updated) setCurrentRun(updated);
              const strats = await api.listStrategies().catch(() => null);
              if (strats) setStrategies(strats);
            }}
          />
        </main>
      </div>

      <AddStrategyDialog
        open={addOpen}
        onOpenChange={setAddOpen}
        onSubmit={handleRegisterStrategy}
      />

      <SystemConfigDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        config={config}
        onSave={handleSaveSystemDefaults}
      />

      <Dialog open={Boolean(renameTarget)} onOpenChange={(o) => !o && setRenameTarget(null)}>
        <DialogContent className="w-[min(92vw,420px)]" aria-labelledby="rename-strat-title">
          <DialogHeader>
            <DialogTitle id="rename-strat-title">重命名策略</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleConfirmRename} className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor="rename-strat-input">显示名称</Label>
              <Input
                id="rename-strat-input"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                autoFocus
                required
              />
            </div>
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setRenameTarget(null)}>
                取消
              </Button>
              <Button type="submit">保存</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
export default App;
