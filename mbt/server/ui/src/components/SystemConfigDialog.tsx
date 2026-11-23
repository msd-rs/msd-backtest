import { useEffect, useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SelectNative } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { ServerConfigData, SystemDefaults } from "@/types/api";

interface SystemConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  config: ServerConfigData | null;
  onSave: (defaults: SystemDefaults) => Promise<void>;
}

export function SystemConfigDialog({
  open,
  onOpenChange,
  config,
  onSave,
}: SystemConfigDialogProps) {
  const [draft, setDraft] = useState<SystemDefaults | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (open && config) {
      setDraft({ ...config.defaults });
      setError("");
    }
  }, [open, config]);

  if (!config || !draft) {
    return null;
  }

  function update<K extends keyof SystemDefaults>(key: K, value: SystemDefaults[K]) {
    setDraft((prev) => (prev ? { ...prev, [key]: value } : prev));
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!draft) return;
    setSaving(true);
    setError("");
    try {
      await onSave(draft);
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[min(95vw,780px)]" aria-labelledby="sys-cfg-title">
        <DialogHeader>
          <DialogTitle id="sys-cfg-title">系统级默认参数设置</DialogTitle>
          <DialogDescription>
            快捷方式留空的项目将自动继承此处的系统默认配置。配置文件：{config.config_file}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <fieldset className="space-y-2.5 rounded-md border p-3">
            <legend className="px-1 text-xs font-semibold">公共回测与选股默认参数</legend>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
              <div className="space-y-1 sm:col-span-2">
                <Label htmlFor="sys-msd-host">MSD 数据服务地址 (-m)</Label>
                <Input
                  id="sys-msd-host"
                  name="msd_host"
                  value={draft.msd_host}
                  onChange={(e) => update("msd_host", e.target.value)}
                  autoCorrect="off"
                  spellCheck={false}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-benchmark">默认基准 (-B)</Label>
                <Input
                  id="sys-benchmark"
                  name="benchmark"
                  value={draft.benchmark}
                  onChange={(e) => update("benchmark", e.target.value)}
                  autoCorrect="off"
                  spellCheck={false}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-begin">开始日期 (-b)</Label>
                <Input
                  id="sys-begin"
                  name="begin"
                  value={draft.begin}
                  onChange={(e) => update("begin", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-end">结束日期 (-e)</Label>
                <Input
                  id="sys-end"
                  name="end"
                  value={draft.end}
                  onChange={(e) => update("end", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-select-date">选股日期 (-d)</Label>
                <Input
                  id="sys-select-date"
                  name="select_date"
                  value={draft.select_date}
                  placeholder="留空默认最新交易日"
                  onChange={(e) => update("select_date", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-cash">初始资金 (-c)</Label>
                <Input
                  id="sys-cash"
                  name="cash"
                  type="number"
                  min={1}
                  step="any"
                  value={String(draft.cash)}
                  onChange={(e) => update("cash", Number(e.target.value) || 10000000)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-exec-mode">撮合方式</Label>
                <SelectNative
                  id="sys-exec-mode"
                  name="exec_mode"
                  value={draft.exec_mode}
                  onChange={(e) => update("exec_mode", e.target.value as "close" | "next_open")}
                >
                  <option value="close">close (收盘成交)</option>
                  <option value="next_open">next_open (次日开盘)</option>
                </SelectNative>
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-warmup">预热 Bar (--warmup-bars)</Label>
                <Input
                  id="sys-warmup"
                  name="warmup_bars"
                  type="number"
                  min={0}
                  value={String(draft.warmup_bars)}
                  onChange={(e) => update("warmup_bars", Number(e.target.value) || 0)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-holdout">样本外划分 (--holdout)</Label>
                <Input
                  id="sys-holdout"
                  name="holdout"
                  value={draft.holdout}
                  placeholder="如 2025-01-01 或 1y"
                  onChange={(e) => update("holdout", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-fee">费率配置路径 (-f)</Label>
                <Input
                  id="sys-fee"
                  name="fee"
                  value={draft.fee}
                  placeholder="留空使用默认费率"
                  onChange={(e) => update("fee", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-fin-fields">财务字段 (--fin-fields)</Label>
                <Input
                  id="sys-fin-fields"
                  name="fin_fields"
                  value={draft.fin_fields}
                  placeholder="如 f001,f003"
                  onChange={(e) => update("fin_fields", e.target.value)}
                />
              </div>
            </div>
            <div className="flex items-center gap-2 pt-1">
              <Switch
                id="sys-fin-only-year"
                checked={draft.fin_only_year}
                onCheckedChange={(v) => update("fin_only_year", v)}
              />
              <Label htmlFor="sys-fin-only-year">仅加载年报财务数据 (--fin-only-year)</Label>
            </div>
          </fieldset>

          <fieldset className="space-y-2.5 rounded-md border p-3">
            <legend className="px-1 text-xs font-semibold">因子评估默认参数</legend>
            <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
              <div className="space-y-1">
                <Label htmlFor="sys-f-periods">持有期 (--periods)</Label>
                <Input
                  id="sys-f-periods"
                  value={draft.factor_periods}
                  onChange={(e) => update("factor_periods", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-quantiles">分层数 (--quantiles)</Label>
                <Input
                  id="sys-f-quantiles"
                  type="number"
                  min={2}
                  value={String(draft.factor_quantiles)}
                  onChange={(e) => update("factor_quantiles", Number(e.target.value) || 5)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-entry">前瞻收益对齐 (--entry)</Label>
                <SelectNative
                  id="sys-f-entry"
                  value={draft.factor_entry}
                  onChange={(e) =>
                    update("factor_entry", e.target.value as "next_open" | "open_close" | "close")
                  }
                >
                  <option value="next_open">next_open</option>
                  <option value="open_close">open_close</option>
                  <option value="close">close</option>
                </SelectNative>
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-winsorize">去极值 (--winsorize)</Label>
                <SelectNative
                  id="sys-f-winsorize"
                  value={draft.factor_winsorize}
                  onChange={(e) =>
                    update(
                      "factor_winsorize",
                      e.target.value as "none" | "mad" | "sigma" | "percentile",
                    )
                  }
                >
                  <option value="none">none</option>
                  <option value="mad">mad</option>
                  <option value="sigma">sigma</option>
                  <option value="percentile">percentile</option>
                </SelectNative>
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-standardize">标准化 (--standardize)</Label>
                <SelectNative
                  id="sys-f-standardize"
                  value={draft.factor_standardize}
                  onChange={(e) =>
                    update("factor_standardize", e.target.value as "none" | "zscore" | "rank")
                  }
                >
                  <option value="none">none</option>
                  <option value="zscore">zscore</option>
                  <option value="rank">rank</option>
                </SelectNative>
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-neutralize">中性化 (--neutralize)</Label>
                <Input
                  id="sys-f-neutralize"
                  value={draft.factor_neutralize}
                  placeholder="none 或 industry,mktcap"
                  onChange={(e) => update("factor_neutralize", e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-cost">单边成本 bps (--cost-bps)</Label>
                <Input
                  id="sys-f-cost"
                  type="number"
                  step="any"
                  value={String(draft.factor_cost_bps)}
                  onChange={(e) => update("factor_cost_bps", Number(e.target.value) || 0)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-minobs">最少截面样本 (--min-obs)</Label>
                <Input
                  id="sys-f-minobs"
                  type="number"
                  min={2}
                  value={String(draft.factor_min_obs)}
                  onChange={(e) => update("factor_min_obs", Number(e.target.value) || 10)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="sys-f-indcat">行业分类</Label>
                <Input
                  id="sys-f-indcat"
                  value={draft.factor_industry_category}
                  onChange={(e) => update("factor_industry_category", e.target.value)}
                />
              </div>
            </div>
            <div className="flex flex-wrap gap-4 pt-1">
              <div className="flex items-center gap-2">
                <Switch
                  id="sys-f-excl-limit"
                  checked={draft.factor_exclude_limit}
                  onCheckedChange={(v) => update("factor_exclude_limit", v)}
                />
                <Label htmlFor="sys-f-excl-limit">过滤涨跌停</Label>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  id="sys-f-by-ind"
                  checked={draft.factor_by_industry}
                  onCheckedChange={(v) => update("factor_by_industry", v)}
                />
                <Label htmlFor="sys-f-by-ind">按行业拆分 RankIC</Label>
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  id="sys-f-causality"
                  checked={draft.factor_causality_check}
                  onCheckedChange={(v) => update("factor_causality_check", v)}
                />
                <Label htmlFor="sys-f-causality">启用因果探针检查</Label>
              </div>
            </div>
          </fieldset>

          {error && (
            <div role="alert" className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              取消
            </Button>
            <Button type="submit" disabled={saving}>
              {saving ? "保存中..." : "保存系统配置"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
