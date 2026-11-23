import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fmtNum, fmtPct } from "@/lib/format";
import type { RunManifest, RunSummaryRecord } from "@/types/api";
import { EquityChart } from "./EquityChart";
import { RunLogsView } from "./RunLogsView";
import { ServerDataTable, type ColumnSpec } from "./ServerDataTable";

interface TimingResultViewProps {
  run: RunSummaryRecord;
  manifest: RunManifest;
  darkMode: boolean;
  onRefreshRun: () => void;
}

const TIMING_COLUMNS: ColumnSpec[] = [
  { key: "obj", label: "标的", align: "left" },
  { key: "profit", label: "策略收益", fmt: "pct", align: "right", colorSign: true },
  { key: "profit_original", label: "买入持有", fmt: "pct", align: "right", colorSign: true },
  { key: "benchmark_profit", label: "基准收益", fmt: "pct", align: "right" },
  { key: "alpha", label: "Alpha", fmt: "f4", align: "right" },
  { key: "beta", label: "Beta", fmt: "f4", align: "right" },
  { key: "sharpe", label: "Sharpe", fmt: "f4", align: "right" },
  { key: "sortino", label: "Sortino", fmt: "f4", align: "right" },
  { key: "max_drawdown", label: "最大回撤", fmt: "pct", align: "right" },
  { key: "calmar", label: "Calmar", fmt: "f4", align: "right" },
  { key: "win_rate", label: "胜率", fmt: "pct_plain", align: "right" },
  { key: "profit_factor", label: "盈亏比", fmt: "f4", align: "right" },
  { key: "fee_rate", label: "费率", fmt: "pct_plain", align: "right" },
  { key: "trade_count", label: "交易次数", fmt: "int", align: "right" },
];

export function TimingResultView({
  run,
  manifest,
  darkMode,
  onRefreshRun,
}: TimingResultViewProps) {
  const s = manifest.summary || {};
  const hasSummary = Boolean(s.has_summary_curve);
  const objs = (s.objs as string[] | undefined) || [];
  const [selectedObj, setSelectedObj] = useState<string>(
    hasSummary ? "__SUMMARY__" : objs[0] || "",
  );
  const benchObj = String(s.benchmark_obj || "基准");
  const holdout = s.holdout as Record<string, unknown> | null;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col gap-2 overflow-hidden">
      <div className="grid shrink-0 grid-cols-3 gap-2 lg:grid-cols-6">
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">标的数量</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {String(s.total_objs ?? 0)}（活跃 {String(s.active_objs ?? 0)}）
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">正收益占比</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {String(s.positive_count ?? 0)} / {String(s.total_objs ?? 0)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">跑赢买入持有</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {String(s.beat_buy_hold_count ?? 0)} / {String(s.total_objs ?? 0)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">平均策略收益</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">{fmtPct(s.mean_profit)}</div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">平均 Sharpe</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">{fmtNum(s.mean_sharpe, 3)}</div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">基准收益 ({benchObj})</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">{fmtPct(s.benchmark_profit)}</div>
        </div>
      </div>

      <Tabs defaultValue="main" className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
          <TabsList>
            <TabsTrigger value="main">收益曲线与标的表</TabsTrigger>
            {holdout && <TabsTrigger value="holdout">样本内外检验</TabsTrigger>}
            <TabsTrigger value="logs">运行日志</TabsTrigger>
          </TabsList>

          {hasSummary && selectedObj !== "__SUMMARY__" && (
            <button
              type="button"
              onClick={() => setSelectedObj("__SUMMARY__")}
              className="rounded-md border bg-background px-2.5 py-1 text-xs hover:bg-accent cursor-pointer"
            >
              返回等权汇总曲线
            </button>
          )}
        </div>

        <TabsContent
          value="main"
          className="mt-2 grid min-h-0 flex-1 grid-rows-[minmax(140px,54fr)_minmax(120px,46fr)] gap-2 overflow-hidden data-[state=inactive]:hidden"
        >
          <EquityChart
            runId={run.id}
            obj={selectedObj}
            title={
              selectedObj === "__SUMMARY__"
                ? `等权汇总收益曲线 (${objs.length} 只)`
                : `${selectedObj} 收益曲线`
            }
            origLabel={selectedObj === "__SUMMARY__" ? "等权买入持有" : "买入持有"}
            benchLabel={`基准 (${benchObj})`}
            darkMode={darkMode}
            fillHeight
          />

          <ServerDataTable
            runId={run.id}
            tableName="metrics"
            title="各标的回测指标（点击任意行切换上方曲线）"
            columns={TIMING_COLUMNS}
            defaultSortBy="profit"
            defaultSortDir="desc"
            rowKeyField="obj"
            selectedRowValue={selectedObj === "__SUMMARY__" ? undefined : selectedObj}
            onRowClick={(row) => {
              if (row.obj) setSelectedObj(String(row.obj));
            }}
            fillHeight
          />
        </TabsContent>

        {holdout && (
          <TabsContent
            value="holdout"
            className="mt-2 min-h-0 flex-1 overflow-y-auto overscroll-contain data-[state=inactive]:hidden"
          >
            <div className="rounded-lg border bg-card p-4 text-xs space-y-2">
              <div className="font-semibold">样本划分日期: {String(holdout.split_date)}</div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="rounded border p-3">
                  <div className="text-muted-foreground">样本内 (IS): {String(holdout.is_range)}</div>
                  <div className="mt-1">收益: {fmtPct(holdout.is_profit)}</div>
                  <div>Sharpe: {fmtNum(holdout.is_sharpe, 3)} | MaxDD: {fmtPct(holdout.is_max_dd, false)}</div>
                </div>
                <div className="rounded border p-3">
                  <div className="text-muted-foreground">样本外 (OOS): {String(holdout.oos_range)}</div>
                  <div className="mt-1">收益: {fmtPct(holdout.oos_profit)}</div>
                  <div>Sharpe: {fmtNum(holdout.oos_sharpe, 3)} | MaxDD: {fmtPct(holdout.oos_max_dd, false)}</div>
                </div>
                <div className="rounded border p-3">
                  <div className="text-muted-foreground">过拟合比 (OOS/IS Sharpe)</div>
                  <div className="mt-1 text-base font-semibold">{fmtNum(holdout.overfit_ratio, 3)}</div>
                </div>
              </div>
            </div>
          </TabsContent>
        )}

        <TabsContent
          value="logs"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <RunLogsView runId={run.id} status={run.status} onFinished={onRefreshRun} fillHeight />
        </TabsContent>
      </Tabs>
    </div>
  );
}
