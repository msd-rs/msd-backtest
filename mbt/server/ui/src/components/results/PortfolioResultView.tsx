import { Download } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { fmtMoney, fmtNum, fmtPct } from "@/lib/format";
import type { RunManifest, RunSummaryRecord } from "@/types/api";
import { EquityChart } from "./EquityChart";
import { RunLogsView } from "./RunLogsView";
import { ServerDataTable, type ColumnSpec } from "./ServerDataTable";

interface PortfolioResultViewProps {
  run: RunSummaryRecord;
  manifest: RunManifest;
  darkMode: boolean;
  onRefreshRun: () => void;
}

const REBALANCE_COLS: ColumnSpec[] = [
  { key: "date", label: "日期", align: "left" },
  { key: "obj", label: "标的", align: "left" },
  { key: "action", label: "方向", align: "center" },
  { key: "shares", label: "成交股数", fmt: "int", align: "right" },
  { key: "deal_price", label: "成交价", fmt: "price", align: "right" },
  { key: "amount", label: "成交金额", fmt: "money", align: "right" },
  { key: "fee", label: "手续费", fmt: "money", align: "right" },
  { key: "pos_shares", label: "持仓股数", fmt: "int", align: "right" },
  { key: "cash", label: "可用现金", fmt: "money", align: "right" },
];

const TRADE_COLS: ColumnSpec[] = [
  { key: "obj", label: "标的", align: "left" },
  { key: "entry_date", label: "买入日期", align: "left" },
  { key: "exit_date", label: "卖出日期", align: "left" },
  { key: "profit_rate", label: "持仓收益率", fmt: "pct", align: "right", colorSign: true },
  { key: "profit_amount", label: "持仓盈亏额", fmt: "money_signed", align: "right", colorSign: true },
];

const ATTRIB_COLS: ColumnSpec[] = [
  { key: "obj", label: "标的", align: "left" },
  { key: "trade_count", label: "交易轮次", fmt: "int", align: "right" },
  { key: "win_rate", label: "胜率", fmt: "pct_plain", align: "right" },
  { key: "total_profit", label: "累计盈亏", fmt: "money_signed", align: "right", colorSign: true },
  { key: "capital_contribution", label: "资金贡献", fmt: "pct", align: "right", colorSign: true },
  { key: "avg_return", label: "单次均收益", fmt: "pct", align: "right", colorSign: true },
];

export function PortfolioResultView({
  run,
  manifest,
  darkMode,
  onRefreshRun,
}: PortfolioResultViewProps) {
  const s = manifest.summary || {};
  const m = (s.metrics as Record<string, unknown>) || {};
  const benchObj = String(s.benchmark_obj || "基准");
  const holdout = s.holdout as Record<string, unknown> | null;
  const diag = s.diagnostics as Record<string, unknown> | null;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col gap-2 overflow-hidden">
      <div className="grid shrink-0 grid-cols-3 gap-2 lg:grid-cols-6">
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">累计收益</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">{fmtPct(m.profit)}</div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">基准 ({benchObj})</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">{fmtPct(m.benchmark_profit)}</div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">Sharpe / Calmar</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtNum(m.sharpe, 2)} / {fmtNum(m.calmar, 2)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">最大回撤</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">{fmtPct(m.max_drawdown, false)}</div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">期末资产</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">{fmtMoney(m.final_assets)}</div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">胜率 / 调仓笔数</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtPct(m.win_rate, false, 1)} / {String(s.rebalance_records_count ?? 0)}
          </div>
        </div>
      </div>

      <Tabs defaultValue="curve" className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex shrink-0 flex-wrap items-center justify-between gap-2">
          <TabsList>
            <TabsTrigger value="curve">收益曲线</TabsTrigger>
            <TabsTrigger value="rebalances">调仓明细 ({String(s.rebalance_records_count ?? 0)})</TabsTrigger>
            <TabsTrigger value="trades">持仓周期 ({String(s.trades_count ?? 0)})</TabsTrigger>
            <TabsTrigger value="attribution">个股归因</TabsTrigger>
            <TabsTrigger value="metrics">指标与诊断</TabsTrigger>
            <TabsTrigger value="logs">运行日志</TabsTrigger>
          </TabsList>

          <div className="flex items-center gap-1.5">
            <a
              href={api.exportUrl(run.id, "xlsx")}
              className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs hover:bg-accent"
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              导出全表 XLSX
            </a>
            <a
              href={api.exportUrl(run.id, "parquet")}
              className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs hover:bg-accent"
            >
              Parquet ZIP
            </a>
          </div>
        </div>

        <TabsContent
          value="curve"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <EquityChart
            runId={run.id}
            title={`${manifest.strategy_name} 组合净值收益率`}
            origLabel="池内等权买入持有"
            benchLabel={`基准 (${benchObj})`}
            darkMode={darkMode}
            fillHeight
          />
        </TabsContent>

        <TabsContent
          value="rebalances"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <ServerDataTable
            runId={run.id}
            tableName="rebalances"
            title="逐笔调仓委托成交记录"
            columns={REBALANCE_COLS}
            defaultSortBy="bar"
            defaultSortDir="asc"
            fillHeight
          />
        </TabsContent>

        <TabsContent
          value="trades"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <ServerDataTable
            runId={run.id}
            tableName="trades"
            title="持仓盈亏周期"
            columns={TRADE_COLS}
            defaultSortBy="entry_bar"
            defaultSortDir="asc"
            fillHeight
          />
        </TabsContent>

        <TabsContent
          value="attribution"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <ServerDataTable
            runId={run.id}
            tableName="attribution"
            title="个股收益归因"
            columns={ATTRIB_COLS}
            defaultSortBy="total_profit"
            defaultSortDir="desc"
            fillHeight
          />
        </TabsContent>

        <TabsContent
          value="metrics"
          className="mt-2 flex min-h-0 flex-1 flex-col gap-2 overflow-hidden data-[state=inactive]:hidden"
        >
          {holdout && (
            <div className="shrink-0 space-y-1.5 rounded-lg border bg-card px-3 py-2 text-xs">
              <div className="font-semibold">样本外 Holdout 检验 (Split: {String(holdout.split_date)})</div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                <div className="rounded border p-2">
                  <div className="text-muted-foreground">样本内: {String(holdout.is_range)}</div>
                  <div>收益: {fmtPct(holdout.is_profit)} | Sharpe: {fmtNum(holdout.is_sharpe, 3)} | MaxDD: {fmtPct(holdout.is_max_dd, false)}</div>
                </div>
                <div className="rounded border p-2">
                  <div className="text-muted-foreground">样本外: {String(holdout.oos_range)}</div>
                  <div>收益: {fmtPct(holdout.oos_profit)} | Sharpe: {fmtNum(holdout.oos_sharpe, 3)} | MaxDD: {fmtPct(holdout.oos_max_dd, false)}</div>
                </div>
                <div className="rounded border p-2">
                  <div className="text-muted-foreground">OOS/IS Sharpe 比</div>
                  <div className="font-semibold">{fmtNum(holdout.overfit_ratio, 3)}</div>
                </div>
              </div>
            </div>
          )}

          <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-2 gap-2 overflow-hidden md:grid-cols-2 md:grid-rows-1">
            <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card p-3">
              <h4 className="mb-2 shrink-0 text-xs font-semibold">完整绩效指标</h4>
              <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]">
                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                  {Object.entries(m).map(([k, v]) => (
                    <div key={k} className="flex justify-between border-b border-border/50 py-1">
                      <span className="text-muted-foreground">{k}</span>
                      <span className="font-mono">{typeof v === "number" ? fmtNum(v, 4) : String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card p-3">
              <h4 className="mb-2 shrink-0 text-xs font-semibold">策略诊断 (finalize)</h4>
              {diag ? (
                <pre className="min-h-0 flex-1 overflow-auto overscroll-contain rounded bg-muted/40 p-2 font-mono text-[11px]">
                  {JSON.stringify(diag, null, 2)}
                </pre>
              ) : (
                <div className="flex flex-1 items-center justify-center text-xs text-muted-foreground">
                  无诊断输出
                </div>
              )}
            </div>
          </div>
        </TabsContent>

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
