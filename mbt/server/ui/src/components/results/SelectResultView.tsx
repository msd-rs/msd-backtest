import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fmtNum } from "@/lib/format";
import type { RunManifest, RunSummaryRecord } from "@/types/api";
import { RunLogsView } from "./RunLogsView";
import { ServerDataTable } from "./ServerDataTable";

interface SelectResultViewProps {
  run: RunSummaryRecord;
  manifest: RunManifest;
  onRefreshRun: () => void;
}

export function SelectResultView({ run, manifest, onRefreshRun }: SelectResultViewProps) {
  const s = manifest.summary || {};
  return (
    <div className="flex h-full min-h-0 flex-1 flex-col gap-2 overflow-hidden">
      <div className="grid shrink-0 grid-cols-2 gap-2 sm:grid-cols-4">
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">选股器</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {String(s.strategy_name || manifest.strategy_name)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">选股日期</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {String(s.select_date || "-")}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">入选股票数</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {String(s.selected_count ?? 0)} 只
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">运行耗时</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtNum(manifest.elapsed, 2)}s
          </div>
        </div>
      </div>

      <Tabs defaultValue="table" className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <TabsList className="w-fit shrink-0">
          <TabsTrigger value="table">选股与因子明细</TabsTrigger>
          <TabsTrigger value="logs">运行日志</TabsTrigger>
        </TabsList>
        <TabsContent
          value="table"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <ServerDataTable
            runId={run.id}
            tableName="select_factors"
            title="入选标的与因子表"
            defaultSortBy="obj"
            defaultSortDir="asc"
            fillHeight
          />
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
