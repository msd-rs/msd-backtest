import { useEffect, useState } from "react";
import { Loader2, Terminal, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import type { RunManifest, RunSummaryRecord } from "@/types/api";
import { FactorResultView } from "./results/FactorResultView";
import { PortfolioResultView } from "./results/PortfolioResultView";
import { RunLogsView } from "./results/RunLogsView";
import { SelectResultView } from "./results/SelectResultView";
import { TimingResultView } from "./results/TimingResultView";

interface ResultPaneProps {
  run: RunSummaryRecord | null;
  darkMode: boolean;
  onRefreshRun: () => void;
}

export function ResultPane({ run, darkMode, onRefreshRun }: ResultPaneProps) {
  const [manifest, setManifest] = useState<RunManifest | null>(null);
  const [loadingManifest, setLoadingManifest] = useState(false);
  const [manifestErr, setManifestErr] = useState("");

  useEffect(() => {
    if (!run || run.status !== "completed") {
      setManifest(null);
      setManifestErr("");
      return;
    }
    let cancelled = false;
    setLoadingManifest(true);
    setManifestErr("");
    api
      .getRunSummary(run.id)
      .then((res) => {
        if (!cancelled) setManifest(res);
      })
      .catch((err) => {
        if (!cancelled) setManifestErr(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoadingManifest(false);
      });
    return () => {
      cancelled = true;
    };
  }, [run?.id, run?.status]);

  if (!run) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center text-xs text-muted-foreground">
        在左侧选择策略或快捷方式，调整上方参数后点击「运行」即可在此查看按需加载的回测与因子分析结果。
      </div>
    );
  }

  if (run.status === "running") {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden p-3">
        <div className="flex shrink-0 items-center gap-2 rounded-lg border bg-card px-3 py-2 text-xs">
          <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden="true" />
          <span className="font-medium">策略运行中</span>
          <Badge variant={run.mode}>{run.mode}</Badge>
          <span className="text-muted-foreground">Run ID: {run.id} · PID: {run.pid ?? "-"}</span>
        </div>
        <RunLogsView runId={run.id} status={run.status} onFinished={onRefreshRun} fillHeight />
      </div>
    );
  }

  if (run.status === "failed" || run.status === "cancelled") {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden p-3">
        <div
          role="alert"
          className="shrink-0 max-h-40 overflow-auto overscroll-contain rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive space-y-1"
        >
          <div className="flex items-center gap-1.5 font-semibold">
            <XCircle className="h-4 w-4 shrink-0" aria-hidden="true" />
            {run.status === "cancelled" ? "任务已手动停止" : "任务执行失败"}
          </div>
          {run.error && <pre className="whitespace-pre-wrap font-mono text-[11px]">{run.error}</pre>}
        </div>
        <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-hidden">
          <div className="flex shrink-0 items-center gap-1.5 text-xs font-semibold">
            <Terminal className="h-3.5 w-3.5" aria-hidden="true" />
            运行日志
          </div>
          <RunLogsView runId={run.id} status={run.status} fillHeight />
        </div>
      </div>
    );
  }

  if (loadingManifest || !manifest) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-xs text-muted-foreground">
        {manifestErr ? (
          <span className="text-destructive">加载结果摘要失败: {manifestErr}</span>
        ) : (
          <span className="inline-flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            正在读取结果清单...
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-3">
      {manifest.mode === "select" && (
        <SelectResultView run={run} manifest={manifest} onRefreshRun={onRefreshRun} />
      )}
      {manifest.mode === "timing" && (
        <TimingResultView
          run={run}
          manifest={manifest}
          darkMode={darkMode}
          onRefreshRun={onRefreshRun}
        />
      )}
      {manifest.mode === "portfolio" && (
        <PortfolioResultView
          run={run}
          manifest={manifest}
          darkMode={darkMode}
          onRefreshRun={onRefreshRun}
        />
      )}
      {manifest.mode === "factor" && (
        <FactorResultView
          run={run}
          manifest={manifest}
          darkMode={darkMode}
          onRefreshRun={onRefreshRun}
        />
      )}
    </div>
  );
}
