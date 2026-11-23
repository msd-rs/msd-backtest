import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

interface RunLogsViewProps {
  runId: string;
  status: string;
  onFinished?: () => void;
  fillHeight?: boolean;
}

export function RunLogsView({
  runId,
  status,
  onFinished,
  fillHeight = false,
}: RunLogsViewProps) {
  const [logs, setLogs] = useState("");
  const preRef = useRef<HTMLPreElement | null>(null);
  const onFinishedRef = useRef(onFinished);
  onFinishedRef.current = onFinished;

  useEffect(() => {
    setLogs("");
    if (status !== "running") {
      let cancelled = false;
      api
        .getRunLogs(runId)
        .then((txt) => {
          if (!cancelled) setLogs(txt);
        })
        .catch(() => { });
      return () => {
        cancelled = true;
      };
    }

    const es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/logs?stream=true`);
    es.onmessage = (ev) => {
      setLogs((prev) => (prev ? `${prev}\n${ev.data}` : ev.data));
    };
    es.addEventListener("done", () => {
      es.close();
      onFinishedRef.current?.();
    });
    es.onerror = () => {
      es.close();
    };
    return () => {
      es.close();
    };
  }, [runId, status]);

  useEffect(() => {
    if (preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <pre
      ref={preRef}
      className={
        fillHeight
          ? "min-h-0 h-full w-full flex-1 overflow-auto overscroll-contain rounded-lg border bg-slate-950 p-3 font-mono text-[11px] leading-relaxed text-slate-100"
          : "h-85 w-full overflow-auto rounded-lg border bg-slate-950 p-3 font-mono text-[11px] leading-relaxed text-slate-100"
      }
    >
      {logs || "(暂无日志输出)"}
    </pre>
  );
}
