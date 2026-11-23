import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Copy,
  FileCode2,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Search,
  Trash2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import {
  RUN_MODE_LABELS,
  type RunSummaryRecord,
  type ShortcutRecord,
  type StrategyRecord,
} from "@/types/api";

export type SelectionTarget =
  | { kind: "strategy"; strategyId: string }
  | { kind: "shortcut"; strategyId: string; shortcutId: string }
  | null;

interface StrategyTreeProps {
  strategies: StrategyRecord[];
  selection: SelectionTarget;
  refreshingList?: boolean;
  onSelect: (target: SelectionTarget) => void;
  onRefreshList: () => void;
  onOpenAddDialog: () => void;
  onRefreshStrategy: (strategyId: string) => void;
  onRenameStrategy: (strat: StrategyRecord) => void;
  onDeleteStrategy: (strat: StrategyRecord) => void;
  onAddShortcut: (strat: StrategyRecord) => void;
  onDuplicateShortcut: (sc: ShortcutRecord) => void;
  onDeleteShortcut: (sc: ShortcutRecord) => void;
  onQuickRun: (target: { strategyId?: string; shortcutId?: string }) => void;
}

function StatusDot({
  active,
  latest,
}: {
  active: RunSummaryRecord | null;
  latest: RunSummaryRecord | null;
}) {
  if (active && active.status === "running") {
    return (
      <span
        title="运行中"
        className="inline-block h-2 w-2 shrink-0 animate-pulse rounded-full bg-blue-500"
      />
    );
  }
  if (!latest) return null;
  if (latest.status === "completed") {
    return <span title="最近运行成功" className="inline-block h-2 w-2 shrink-0 rounded-full bg-emerald-500" />;
  }
  if (latest.status === "failed") {
    return <span title="最近运行失败" className="inline-block h-2 w-2 shrink-0 rounded-full bg-red-500" />;
  }
  if (latest.status === "cancelled") {
    return <span title="已取消" className="inline-block h-2 w-2 shrink-0 rounded-full bg-amber-500" />;
  }
  return null;
}

export function StrategyTree({
  strategies,
  selection,
  refreshingList = false,
  onSelect,
  onRefreshList,
  onOpenAddDialog,
  onRefreshStrategy,
  onRenameStrategy,
  onDeleteStrategy,
  onAddShortcut,
  onDuplicateShortcut,
  onDeleteShortcut,
  onQuickRun,
}: StrategyTreeProps) {
  const [filterText, setFilterText] = useState("");
  const [collapsedIds, setCollapsedIds] = useState<Record<string, boolean>>({});

  const filtered = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    if (!q) return strategies;
    return strategies
      .map((s) => {
        const stratMatch =
          s.name.toLowerCase().includes(q) || s.path.toLowerCase().includes(q);
        const matchedSc = s.shortcuts.filter((sc) => sc.name.toLowerCase().includes(q));
        if (stratMatch) return s;
        if (matchedSc.length > 0) return { ...s, shortcuts: matchedSc };
        return null;
      })
      .filter((x): x is StrategyRecord => x !== null);
  }, [strategies, filterText]);

  function toggleCollapse(id: string) {
    setCollapsedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-r bg-card/50">
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2.5">
        <span className="text-xs font-semibold tracking-wide text-foreground">
          策略与快捷方式 ({strategies.length})
        </span>
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="icon"
            onClick={onRefreshList}
            disabled={refreshingList}
            title="刷新策略列表"
            aria-label="刷新策略列表"
          >
            <RefreshCw
              className={cn("h-3.5 w-3.5", refreshingList && "animate-spin")}
              aria-hidden="true"
            />
          </Button>
          <Button size="sm" onClick={onOpenAddDialog}>
            <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            添加策略
          </Button>
        </div>
      </div>

      <div className="border-b px-2.5 py-2">
        <div className="relative">
          <Search
            className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-muted-foreground"
            aria-hidden="true"
          />
          <Input
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            placeholder="过滤策略名称或路径..."
            className="h-7 pl-8 text-xs"
            aria-label="过滤策略树"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {filtered.length === 0 ? (
          <div className="px-3 py-8 text-center text-xs text-muted-foreground">
            {strategies.length === 0
              ? "暂无策略，点击右上角「添加策略」或由 Agent 自动提交。"
              : "没有匹配的策略或快捷方式。"}
          </div>
        ) : (
          filtered.map((strat) => {
            const isExpanded = !collapsedIds[strat.id];
            const isStratSelected =
              selection?.kind === "strategy" && selection.strategyId === strat.id;
            return (
              <div key={strat.id} className="space-y-0.5">
                <div
                  className={cn(
                    "group flex items-center gap-1 rounded-md px-1.5 py-1.5 text-xs transition-colors",
                    isStratSelected
                      ? "bg-primary/12 text-foreground font-medium"
                      : "hover:bg-accent/70 text-foreground/90",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => toggleCollapse(strat.id)}
                    className="rounded p-0.5 text-muted-foreground hover:bg-accent cursor-pointer"
                    aria-label={isExpanded ? "收起快捷方式" : "展开快捷方式"}
                  >
                    {isExpanded ? (
                      <ChevronDown className="h-3.5 w-3.5" />
                    ) : (
                      <ChevronRight className="h-3.5 w-3.5" />
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={() => onSelect({ kind: "strategy", strategyId: strat.id })}
                    onDoubleClick={() => onQuickRun({ strategyId: strat.id })}
                    className="flex min-w-0 flex-1 items-center gap-1.5 text-left cursor-pointer"
                    title={`${strat.path}\n双击使用默认参数运行`}
                  >
                    <FileCode2 className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
                    <span className="truncate">{strat.name}</span>
                    {!strat.file_exists && (
                      <span title="磁盘文件不存在">
                        <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-destructive" aria-hidden="true" />
                      </span>
                    )}
                    <StatusDot active={strat.active_run} latest={strat.latest_run} />
                  </button>

                  <Badge variant={strat.default_mode} className="shrink-0">
                    {RUN_MODE_LABELS[strat.default_mode] || strat.default_mode}
                  </Badge>

                  <div className="hidden items-center gap-0.5 group-hover:flex group-focus-within:flex">
                    <button
                      type="button"
                      onClick={() => onAddShortcut(strat)}
                      className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
                      title="新建快捷方式"
                    >
                      <Plus className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => onRefreshStrategy(strat.id)}
                      className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
                      title="刷新元数据"
                    >
                      <RefreshCw className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => onRenameStrategy(strat)}
                      className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
                      title="重命名策略"
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => onDeleteStrategy(strat)}
                      className="rounded p-1 text-muted-foreground hover:bg-destructive/15 hover:text-destructive cursor-pointer"
                      title="移除策略"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>

                {isExpanded && strat.shortcuts.length > 0 && (
                  <div className="ml-5 space-y-0.5 border-l pl-2">
                    {strat.shortcuts.map((sc) => {
                      const isScSelected =
                        selection?.kind === "shortcut" && selection.shortcutId === sc.id;
                      return (
                        <div
                          key={sc.id}
                          className={cn(
                            "group flex items-center gap-1 rounded-md px-2 py-1 text-xs transition-colors",
                            isScSelected
                              ? "bg-primary/12 text-foreground font-medium"
                              : "hover:bg-accent/70 text-foreground/85",
                          )}
                        >
                          <button
                            type="button"
                            onClick={() =>
                              onSelect({
                                kind: "shortcut",
                                strategyId: strat.id,
                                shortcutId: sc.id,
                              })
                            }
                            onDoubleClick={() => onQuickRun({ shortcutId: sc.id })}
                            className="flex min-w-0 flex-1 items-center gap-1.5 text-left cursor-pointer"
                            title="双击运行此快捷方式"
                          >
                            <span className="truncate">{sc.name}</span>
                            <StatusDot active={sc.active_run} latest={sc.latest_run} />
                          </button>

                          <Badge variant={sc.mode} className="shrink-0">
                            {RUN_MODE_LABELS[sc.mode] || sc.mode}
                          </Badge>

                          <div className="hidden items-center gap-0.5 group-hover:flex group-focus-within:flex">
                            <button
                              type="button"
                              onClick={() => onQuickRun({ shortcutId: sc.id })}
                              className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-primary cursor-pointer"
                              title="运行"
                            >
                              <Play className="h-3 w-3" />
                            </button>
                            <button
                              type="button"
                              onClick={() => onDuplicateShortcut(sc)}
                              className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
                              title="复制快捷方式"
                            >
                              <Copy className="h-3 w-3" />
                            </button>
                            <button
                              type="button"
                              onClick={() => onDeleteShortcut(sc)}
                              className="rounded p-1 text-muted-foreground hover:bg-destructive/15 hover:text-destructive cursor-pointer"
                              title="删除快捷方式"
                            >
                              <Trash2 className="h-3 w-3" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
