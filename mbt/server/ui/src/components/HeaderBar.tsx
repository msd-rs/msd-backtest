import { BookOpen, Moon, Settings, Sun } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface HeaderBarProps {
  healthText: string;
  healthOk: boolean;
  darkMode: boolean;
  onToggleDark: () => void;
  onOpenSettings: () => void;
}

export function HeaderBar({
  healthText,
  healthOk,
  darkMode,
  onToggleDark,
  onOpenSettings,
}: HeaderBarProps) {
  const versionMatch = healthText.match(/^OK,\s*msd-backtest\s+(.+)$/i);
  const badgeText = healthOk
    ? `v${(versionMatch?.[1] || healthText).trim()}`
    : healthText
      ? "离线"
      : "连接中...";

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b bg-card px-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold tracking-tight">MSD Backtest Studio</span>
        <Badge
          variant={healthOk ? "select" : "destructive"}
          className="font-mono text-[11px]"
          title={healthText || "连接中..."}
        >
          {badgeText}
        </Badge>
      </div>

      <div className="flex items-center gap-2">
        <a
          href="/api/docs"
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-7 items-center gap-1.5 rounded-md border border-input bg-background px-2.5 text-xs font-medium hover:bg-accent hover:text-accent-foreground"
        >
          <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
          OpenAPI
        </a>
        <Button variant="outline" size="sm" onClick={onToggleDark} aria-label="切换主题">
          {darkMode ? (
            <Sun className="h-3.5 w-3.5" aria-hidden="true" />
          ) : (
            <Moon className="h-3.5 w-3.5" aria-hidden="true" />
          )}
          <span>{darkMode ? "浅色" : "深色"}</span>
        </Button>
        <Button variant="outline" size="sm" onClick={onOpenSettings}>
          <Settings className="h-3.5 w-3.5" aria-hidden="true" />
          <span>系统设置</span>
        </Button>
      </div>
    </header>
  );
}
