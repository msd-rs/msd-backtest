import { useState, type FormEvent } from "react";
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

interface AddStrategyDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (payload: { path: string; name?: string; notes?: string }) => Promise<void>;
}

export function AddStrategyDialog({ open, onOpenChange, onSubmit }: AddStrategyDialogProps) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const cleanPath = path.trim();
    if (!cleanPath) {
      setError("请输入策略或因子文件的完整本地路径。");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await onSubmit({
        path: cleanPath,
        name: name.trim() || undefined,
        notes: notes.trim() || undefined,
      });
      setPath("");
      setName("");
      setNotes("");
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent aria-labelledby="add-strategy-title">
        <DialogHeader>
          <DialogTitle id="add-strategy-title">添加策略 / 因子文件</DialogTitle>
          <DialogDescription>
            输入本地 `.py`（策略/选股器/因子）或 `.txt`/`.json`（因子表达式）路径，同一规范化路径将幂等刷新。
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="strategy-file-path">
              文件路径 <span className="text-destructive">*</span>
            </Label>
            <Input
              id="strategy-file-path"
              name="path"
              value={path}
              onChange={(e) => {
                setPath(e.target.value);
                if (error) setError("");
              }}
              placeholder="/home/user/repo/strategy/my_strategy.py"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="none"
              spellCheck={false}
              autoFocus
              required
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="strategy-display-name">显示名称（可选）</Label>
            <Input
              id="strategy-display-name"
              name="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="留空默认使用策略 docstring（无则使用文件名）"
              autoComplete="off"
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="strategy-notes">备注（可选）</Label>
            <Input
              id="strategy-notes"
              name="notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="策略说明或标签"
              autoComplete="off"
            />
          </div>

          {error && (
            <div role="alert" className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              取消
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? "提交中..." : "添加 / 刷新"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
