# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

"""Daily portfolio signal formatting and multi-target webhook notification via Apprise."""

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("mbt.notify")

VALID_DETAILS = ("compact", "minimal")
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass
class NotifyTarget:
  """Configuration for a single webhook notification target."""

  url: str
  name: str = ""
  detail: str = "compact"
  on_trade_only: bool = False
  keyword: str = ""
  enabled: bool = True

  def __post_init__(self):
    if not self.url or not isinstance(self.url, str):
      raise ValueError("NotifyTarget.url must be a non-empty string")
    if self.detail not in VALID_DETAILS:
      raise ValueError(
        f"Unsupported detail level '{self.detail}', expected one of {VALID_DETAILS}"
      )


def _expand_url_env(raw_url: str, label: str) -> str:
  """Expand environment variables in a target URL and fail fast if any are unset."""
  missing: list[str] = []

  def _replace(match: re.Match[str]) -> str:
    var_name = match.group(1) or match.group(2)
    val = os.environ.get(var_name)
    if val is None or val == "":
      missing.append(var_name)
      return match.group(0)
    return val

  expanded = _ENV_VAR_PATTERN.sub(_replace, raw_url.strip())
  if missing:
    raise ValueError(
      f"Notification target '{label}' references unset environment variable(s): {', '.join(missing)}"
    )
  if not expanded:
    raise ValueError(f"Notification target '{label}' resolved to an empty URL")
  return expanded


def load_notify_config(path: str) -> list[NotifyTarget]:
  """Load and validate multi-target notification config from a JSON file.

  Supports either a JSON array of target objects or ``{"targets": [...]}``.
  Each target object supports:
    - ``url`` (required): Apprise URI (e.g. ``slack://...``, ``feishu://...``),
      supports ``${ENV_VAR}`` expansion.
    - ``name`` (optional): Human-readable label for logging.
    - ``detail`` (optional): ``"compact"`` (default) or ``"minimal"``.
    - ``on_trade_only`` (optional): bool, skip when no trade/pending signal (default False).
    - ``keyword`` (optional): str, required security keyword ensured in message title.
    - ``enabled`` (optional): bool, whether target is active (default True).
  """
  if not os.path.isfile(path):
    raise FileNotFoundError(f"Notification config file not found: {path}")

  with open(path, "r", encoding="utf-8") as f:
    raw = json.load(f)

  if isinstance(raw, dict):
    items = raw.get("targets")
    if not isinstance(items, list):
      raise ValueError(
        f"Invalid notification config in {path}: expected a JSON list or {{\"targets\": [...]}}"
      )
  elif isinstance(raw, list):
    items = raw
  else:
    raise ValueError(f"Invalid notification config in {path}: expected list or object")

  targets: list[NotifyTarget] = []
  for idx, item in enumerate(items):
    if not isinstance(item, dict):
      raise ValueError(f"Invalid target at index {idx} in {path}: expected object")
    enabled = bool(item.get("enabled", True))
    name = str(item.get("name") or f"target-{idx}")
    raw_url = item.get("url")
    if not raw_url or not isinstance(raw_url, str):
      raise ValueError(f"Missing or invalid 'url' for target '{name}' in {path}")
    url = _expand_url_env(raw_url, name) if enabled else raw_url
    detail = str(item.get("detail", "compact"))
    on_trade_only = bool(item.get("on_trade_only", False))
    keyword = str(item.get("keyword") or "")
    targets.append(
      NotifyTarget(
        url=url,
        name=name,
        detail=detail,
        on_trade_only=on_trade_only,
        keyword=keyword,
        enabled=enabled,
      )
    )

  return targets


def has_trade_signals(snapshot: dict[str, Any]) -> bool:
  """Return True if the snapshot contains executed trades or pending signals."""
  return bool(snapshot.get("trades") or snapshot.get("pending_signals"))


def _format_signal_lines(snapshot: dict[str, Any]) -> list[str]:
  trades = snapshot.get("trades") or []
  pending = snapshot.get("pending_signals") or []
  lines: list[str] = []

  if trades:
    lines.append("**Executed Trades (Today)**")
    for t in trades:
      lines.append(
        f"- `{t['action']}` **{t['obj']}**: {t['shares']:.0f} sh @ {t['deal_price']:.3f} "
        f"(amt: {t['amount']:,.0f}, fee: {t['fee']:.2f})"
      )

  if pending:
    lines.append("**Pending Signals (Next Open)**")
    for p in pending:
      tw = p.get("target_weight")
      sh = p.get("est_shares")
      parts = [f"ref: {p['ref_price']:.3f}"]
      if tw is not None:
        parts.append(f"target_w: {tw:.1%}")
      if sh is not None and sh > 0:
        parts.append(f"est: {sh:.0f} sh")
      if p.get("buy_percent") is not None:
        parts.append(f"buy_pct: {p['buy_percent']:.1%}")
      if p.get("sell_percent") is not None:
        parts.append(f"sell_pct: {p['sell_percent']:.1%}")
      lines.append(f"- `{p['action']}` **{p['obj']}**: {', '.join(parts)}")

  if not trades and not pending:
    lines.append("- No rebalance / trade signal today (HOLD)")

  return lines


def format_daily_signal(
  snapshot: dict[str, Any],
  detail: str = "compact",
  strategy_name: str = "",
) -> tuple[str, str]:
  """Format a PortfolioAccount daily snapshot into ``(title, markdown_body)``."""
  if detail not in VALID_DETAILS:
    raise ValueError(f"Unsupported detail level '{detail}', expected one of {VALID_DETAILS}")

  date_str = snapshot.get("date") or "N/A"
  prefix = f"[{strategy_name}] " if strategy_name else ""
  has_sig = has_trade_signals(snapshot)
  tag = "Signal" if has_sig else "Daily"
  title = f"{prefix}Portfolio {tag} ({date_str})"

  pos_ratio = float(snapshot.get("position_ratio", 0.0))
  exec_mode = snapshot.get("exec_mode", "close")

  if detail == "minimal":
    lines = [
      f"**Date**: {date_str} | **Mode**: `{exec_mode}` | **Position**: {pos_ratio:.1%}",
      "",
    ]
    lines.extend(_format_signal_lines(snapshot))
    return title, "\n".join(lines)

  total_assets = float(snapshot.get("total_assets", 0.0))
  cash = float(snapshot.get("cash", 0.0))
  daily_ror = float(snapshot.get("daily_ror", 0.0))
  total_ror = float(snapshot.get("total_ror", 0.0))
  holdings = snapshot.get("holdings") or []

  lines = [
    f"**Date**: {date_str} | **Mode**: `{exec_mode}`",
    f"**NAV**: {total_assets:,.2f} | **Daily**: {daily_ror:+.2%} | **Total**: {total_ror:+.2%}",
    f"**Position**: {pos_ratio:.1%} | **Cash**: {cash:,.2f} | **Holdings**: {len(holdings)}",
    "",
  ]
  lines.extend(_format_signal_lines(snapshot))

  if holdings:
    lines.append("")
    lines.append("**Current Holdings**")
    for h in holdings:
      lines.append(
        f"- **{h['obj']}**: w={h['weight']:.1%}, mv={h['market_value']:,.0f}, "
        f"ror={h['ror']:+.2%} ({h['shares']:.0f} sh @ {h['price']:.3f})"
      )

  return title, "\n".join(lines)


def send_notifications(
  snapshot: dict[str, Any],
  targets: list[NotifyTarget],
  *,
  strategy_name: str = "",
  on_trade_only: bool | None = None,
  detail_override: str | None = None,
) -> dict[str, bool]:
  """Deliver formatted daily signal notifications to configured targets via Apprise.

  Raises ImportError if ``apprise`` is not installed. Individual target delivery
  failures are logged as errors without aborting remaining targets.
  """
  if detail_override is not None and detail_override not in VALID_DETAILS:
    raise ValueError(
      f"Unsupported detail level '{detail_override}', expected one of {VALID_DETAILS}"
    )

  active_targets = [t for t in targets if t.enabled]
  if not active_targets:
    logger.info("No enabled notification targets configured; skipping delivery.")
    return {}

  try:
    import apprise
  except ImportError as exc:
    raise ImportError(
      "apprise is required for webhook notifications. Install with: uv add 'msd-backtest[notify]'"
    ) from exc

  has_sig = has_trade_signals(snapshot)
  results: dict[str, bool] = {}

  for target in active_targets:
    label = target.name or target.url
    effective_on_trade = on_trade_only if on_trade_only is not None else target.on_trade_only
    if effective_on_trade and not has_sig:
      logger.info(f"Skipping notification for '{label}': no trade/pending signal today.")
      continue

    effective_detail = detail_override or target.detail
    title, body = format_daily_signal(
      snapshot, detail=effective_detail, strategy_name=strategy_name
    )
    if target.keyword and target.keyword not in title and target.keyword not in body:
      title = f"[{target.keyword}] {title}"

    apobj = apprise.Apprise()
    if not apobj.add(target.url):
      logger.error(f"Failed to register notification target '{label}': invalid Apprise URL")
      results[label] = False
      continue

    try:
      ok = bool(
        apobj.notify(
          title=title,
          body=body,
          body_format=apprise.NotifyFormat.MARKDOWN,
        )
      )
    except Exception as exc:
      logger.error(f"Exception delivering notification to '{label}': {exc}")
      results[label] = False
      continue

    if ok:
      logger.info(f"Notification delivered to '{label}' ({effective_detail})")
    else:
      logger.error(f"Notification delivery failed for '{label}'")
    results[label] = ok

  return results
