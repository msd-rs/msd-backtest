import json
import sys
import types
import numpy as np
import pytest

from mbt import DataProvider, PortfolioAccount, PortfolioContext, PortfolioRunner, Strategy
from mbt.notify import (
  NotifyTarget,
  format_daily_signal,
  has_trade_signals,
  load_notify_config,
  send_notifications,
)


def _make_dp(prices_a: np.ndarray, prices_b: np.ndarray) -> DataProvider:
  n = len(prices_a)
  prices = np.concatenate([prices_a, prices_b])
  ts = np.tile(
    np.array(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"][:n], dtype="datetime64[D]"),
    2,
  )
  return DataProvider(
    {
      "ts": ts,
      "open": prices,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(2 * n),
      "transfer_shares": np.zeros(2 * n),
      "right_shares": np.zeros(2 * n),
      "right_price": np.zeros(2 * n),
    },
    objs=["SH600000", "SZ000001"],
  )


def test_daily_snapshot_close_mode():
  dp = _make_dp(
    np.array([10.0, 10.0, 12.0]),
    np.array([20.0, 20.0, 20.0]),
  )

  class BuyLastBarStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):  # type: ignore[override]
      if ctx.dp.i == 1:
        ctx.rebalance(np.array([0.5, 0.0]))
      elif ctx.dp.i == 2:
        ctx.rebalance(np.array([0.5, 0.5]))

  acc = PortfolioAccount(10000.0, dp)
  PortfolioRunner(acc, exec_mode="close").run(BuyLastBarStrategy())

  snap = acc.daily_snapshot()
  assert snap["date"] == "2026-01-07"
  assert snap["exec_mode"] == "close"
  assert len(snap["trades"]) >= 1
  assert any(t["obj"] == "SZ000001" and t["action"] == "BUY" for t in snap["trades"])
  assert snap["pending_signals"] == []
  assert len(snap["holdings"]) == 2
  assert snap["total_assets"] > 10000.0
  assert snap["daily_ror"] > 0.0
  assert has_trade_signals(snap) is True


def test_daily_snapshot_next_open_pending_signals():
  dp = _make_dp(
    np.array([10.0, 10.0, 10.0, 10.0]),
    np.array([20.0, 20.0, 20.0, 20.0]),
  )

  class NextOpenSignalStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):  # type: ignore[override]
      if ctx.dp.i == 1:
        ctx.rebalance(np.array([1.0, 0.0]))
      elif ctx.dp.i == 3:
        # Generated on final bar close -> pending for next open
        ctx.rebalance(np.array([0.0, 1.0]))

  acc = PortfolioAccount(10000.0, dp)
  PortfolioRunner(acc, exec_mode="next_open").run(NextOpenSignalStrategy())

  snap = acc.daily_snapshot()
  assert snap["exec_mode"] == "next_open"
  assert snap["trades"] == []
  assert len(snap["pending_signals"]) == 2
  actions = {s["obj"]: s["action"] for s in snap["pending_signals"]}
  assert actions == {"SH600000": "SELL", "SZ000001": "BUY"}
  assert has_trade_signals(snap) is True


def test_load_notify_config_and_env_expansion(tmp_path, monkeypatch):
  monkeypatch.setenv("TEST_FEISHU_KEY", "abc123")
  cfg_file = tmp_path / "notify.json"
  cfg_file.write_text(
    json.dumps({
      "targets": [
        {
          "name": "feishu",
          "url": "feishu://${TEST_FEISHU_KEY}",
          "detail": "compact",
          "on_trade_only": False,
          "keyword": "mbt-signal",
        },
        {
          "name": "disabled-slack",
          "url": "slack://${UNSET_SLACK_VAR}",
          "detail": "minimal",
          "enabled": False,
        },
      ]
    }),
    encoding="utf-8",
  )

  targets = load_notify_config(str(cfg_file))
  assert len(targets) == 2
  assert targets[0].url == "feishu://abc123"
  assert targets[0].keyword == "mbt-signal"
  assert targets[0].enabled is True
  assert targets[1].enabled is False


def test_load_notify_config_fail_fast_unset_env(tmp_path, monkeypatch):
  monkeypatch.delenv("MISSING_WEBHOOK_VAR", raising=False)
  cfg_file = tmp_path / "notify_bad.json"
  cfg_file.write_text(
    json.dumps([{"name": "bad", "url": "feishu://${MISSING_WEBHOOK_VAR}"}]),
    encoding="utf-8",
  )

  with pytest.raises(ValueError, match="MISSING_WEBHOOK_VAR"):
    load_notify_config(str(cfg_file))


def test_format_daily_signal_compact_and_minimal():
  snap = {
    "date": "2026-01-07",
    "exec_mode": "close",
    "total_assets": 1050000.0,
    "cash": 50000.0,
    "market_value": 1000000.0,
    "position_ratio": 0.9524,
    "daily_ror": 0.0125,
    "total_ror": 0.05,
    "holdings": [
      {
        "obj": "SH600000",
        "shares": 10000.0,
        "price": 10.0,
        "market_value": 100000.0,
        "weight": 0.0952,
        "cost_price": 9.5,
        "ror": 0.0526,
        "unrealized_pnl": 5000.0,
        "holding_days": 5,
      }
    ],
    "trades": [],
    "pending_signals": [],
  }

  title_c, body_c = format_daily_signal(snap, detail="compact", strategy_name="DemoStrat")
  assert "[DemoStrat] Portfolio Daily (2026-01-07)" == title_c
  assert "Current Holdings" in body_c
  assert "No rebalance / trade signal today (HOLD)" in body_c
  assert "SH600000" in body_c

  title_m, body_m = format_daily_signal(snap, detail="minimal", strategy_name="DemoStrat")
  assert title_m == title_c
  assert "Current Holdings" not in body_m
  assert "No rebalance / trade signal today (HOLD)" in body_m


def test_send_notifications_handles_on_trade_and_failure(monkeypatch, caplog):
  sent_calls: list[tuple[str, str, str]] = []

  class FakeApprise:
    def __init__(self):
      self.urls: list[str] = []

    def add(self, url: str) -> bool:
      self.urls.append(url)
      return True

    def notify(self, title: str, body: str, body_format=None) -> bool:
      sent_calls.append((self.urls[0], title, body))
      return "fail" not in self.urls[0]

  fake_mod = types.ModuleType("apprise")
  fake_mod.Apprise = FakeApprise  # type: ignore[attr-defined]
  fake_mod.NotifyFormat = types.SimpleNamespace(MARKDOWN="markdown")  # type: ignore[attr-defined]
  monkeypatch.setitem(sys.modules, "apprise", fake_mod)

  hold_snap = {
    "date": "2026-01-07",
    "exec_mode": "close",
    "total_assets": 100000.0,
    "cash": 100000.0,
    "position_ratio": 0.0,
    "daily_ror": 0.0,
    "total_ror": 0.0,
    "holdings": [],
    "trades": [],
    "pending_signals": [],
  }
  targets = [
    NotifyTarget(url="feishu://ok", name="ok-target", on_trade_only=False, keyword="SEC_KW"),
    NotifyTarget(url="slack://skip", name="trade-only-target", on_trade_only=True),
    NotifyTarget(url="feishu://fail", name="fail-target", on_trade_only=False),
  ]

  res = send_notifications(hold_snap, targets, strategy_name="Test")
  assert res == {"ok-target": True, "fail-target": False}
  assert len(sent_calls) == 2
  assert sent_calls[0][1].startswith("[SEC_KW] ")
  assert "Notification delivery failed for 'fail-target'" in caplog.text
