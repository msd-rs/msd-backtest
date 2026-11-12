"""mbt.strategy_templates: 官方标准策略模版库 (Strategy Cookbook & Templates).

本模块提供经过官方严格验证的高性能标准策略模版，涵盖三大典型量化范式：
1. `fundamental_timing`: 基本面筛选 + 趋势择时模版 (类费雪/德雷曼/成长价值双重驱动)
2. `multifactor_rotation`: 截面多因子打分全市场定期轮动模版 (质量+估值+动量定期等权/市值加权换仓)
3. `cta_momentum`: CTA 纯量价动量与通道跟踪模版 (唐奇安突破 + ATR 波动率追踪止损)

所有模版均严格遵循 Agent-First 契约与最佳实践：
- 零容忍复权与未复权价格混算，严禁直接手写浮盈计算；统一调用 `ctx.holding_ror(obj)` 与 `ctx.cost_price(obj, adjusted=True)`.
- 静态指标与特征在 `prepare()` 中进行向量化一次性预分配，消除 `on_bar` 重复计算开销.
- 估值计算与基本面指标严格使用原始未复权价格 `raw_close` 与披露日 backward as-of 对齐.
- 浮点数与零除使用 `np.errstate(divide='ignore', invalid='ignore')` 防御性安全处理.
"""

from __future__ import annotations

import os

TEMPLATES = {
  "fundamental_timing": "fundamental_timing.py",
  "multifactor_rotation": "multifactor_rotation.py",
  "cta_momentum": "cta_momentum.py",
}


def get_template_path(name: str) -> str:
  """获取指定策略模版的绝对文件路径.

  Args:
    name: 模版短名称 (如 'fundamental_timing') 或文件名 (如 'cta_momentum.py').

  Returns:
    模版文件的绝对路径.

  Raises:
    KeyError: 当模版名称不存在时抛出.
  """
  clean_name = name.removesuffix(".py")
  if clean_name not in TEMPLATES:
    available = ", ".join(repr(k) for k in TEMPLATES.keys())
    raise KeyError(f"Template '{name}' not found. Available templates: [{available}]")

  filename = TEMPLATES[clean_name]
  return os.path.join(os.path.dirname(__file__), filename)


def list_templates() -> list[str]:
  """返回所有可用的官方策略模版名称列表."""
  return sorted(list(TEMPLATES.keys()))


__all__ = [
  "TEMPLATES",
  "get_template_path",
  "list_templates",
]
