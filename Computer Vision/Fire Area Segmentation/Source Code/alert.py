"""Fire Area Segmentation — alert logic (separate from segmentation).

Evaluates per-frame metrics and trend data to determine alert levels.
This module is intentionally decoupled from the segmentation pipeline
so it can be replaced or extended independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from metrics import FrameMetrics
from trend import TrendSummary


class AlertLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertState:
    """Current alert evaluation for a frame."""

    level: AlertLevel
    fire_coverage: float
    growing: bool           # fire area is increasing over recent frames
    reason: str


# ── configurable thresholds ────────────────────────────────

_THRESHOLDS = {
    AlertLevel.CRITICAL: 0.25,     # ≥25 % fire coverage
    AlertLevel.HIGH: 0.10,         # ≥10 %
    AlertLevel.MEDIUM: 0.03,       # ≥3 %
    AlertLevel.LOW: 0.005,         # ≥0.5 %
}


def evaluate_alert(
    metrics: FrameMetrics,
    trend: TrendSummary | None = None,
) -> AlertState:
    """Determine the alert level from metrics and optional trend."""
    cov = metrics.fire_coverage
    growing = False
    if trend is not None and trend.frames_seen >= 2:
        growing = trend.fire_growth_rate > 0

    level = AlertLevel.NONE
    reason = "No fire detected"

    for lvl in (AlertLevel.CRITICAL, AlertLevel.HIGH, AlertLevel.MEDIUM,
                AlertLevel.LOW):
        if cov >= _THRESHOLDS[lvl]:
            level = lvl
            reason = f"Fire coverage {cov:.2%} ≥ {_THRESHOLDS[lvl]:.1%}"
            break

    # Upgrade one level if fire is growing rapidly
    if growing and trend is not None and trend.fire_growth_rate > 0.005:
        upgrades = {
            AlertLevel.LOW: AlertLevel.MEDIUM,
            AlertLevel.MEDIUM: AlertLevel.HIGH,
            AlertLevel.HIGH: AlertLevel.CRITICAL,
        }
        if level in upgrades:
            reason += f" + rapid growth ({trend.fire_growth_rate:+.4%}/frame)"
            level = upgrades[level]

    return AlertState(
        level=level,
        fire_coverage=cov,
        growing=growing,
        reason=reason,
    )
