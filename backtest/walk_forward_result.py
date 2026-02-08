"""Walk-forward optimization result container."""

from dataclasses import dataclass, field


@dataclass
class WalkForwardResult:
    """Results from a walk-forward optimization run."""

    windows: list = field(default_factory=list)  # Per-window: {is_params, is_metrics, oos_metrics}
    aggregated_oos: dict = field(default_factory=dict)  # Aggregated OOS metrics
    best_params: dict = field(default_factory=dict)  # Most frequently selected params
    avg_decay_pct: float = 0.0  # Average IS->OOS performance decay
    is_valid: bool = True  # True if avg_decay < 50% (CLAUDE.md guardrail)
    strategy_class: str = ""
    param_grid: dict = field(default_factory=dict)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary of the walk-forward result."""
        lines = [
            f"Walk-Forward Optimization: {self.strategy_class}",
            f"  Windows: {len(self.windows)}",
            f"  Best params: {self.best_params}",
            f"  Avg IS->OOS decay: {self.avg_decay_pct:.1f}%",
            f"  Valid (decay < 50%): {self.is_valid}",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        if self.aggregated_oos:
            lines.append(f"  Aggregated OOS metrics:")
            for k, v in self.aggregated_oos.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize result to a dictionary."""
        return {
            "windows": self.windows,
            "aggregated_oos": self.aggregated_oos,
            "best_params": self.best_params,
            "avg_decay_pct": self.avg_decay_pct,
            "is_valid": self.is_valid,
            "strategy_class": self.strategy_class,
            "param_grid": self.param_grid,
            "duration_seconds": self.duration_seconds,
        }
