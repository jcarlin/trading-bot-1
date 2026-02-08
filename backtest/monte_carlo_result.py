"""Monte Carlo simulation result container."""

from dataclasses import dataclass, field


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run."""

    n_simulations: int = 0
    metric_distributions: dict = field(default_factory=dict)  # {metric: [values]}
    percentiles: dict = field(default_factory=dict)  # {metric: {5:v, 25:v, 50:v, 75:v, 95:v}}
    confidence_intervals: dict = field(default_factory=dict)  # {metric: {lower, upper, level}}
    original_metrics: dict = field(default_factory=dict)
    p_value_vs_random: float = 1.0
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """Return a human-readable summary of the Monte Carlo result."""
        lines = [
            f"Monte Carlo Simulation",
            f"  Simulations: {self.n_simulations}",
            f"  p-value vs random: {self.p_value_vs_random:.4f}",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        if self.percentiles:
            lines.append("  Percentiles:")
            for metric, pcts in self.percentiles.items():
                pct_str = ", ".join(f"p{k}={v:.4f}" for k, v in sorted(pcts.items()))
                lines.append(f"    {metric}: {pct_str}")
        if self.original_metrics:
            lines.append("  Original metrics:")
            for k, v in self.original_metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize result to a dictionary."""
        return {
            "n_simulations": self.n_simulations,
            "metric_distributions": self.metric_distributions,
            "percentiles": self.percentiles,
            "confidence_intervals": self.confidence_intervals,
            "original_metrics": self.original_metrics,
            "p_value_vs_random": self.p_value_vs_random,
            "duration_seconds": self.duration_seconds,
        }
