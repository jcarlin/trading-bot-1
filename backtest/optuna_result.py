"""Optuna optimization result container."""

from dataclasses import dataclass, field


@dataclass
class OptunaResult:
    """Results from an Optuna hyperparameter optimization run."""

    best_params: dict = field(default_factory=dict)
    best_score: float = 0.0
    n_trials: int = 0
    n_completed: int = 0
    n_pruned: int = 0
    param_importance: dict = field(default_factory=dict)  # {param_name: importance}
    trial_history: list = field(default_factory=list)  # [{params, score, state, duration_s}]
    duration_seconds: float = 0.0
    strategy_class: str = ""
    metric: str = ""

    def summary(self) -> str:
        """Return a human-readable summary of the optimization result."""
        lines = [
            f"Optuna Optimization: {self.strategy_class}",
            f"  Metric: {self.metric}",
            f"  Best score: {self.best_score:.4f}",
            f"  Best params: {self.best_params}",
            f"  Trials: {self.n_completed} completed, {self.n_pruned} pruned, {self.n_trials} total",
            f"  Duration: {self.duration_seconds:.1f}s",
        ]
        if self.param_importance:
            lines.append("  Param importance:")
            for name, imp in sorted(self.param_importance.items(),
                                     key=lambda x: x[1], reverse=True):
                lines.append(f"    {name}: {imp:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize result to a dictionary."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_pruned": self.n_pruned,
            "param_importance": self.param_importance,
            "trial_history": self.trial_history,
            "duration_seconds": self.duration_seconds,
            "strategy_class": self.strategy_class,
            "metric": self.metric,
        }
