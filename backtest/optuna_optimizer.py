"""Optuna-based hyperparameter optimizer.

Lazy-imports optuna (optional dependency). If not installed, optimize() returns None.
Integrates with BacktestEngine for objective evaluation.
"""

import logging
import time

from backtest.optuna_result import OptunaResult

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """Optuna-based hyperparameter optimizer.

    Lazy-imports optuna (optional dep). If not installed, optimize() returns None.

    Config keys:
        n_trials: Number of optimization trials (default 100).
        n_jobs: Parallel jobs for Optuna study (default 1).
        sampler: Sampler type - "tpe" or "random" (default "tpe").
        pruner: Pruner type - "median" or "none" (default "median").
        metric: Default metric to optimize (default "sharpe_ratio").
        direction: "maximize" or "minimize" (default "maximize").
        timeout_s: Timeout in seconds (default 300).
    """

    def __init__(self, backtest_engine, risk_manager, config: dict):
        self.backtest_engine = backtest_engine
        self.risk_manager = risk_manager
        self.n_trials = config.get("n_trials", 100)
        self.n_jobs = config.get("n_jobs", 1)
        self.sampler = config.get("sampler", "tpe")
        self.pruner = config.get("pruner", "median")
        self.default_metric = config.get("metric", "sharpe_ratio")
        self.direction = config.get("direction", "maximize")
        self.timeout_s = config.get("timeout_s", 300)

    def is_available(self) -> bool:
        """Check if optuna is installed."""
        try:
            import optuna  # noqa: F401
            return True
        except ImportError:
            return False

    def optimize(self, df, strategy_class, param_space: dict,
                 metric: str = None) -> OptunaResult | None:
        """Run Optuna optimization.

        Args:
            df: OHLCV DataFrame.
            strategy_class: BaseStrategy subclass to instantiate.
            param_space: Parameter space definition. For each param:
                - list -> suggest_categorical
                - {"low": x, "high": y} with int values -> suggest_int
                - {"low": x, "high": y} with float values -> suggest_float
                - {"low": x, "high": y, "log": True} -> suggest_float(log=True)
            metric: Metric to optimize (overrides config default).

        Returns:
            OptunaResult or None if optuna is not installed.
        """
        try:
            import optuna
        except ImportError:
            logger.warning("optuna not installed, returning None")
            return None

        metric = metric or self.default_metric
        start_time = time.time()

        # Create sampler
        if self.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        else:
            sampler = optuna.samplers.TPESampler()

        # Create pruner
        if self.pruner == "none":
            pruner = optuna.pruners.NopPruner()
        else:
            pruner = optuna.pruners.MedianPruner()

        # Suppress optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

        objective = self._create_objective(df, strategy_class, param_space, metric)

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout_s,
        )

        # Build trial history
        trial_history = []
        for trial in study.trials:
            trial_history.append({
                "params": trial.params,
                "score": trial.value if trial.value is not None else 0.0,
                "state": str(trial.state),
                "duration_s": (trial.datetime_complete - trial.datetime_start).total_seconds()
                if trial.datetime_complete and trial.datetime_start else 0.0,
            })

        # Compute param importance
        param_importance = {}
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            logger.debug("Could not compute param importance")

        n_completed = sum(1 for t in study.trials
                          if str(t.state) == "TrialState.COMPLETE")
        n_pruned = sum(1 for t in study.trials
                       if str(t.state) == "TrialState.PRUNED")

        return OptunaResult(
            best_params=study.best_params if study.best_trial else {},
            best_score=study.best_value if study.best_trial else 0.0,
            n_trials=len(study.trials),
            n_completed=n_completed,
            n_pruned=n_pruned,
            param_importance=param_importance,
            trial_history=trial_history,
            duration_seconds=time.time() - start_time,
            strategy_class=strategy_class.__name__,
            metric=metric,
        )

    def _create_objective(self, df, strategy_class, param_space, metric):
        """Create objective function for Optuna trial."""
        def objective(trial):
            params = self._suggest_params(trial, param_space)
            try:
                strategy = strategy_class(params)
                result = self.backtest_engine.run(df, strategy, self.risk_manager)
                score = self._extract_metric(result, metric)
                return score
            except Exception:
                logger.debug("Trial failed with params: %s", params)
                return float("-inf") if self.direction == "maximize" else float("inf")

        return objective

    def _suggest_params(self, trial, param_space: dict) -> dict:
        """Map param_space to Optuna suggest calls.

        Mapping rules:
            - list -> suggest_categorical
            - {"low": int, "high": int} -> suggest_int
            - {"low": float, "high": float} -> suggest_float
            - {"low": float, "high": float, "log": True} -> suggest_float(log=True)
        """
        params = {}
        for name, spec in param_space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif isinstance(spec, dict):
                low = spec["low"]
                high = spec["high"]
                use_log = spec.get("log", False)
                if isinstance(low, int) and isinstance(high, int) and not use_log:
                    step = spec.get("step", 1)
                    params[name] = trial.suggest_int(name, low, high, step=step)
                else:
                    params[name] = trial.suggest_float(
                        name, float(low), float(high), log=use_log)
            else:
                # Single fixed value
                params[name] = spec
        return params

    def _extract_metric(self, result, metric: str) -> float:
        """Extract metric from backtest result.

        Computes standard metrics from the BacktestResult and returns
        the requested one.
        """
        trades = result.trades
        equity_curve = result.equity_curve

        if not trades:
            return 0.0

        # Total return
        total_return = (result.final_equity - result.initial_capital) / result.initial_capital

        if metric == "total_return":
            return total_return

        # Win rate
        winners = [t for t in trades if t.pnl > 0]
        win_rate = len(winners) / len(trades) * 100.0

        if metric == "win_rate":
            return win_rate

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        if metric == "profit_factor":
            return profit_factor

        # Max drawdown
        if len(equity_curve) > 0:
            import numpy as np
            peak = equity_curve.cummax()
            dd = (equity_curve - peak) / peak * 100
            max_drawdown = abs(dd.min()) if len(dd) > 0 else 0.0
        else:
            max_drawdown = 0.0

        if metric == "max_drawdown":
            return -max_drawdown  # Negative so maximizing = minimizing DD

        # Sharpe ratio (annualized, assuming hourly bars)
        if len(equity_curve) > 1:
            import numpy as np
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(8760))
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        if metric == "sharpe_ratio":
            return sharpe_ratio

        # Calmar ratio
        calmar = total_return / (max_drawdown / 100.0) if max_drawdown > 0 else 0.0

        if metric == "calmar_ratio":
            return calmar

        # Default: sharpe
        return sharpe_ratio
