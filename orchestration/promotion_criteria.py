"""Configurable promotion criteria for A/B testing shadow strategies."""

import logging
import math

logger = logging.getLogger(__name__)


class PromotionCriteria:
    """Evaluates whether a shadow strategy meets criteria for promotion.

    Checks minimum duration, trade count, Sharpe, win rate, drawdown,
    and improvement over the live strategy.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.min_duration_hours = config.get("min_duration_hours", 168)
        self.min_trades = config.get("min_trades", 20)
        self.min_sharpe = config.get("min_sharpe", 0.5)
        self.min_win_rate = config.get("min_win_rate", 45.0)
        self.max_dd = config.get("max_dd", 5.0)
        self.min_improvement_pct = config.get("min_improvement_pct", 10.0)
        self.significance_level = config.get("significance_level", 0.05)

    def evaluate(self, shadow_metrics: dict, live_metrics: dict,
                 duration_hours: float) -> dict:
        """Evaluate shadow strategy against promotion criteria.

        Args:
            shadow_metrics: Performance summary from ShadowRunner.
            live_metrics: Performance metrics from live strategy.
            duration_hours: How long the A/B test has been running.

        Returns:
            Dict with meets_criteria (bool) and per-check details.
        """
        checks = {}

        # Duration check
        checks["duration"] = {
            "passed": duration_hours >= self.min_duration_hours,
            "required": self.min_duration_hours,
            "actual": duration_hours,
        }

        # Trade count
        shadow_trades = shadow_metrics.get("trade_count", 0)
        checks["trades"] = {
            "passed": shadow_trades >= self.min_trades,
            "required": self.min_trades,
            "actual": shadow_trades,
        }

        # Sharpe
        shadow_sharpe = shadow_metrics.get("sharpe", 0.0)
        checks["sharpe"] = {
            "passed": shadow_sharpe >= self.min_sharpe,
            "required": self.min_sharpe,
            "actual": shadow_sharpe,
        }

        # Win rate
        shadow_wr = shadow_metrics.get("win_rate", 0.0)
        checks["win_rate"] = {
            "passed": shadow_wr >= self.min_win_rate,
            "required": self.min_win_rate,
            "actual": shadow_wr,
        }

        # Max drawdown
        shadow_dd = shadow_metrics.get("max_drawdown", 0.0)
        checks["drawdown"] = {
            "passed": shadow_dd <= self.max_dd,
            "required": self.max_dd,
            "actual": shadow_dd,
        }

        # Improvement over live (total_pnl comparison)
        live_pnl = live_metrics.get("total_pnl", 0.0)
        shadow_pnl = shadow_metrics.get("total_pnl", 0.0)

        if live_pnl > 0:
            improvement = ((shadow_pnl - live_pnl) / live_pnl) * 100
        elif live_pnl == 0:
            improvement = 100.0 if shadow_pnl > 0 else 0.0
        else:
            # Live is negative — shadow just needs to be better
            improvement = ((shadow_pnl - live_pnl) / abs(live_pnl)) * 100

        checks["improvement"] = {
            "passed": improvement >= self.min_improvement_pct,
            "required": self.min_improvement_pct,
            "actual": round(improvement, 2),
        }

        meets_criteria = all(c["passed"] for c in checks.values())

        return {
            "meets_criteria": meets_criteria,
            "checks": checks,
        }

    def is_statistically_significant(self, shadow_returns: list[float],
                                     live_returns: list[float]) -> tuple[bool, float]:
        """Welch's t-test for difference in means (no scipy dependency).

        Args:
            shadow_returns: List of per-trade PnL values from shadow.
            live_returns: List of per-trade PnL values from live.

        Returns:
            (is_significant, p_value) tuple.
        """
        n1 = len(shadow_returns)
        n2 = len(live_returns)

        if n1 < 2 or n2 < 2:
            return False, 1.0

        mean1 = sum(shadow_returns) / n1
        mean2 = sum(live_returns) / n2

        var1 = sum((x - mean1) ** 2 for x in shadow_returns) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in live_returns) / (n2 - 1)

        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return False, 1.0

        t_stat = (mean1 - mean2) / se

        # Welch–Satterthwaite degrees of freedom
        num = (var1 / n1 + var2 / n2) ** 2
        denom = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
        df = num / denom if denom > 0 else 1.0

        # Approximate p-value using the t-distribution CDF approximation
        p_value = self._approx_t_pvalue(abs(t_stat), df)

        return p_value < self.significance_level, round(p_value, 6)

    @staticmethod
    def _approx_t_pvalue(t: float, df: float) -> float:
        """Approximate two-tailed p-value for t-distribution.

        Uses the approximation: p ≈ 2 * (1 - Φ(t * sqrt(df / (df - 2 + t²))))
        where Φ is the standard normal CDF, valid for df > 2.

        For small df, falls back to a cruder approximation.
        """
        if df <= 0:
            return 1.0

        # Convert t to approximate z-score
        if df > 2:
            z = t * math.sqrt(df / (df - 2 + t * t))
        else:
            z = t * math.sqrt(df / (df + t * t))

        # Standard normal CDF approximation (Abramowitz and Stegun)
        p_one_tail = 0.5 * math.erfc(z / math.sqrt(2))

        return 2.0 * p_one_tail
