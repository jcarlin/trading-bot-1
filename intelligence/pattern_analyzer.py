"""Trading pattern analysis for wallet intelligence."""

import logging
import math
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class WalletPatternAnalyzer:
    """Identifies trading patterns from reconstructed positions."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def analyze_entry_patterns(self, positions: list[dict],
                                candles: Optional[dict[str, pd.DataFrame]] = None) -> dict:
        """Analyze when and how positions are entered.

        Args:
            positions: Reconstructed position list
            candles: Optional dict of {symbol: DataFrame} with OHLCV data

        Returns:
            Dict with time_of_day_bias, day_of_week_bias,
            avg_entry_vs_daily_range, funding_correlation
        """
        if not positions:
            return {
                "time_of_day_bias": {},
                "day_of_week_bias": {},
                "avg_entry_vs_daily_range": 0.0,
                "funding_correlation": 0.0,
            }

        # Time-of-day analysis
        hour_counts: dict[int, int] = {}
        day_counts: dict[int, int] = {}

        for pos in positions:
            entry_time = pos.get("entry_time")
            if isinstance(entry_time, datetime):
                hour = entry_time.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
                day = entry_time.weekday()
                day_counts[day] = day_counts.get(day, 0) + 1

        # Normalize to percentages
        total_entries = sum(hour_counts.values()) or 1
        time_of_day_bias = {
            h: round(c / total_entries * 100, 1)
            for h, c in sorted(hour_counts.items())
        }

        total_days = sum(day_counts.values()) or 1
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_of_week_bias = {
            day_names[d]: round(c / total_days * 100, 1)
            for d, c in sorted(day_counts.items())
        }

        # Entry vs daily range (if candles available)
        avg_entry_vs_range = 0.0
        if candles:
            ratios = []
            for pos in positions:
                symbol = pos.get("symbol", "")
                avg_entry = pos.get("avg_entry", 0)
                entry_time = pos.get("entry_time")
                if symbol in candles and avg_entry > 0 and isinstance(entry_time, datetime):
                    df = candles[symbol]
                    if hasattr(df.index, 'date'):
                        day_data = df[df.index.date == entry_time.date()]
                        if not day_data.empty:
                            high = day_data["high"].max()
                            low = day_data["low"].min()
                            daily_range = high - low
                            if daily_range > 0:
                                ratio = (avg_entry - low) / daily_range
                                ratios.append(ratio)
            if ratios:
                avg_entry_vs_range = sum(ratios) / len(ratios)

        return {
            "time_of_day_bias": time_of_day_bias,
            "day_of_week_bias": day_of_week_bias,
            "avg_entry_vs_daily_range": round(avg_entry_vs_range, 4),
            "funding_correlation": 0.0,  # Requires funding rate data
        }

    def analyze_sizing_patterns(self, positions: list[dict]) -> dict:
        """Analyze position sizing patterns.

        Returns:
            Dict with sizing_type, avg_size, size_std, size_vs_confidence_correlation
        """
        if not positions:
            return {
                "sizing_type": "unknown",
                "avg_size": 0.0,
                "size_std": 0.0,
                "size_vs_confidence_correlation": 0.0,
            }

        sizes = [p.get("max_size", 0) for p in positions if p.get("max_size", 0) > 0]
        if not sizes:
            return {
                "sizing_type": "unknown",
                "avg_size": 0.0,
                "size_std": 0.0,
                "size_vs_confidence_correlation": 0.0,
            }

        avg_size = sum(sizes) / len(sizes)
        if len(sizes) > 1:
            variance = sum((s - avg_size) ** 2 for s in sizes) / (len(sizes) - 1)
            size_std = math.sqrt(variance)
        else:
            size_std = 0.0

        # Classify sizing type
        cv = size_std / avg_size if avg_size > 0 else 0
        if cv < 0.1:
            sizing_type = "fixed"
        elif cv < 0.5:
            sizing_type = "volatility_scaled"
        else:
            sizing_type = "conviction"

        return {
            "sizing_type": sizing_type,
            "avg_size": round(avg_size, 6),
            "size_std": round(size_std, 6),
            "size_vs_confidence_correlation": 0.0,
        }

    def analyze_exit_patterns(self, positions: list[dict]) -> dict:
        """Analyze position exit patterns.

        Returns:
            Dict with exit_type, avg_hold_hours, avg_profit_target_pct, avg_stop_loss_pct
        """
        if not positions:
            return {
                "exit_type": "unknown",
                "avg_hold_hours": 0.0,
                "avg_profit_target_pct": 0.0,
                "avg_stop_loss_pct": 0.0,
            }

        hold_hours = []
        profit_pcts = []
        loss_pcts = []

        for pos in positions:
            # Hold duration
            duration = pos.get("hold_duration")
            if duration is not None:
                hold_hours.append(duration / 3600)

            # PnL as percentage of entry
            avg_entry = pos.get("avg_entry", 0)
            total_pnl = pos.get("total_pnl", 0)
            max_size = pos.get("max_size", 0)

            if avg_entry > 0 and max_size > 0:
                pnl_pct = (total_pnl / (avg_entry * max_size)) * 100
                if total_pnl > 0:
                    profit_pcts.append(pnl_pct)
                elif total_pnl < 0:
                    loss_pcts.append(abs(pnl_pct))

        avg_hold = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0
        avg_profit = sum(profit_pcts) / len(profit_pcts) if profit_pcts else 0.0
        avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0

        # Classify exit type based on hold duration variance
        if hold_hours and len(hold_hours) > 1:
            h_mean = sum(hold_hours) / len(hold_hours)
            h_var = sum((h - h_mean) ** 2 for h in hold_hours) / (len(hold_hours) - 1)
            h_cv = math.sqrt(h_var) / h_mean if h_mean > 0 else 0

            if h_cv < 0.2:
                exit_type = "time_based"
            elif avg_profit > 0 and avg_loss > 0 and abs(avg_profit - avg_loss) / max(avg_profit, avg_loss) < 0.3:
                exit_type = "fixed_target"
            else:
                exit_type = "signal"
        else:
            exit_type = "unknown"

        return {
            "exit_type": exit_type,
            "avg_hold_hours": round(avg_hold, 2),
            "avg_profit_target_pct": round(avg_profit, 4),
            "avg_stop_loss_pct": round(avg_loss, 4),
        }

    def generate_hypotheses(self, positions: list[dict],
                            candles: Optional[dict[str, pd.DataFrame]] = None) -> list[dict]:
        """Generate trading strategy hypotheses from observed patterns.

        Returns:
            List of hypothesis dicts with: hypothesis, confidence,
            explanatory_power, supporting_evidence
        """
        if not positions:
            return []

        hypotheses = []

        # Analyze patterns
        entry_patterns = self.analyze_entry_patterns(positions, candles)
        sizing_patterns = self.analyze_sizing_patterns(positions)
        exit_patterns = self.analyze_exit_patterns(positions)

        # Hypothesis 1: Time-of-day bias
        tod = entry_patterns.get("time_of_day_bias", {})
        if tod:
            max_hour = max(tod, key=tod.get) if tod else None
            max_pct = tod.get(max_hour, 0) if max_hour is not None else 0
            if max_pct > 20:  # More than 20% of entries in one hour
                hypotheses.append({
                    "hypothesis": f"Trader shows strong time-of-day bias, entering {max_pct:.0f}% of positions at hour {max_hour} UTC",
                    "confidence": min(0.9, max_pct / 100 + 0.3),
                    "explanatory_power": max_pct / 100,
                    "supporting_evidence": f"Hour {max_hour} has {max_pct:.1f}% of all entries vs expected {100/24:.1f}%",
                })

        # Hypothesis 2: Fixed sizing
        if sizing_patterns.get("sizing_type") == "fixed":
            hypotheses.append({
                "hypothesis": "Trader uses fixed position sizing (low size variance)",
                "confidence": 0.8,
                "explanatory_power": 0.3,
                "supporting_evidence": f"Size std/mean ratio < 0.1, avg size = {sizing_patterns['avg_size']:.4f}",
            })

        # Hypothesis 3: Conviction-based sizing
        if sizing_patterns.get("sizing_type") == "conviction":
            hypotheses.append({
                "hypothesis": "Trader uses conviction-based sizing (high size variance suggests confidence weighting)",
                "confidence": 0.6,
                "explanatory_power": 0.4,
                "supporting_evidence": f"Size std = {sizing_patterns['size_std']:.4f}, avg = {sizing_patterns['avg_size']:.4f}",
            })

        # Hypothesis 4: Short-term trading
        avg_hold = exit_patterns.get("avg_hold_hours", 0)
        if 0 < avg_hold < 4:
            hypotheses.append({
                "hypothesis": "Trader is a short-term/scalp trader with sub-4h hold times",
                "confidence": 0.7,
                "explanatory_power": 0.5,
                "supporting_evidence": f"Average hold time: {avg_hold:.1f} hours",
            })
        elif avg_hold > 48:
            hypotheses.append({
                "hypothesis": "Trader is a swing/position trader with multi-day holds",
                "confidence": 0.7,
                "explanatory_power": 0.5,
                "supporting_evidence": f"Average hold time: {avg_hold:.1f} hours ({avg_hold/24:.1f} days)",
            })

        # Hypothesis 5: Fixed target exits
        if exit_patterns.get("exit_type") == "fixed_target":
            hypotheses.append({
                "hypothesis": f"Trader uses fixed profit targets (~{exit_patterns['avg_profit_target_pct']:.2f}%) and stop losses (~{exit_patterns['avg_stop_loss_pct']:.2f}%)",
                "confidence": 0.7,
                "explanatory_power": 0.6,
                "supporting_evidence": f"Symmetric profit/loss targets with low variance",
            })

        # Sort by confidence * explanatory_power
        hypotheses.sort(
            key=lambda h: h["confidence"] * h["explanatory_power"],
            reverse=True,
        )

        return hypotheses
