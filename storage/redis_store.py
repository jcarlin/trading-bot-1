"""Redis hot-state store for real-time trading data."""

import json
import logging
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class RedisStore:
    """Fast in-memory store for order books, prices, positions, and system state.

    Key patterns
    ------------
    ob:{symbol}                 — latest order book snapshot (Hash)
    price:{symbol}              — latest price (Hash)
    position:{symbol}           — current position (Hash)
    account:state               — portfolio-level state (Hash)
    strategy:{name}:state       — per-strategy runtime state (Hash)
    funding:{symbol}            — latest funding rate (Hash)
    orders:open:{symbol}        — set of JSON-encoded open orders (Set)
    circuit_breaker:state       — circuit breaker state (Hash)
    data_quality:{source}:{sym} — data quality metrics with TTL (Hash)
    """

    def __init__(self, config: dict):
        self._r = redis.Redis(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("db", 0),
            password=config.get("password"),
            decode_responses=True,
        )
        logger.info("RedisStore connected to %s:%s",
                     config.get("host"), config.get("port"))

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    def set_orderbook(self, symbol: str, bids: list, asks: list,
                      mid_price: float, spread: float,
                      timestamp: str) -> None:
        key = f"ob:{symbol}"
        self._r.hset(key, mapping={
            "bids": json.dumps(bids),
            "asks": json.dumps(asks),
            "mid_price": str(mid_price),
            "spread": str(spread),
            "timestamp": timestamp,
        })

    def get_orderbook(self, symbol: str) -> Optional[dict]:
        key = f"ob:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "bids": json.loads(data["bids"]),
            "asks": json.loads(data["asks"]),
            "mid_price": float(data["mid_price"]),
            "spread": float(data["spread"]),
            "timestamp": data["timestamp"],
        }

    # ------------------------------------------------------------------
    # Price
    # ------------------------------------------------------------------

    def set_price(self, symbol: str, last: float, bid: float,
                  ask: float, timestamp: str) -> None:
        key = f"price:{symbol}"
        self._r.hset(key, mapping={
            "last": str(last),
            "bid": str(bid),
            "ask": str(ask),
            "timestamp": timestamp,
        })

    def get_price(self, symbol: str) -> Optional[dict]:
        key = f"price:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "last": float(data["last"]),
            "bid": float(data["bid"]),
            "ask": float(data["ask"]),
            "timestamp": data["timestamp"],
        }

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def set_position(self, symbol: str, entry_price: float, quantity: float,
                     side: str, entry_time: str,
                     unrealized_pnl: float = 0.0) -> None:
        key = f"position:{symbol}"
        self._r.hset(key, mapping={
            "entry_price": str(entry_price),
            "quantity": str(quantity),
            "side": side,
            "entry_time": entry_time,
            "unrealized_pnl": str(unrealized_pnl),
        })

    def get_position(self, symbol: str) -> Optional[dict]:
        key = f"position:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "entry_price": float(data["entry_price"]),
            "quantity": float(data["quantity"]),
            "side": data["side"],
            "entry_time": data["entry_time"],
            "unrealized_pnl": float(data["unrealized_pnl"]),
        }

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def set_account_state(self, total_equity: float, cash: float,
                          position_value: float, unrealized_pnl: float,
                          drawdown_pct: float, timestamp: str) -> None:
        self._r.hset("account:state", mapping={
            "total_equity": str(total_equity),
            "cash": str(cash),
            "position_value": str(position_value),
            "unrealized_pnl": str(unrealized_pnl),
            "drawdown_pct": str(drawdown_pct),
            "timestamp": timestamp,
        })

    def get_account_state(self) -> Optional[dict]:
        data = self._r.hgetall("account:state")
        if not data:
            return None
        return {
            "total_equity": float(data["total_equity"]),
            "cash": float(data["cash"]),
            "position_value": float(data["position_value"]),
            "unrealized_pnl": float(data["unrealized_pnl"]),
            "drawdown_pct": float(data["drawdown_pct"]),
            "timestamp": data["timestamp"],
        }

    # ------------------------------------------------------------------
    # Strategy state
    # ------------------------------------------------------------------

    def set_strategy_state(self, name: str, status: str,
                           last_signal: str,
                           last_signal_time: str) -> None:
        key = f"strategy:{name}:state"
        self._r.hset(key, mapping={
            "status": status,
            "last_signal": last_signal,
            "last_signal_time": last_signal_time,
        })

    def get_strategy_state(self, name: str) -> Optional[dict]:
        key = f"strategy:{name}:state"
        data = self._r.hgetall(key)
        return dict(data) if data else None

    # ------------------------------------------------------------------
    # Funding rate
    # ------------------------------------------------------------------

    def set_funding(self, symbol: str, rate: float, premium: float,
                    mark_price: float, timestamp: str) -> None:
        key = f"funding:{symbol}"
        self._r.hset(key, mapping={
            "rate": str(rate),
            "premium": str(premium),
            "mark_price": str(mark_price),
            "timestamp": timestamp,
        })

    def get_funding(self, symbol: str) -> Optional[dict]:
        key = f"funding:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "rate": float(data["rate"]),
            "premium": float(data["premium"]),
            "mark_price": float(data["mark_price"]),
            "timestamp": data["timestamp"],
        }

    # ------------------------------------------------------------------
    # Open orders (Set of JSON blobs)
    # ------------------------------------------------------------------

    def add_open_order(self, symbol: str, order: dict) -> None:
        key = f"orders:open:{symbol}"
        self._r.sadd(key, json.dumps(order))

    def remove_open_order(self, symbol: str, order: dict) -> None:
        key = f"orders:open:{symbol}"
        self._r.srem(key, json.dumps(order))

    def get_open_orders(self, symbol: str) -> list[dict]:
        key = f"orders:open:{symbol}"
        raw = self._r.smembers(key)
        return [json.loads(item) for item in raw]

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def set_circuit_breaker(self, active: bool, reason: str,
                            activated_at: str) -> None:
        self._r.hset("circuit_breaker:state", mapping={
            "active": str(active).lower(),
            "reason": reason,
            "activated_at": activated_at,
        })

    def get_circuit_breaker(self) -> Optional[dict]:
        data = self._r.hgetall("circuit_breaker:state")
        if not data:
            return None
        return {
            "active": data["active"] == "true",
            "reason": data["reason"],
            "activated_at": data["activated_at"],
        }

    # ------------------------------------------------------------------
    # Data quality (with TTL)
    # ------------------------------------------------------------------

    def set_data_quality(self, source: str, symbol: str,
                         last_update: str, latency_ms: float,
                         status: str, ttl: int = 300) -> None:
        key = f"data_quality:{source}:{symbol}"
        self._r.hset(key, mapping={
            "last_update": last_update,
            "latency_ms": str(latency_ms),
            "status": status,
        })
        self._r.expire(key, ttl)

    def get_data_quality(self, source: str, symbol: str) -> Optional[dict]:
        key = f"data_quality:{source}:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "last_update": data["last_update"],
            "latency_ms": float(data["latency_ms"]),
            "status": data["status"],
        }

    # ------------------------------------------------------------------
    # Market regime (hot state)
    # ------------------------------------------------------------------

    def set_market_regime(self, symbol: str, regime: str,
                          confidence: float, timestamp: str) -> None:
        key = f"regime:{symbol}"
        self._r.hset(key, mapping={
            "regime": regime,
            "confidence": str(confidence),
            "timestamp": timestamp,
        })

    def get_market_regime(self, symbol: str) -> Optional[dict]:
        key = f"regime:{symbol}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "regime": data["regime"],
            "confidence": float(data["confidence"]),
            "timestamp": data["timestamp"],
        }

    # ------------------------------------------------------------------
    # Health score (hot state)
    # ------------------------------------------------------------------

    def set_health_score(self, strategy_name: str, score: float,
                         grade: str, timestamp: str) -> None:
        key = f"health:{strategy_name}"
        self._r.hset(key, mapping={
            "score": str(score),
            "grade": grade,
            "timestamp": timestamp,
        })

    def get_health_score(self, strategy_name: str) -> Optional[dict]:
        key = f"health:{strategy_name}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "score": float(data["score"]),
            "grade": data["grade"],
            "timestamp": data["timestamp"],
        }


    # ------------------------------------------------------------------
    # Portfolio metrics (hot state)
    # ------------------------------------------------------------------

    def set_portfolio_metrics(self, metrics: dict) -> None:
        """Store portfolio-level metrics in Redis."""
        self._r.hset("portfolio:metrics", mapping={
            k: str(v) if not isinstance(v, str) else v
            for k, v in metrics.items()
            if not isinstance(v, dict)  # Skip nested dicts
        })

    def get_portfolio_metrics(self) -> Optional[dict]:
        """Retrieve portfolio metrics from Redis."""
        data = self._r.hgetall("portfolio:metrics")
        if not data:
            return None
        result = {}
        for k, v in data.items():
            try:
                result[k] = float(v)
            except (ValueError, TypeError):
                result[k] = v
        return result

    # ------------------------------------------------------------------
    # Correlation matrix (hot state)
    # ------------------------------------------------------------------

    def set_correlation_matrix(self, matrix_dict: dict) -> None:
        """Store correlation matrix in Redis as JSON."""
        self._r.set("portfolio:correlation", json.dumps(matrix_dict))

    def get_correlation_matrix(self) -> Optional[dict]:
        """Retrieve correlation matrix from Redis."""
        data = self._r.get("portfolio:correlation")
        if not data:
            return None
        return json.loads(data)

    # ------------------------------------------------------------------
    # Wallet intelligence (hot state)
    # ------------------------------------------------------------------

    def set_wallet_score(self, address: str, score: float,
                         grade: str, timestamp: str) -> None:
        """Store a wallet score in Redis."""
        key = f"wallet:{address}"
        self._r.hset(key, mapping={
            "score": str(score),
            "grade": grade,
            "timestamp": timestamp,
        })

    def get_wallet_score(self, address: str) -> Optional[dict]:
        """Retrieve a wallet score from Redis."""
        key = f"wallet:{address}"
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "score": float(data["score"]),
            "grade": data["grade"],
            "timestamp": data["timestamp"],
        }

    def set_top_wallets(self, wallets: list[dict]) -> None:
        """Store top wallet list in Redis."""
        self._r.set("wallets:top", json.dumps(wallets))

    def get_top_wallets(self) -> list[dict]:
        """Retrieve top wallets from Redis."""
        data = self._r.get("wallets:top")
        if not data:
            return []
        return json.loads(data)

    # ------------------------------------------------------------------
    # Allocation weights (Phase 4)
    # ------------------------------------------------------------------

    def set_allocation_weight(self, strategy_name: str, weight: float) -> None:
        """Store allocation weight for a strategy."""
        self._r.set(f"strategy:{strategy_name}:allocation", str(weight))

    def get_allocation_weight(self, strategy_name: str) -> Optional[float]:
        """Retrieve allocation weight for a strategy."""
        raw = self._r.get(f"strategy:{strategy_name}:allocation")
        if raw is None:
            return None
        return float(raw)

    # ------------------------------------------------------------------
    # Shadow runner state (Phase 4)
    # ------------------------------------------------------------------

    def set_shadow_state(self, shadow_name: str, state: dict) -> None:
        """Store shadow runner state."""
        self._r.set(f"shadow:{shadow_name}:state", json.dumps(state))

    def get_shadow_state(self, shadow_name: str) -> Optional[dict]:
        """Retrieve shadow runner state."""
        raw = self._r.get(f"shadow:{shadow_name}:state")
        if raw is None:
            return None
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Walk-forward status (Phase 5)
    # ------------------------------------------------------------------

    def set_walk_forward_status(self, strategy_name: str,
                                status_dict: dict) -> None:
        """Store walk-forward optimization status for a strategy."""
        key = f"walk_forward:{strategy_name}"
        self._r.set(key, json.dumps(status_dict))

    def get_walk_forward_status(self, strategy_name: str) -> Optional[dict]:
        """Retrieve walk-forward optimization status."""
        key = f"walk_forward:{strategy_name}"
        raw = self._r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Volatility regime (Phase 5)
    # ------------------------------------------------------------------

    def set_vol_regime(self, symbol: str, regime: str) -> None:
        """Store volatility regime classification for a symbol."""
        key = f"vol_regime:{symbol}"
        self._r.set(key, regime)

    def get_vol_regime(self, symbol: str) -> Optional[str]:
        """Retrieve volatility regime for a symbol."""
        key = f"vol_regime:{symbol}"
        return self._r.get(key)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            return self._r.ping()
        except redis.ConnectionError:
            return False
