"""Lightweight HTTP API for dashboard consumption."""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Try to import aiohttp — fall back to disabled state
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class DashboardAPI:
    """Lightweight HTTP API for dashboard consumption.

    Endpoints:
        GET /api/v1/status      - System overview
        GET /api/v1/strategies  - Strategy statuses + health
        GET /api/v1/portfolio   - Portfolio metrics
        GET /api/v1/positions   - Current positions
        GET /api/v1/decisions   - Recent OODA decisions
        GET /api/v1/regime      - Market regime
        GET /api/v1/health      - System health check
        GET /api/v1/wallets     - Top wallet intelligence

    Uses aiohttp for async serving. If aiohttp unavailable, falls back to disabled state.
    """

    def __init__(self, redis_store, timescale_store, strategy_manager=None,
                 config: dict = None):
        self.redis = redis_store
        self.timescale = timescale_store
        self.strategy_manager = strategy_manager
        self.config = config or {}

        self._app = None
        self._runner = None
        self._site = None

    async def start(self, port: int = 8080) -> None:
        """Start the HTTP server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available — dashboard API disabled")
            return

        self._app = web.Application()
        self._app.router.add_get("/api/v1/status", self._handle_status)
        self._app.router.add_get("/api/v1/strategies", self._handle_strategies)
        self._app.router.add_get("/api/v1/portfolio", self._handle_portfolio)
        self._app.router.add_get("/api/v1/positions", self._handle_positions)
        self._app.router.add_get("/api/v1/decisions", self._handle_decisions)
        self._app.router.add_get("/api/v1/regime", self._handle_regime)
        self._app.router.add_get("/api/v1/health", self._handle_health)
        self._app.router.add_get("/api/v1/wallets", self._handle_wallets)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", port)
        await self._site.start()
        logger.info("Dashboard API started on port %d", port)

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._runner:
            await self._runner.cleanup()
            logger.info("Dashboard API stopped")

    # ------------------------------------------------------------------
    # Route handlers (return aiohttp responses)
    # ------------------------------------------------------------------

    async def _handle_status(self, request):
        data = self.handle_status()
        return web.json_response(data)

    async def _handle_strategies(self, request):
        data = self.handle_strategies()
        return web.json_response(data)

    async def _handle_portfolio(self, request):
        data = self.handle_portfolio()
        return web.json_response(data)

    async def _handle_positions(self, request):
        data = self.handle_positions()
        return web.json_response(data)

    async def _handle_decisions(self, request):
        data = self.handle_decisions()
        return web.json_response(data)

    async def _handle_regime(self, request):
        data = self.handle_regime()
        return web.json_response(data)

    async def _handle_health(self, request):
        data = self.handle_health()
        return web.json_response(data)

    async def _handle_wallets(self, request):
        data = self.handle_wallets()
        return web.json_response(data)

    # ------------------------------------------------------------------
    # Data fetchers (testable without aiohttp)
    # ------------------------------------------------------------------

    def handle_status(self) -> dict:
        """System overview."""
        try:
            account = self.redis.get_account_state() or {}
        except Exception:
            account = {}

        active = []
        if self.strategy_manager:
            try:
                active = self.strategy_manager.get_active_strategies()
            except Exception:
                pass

        return {
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": account.get("total_equity", 0),
            "drawdown_pct": account.get("drawdown_pct", 0),
            "active_strategies": active,
        }

    def handle_strategies(self) -> dict:
        """Strategy statuses and health."""
        strategies = []
        if self.strategy_manager:
            try:
                for name in self.strategy_manager.get_active_strategies():
                    info = self.strategy_manager._strategies.get(name, {})
                    strategies.append({
                        "name": name,
                        "status": info.get("status", "unknown"),
                        "started_at": info.get("started_at", ""),
                    })
            except Exception:
                pass
        return {"strategies": strategies}

    def handle_portfolio(self) -> dict:
        """Portfolio metrics."""
        try:
            account = self.redis.get_account_state() or {}
        except Exception:
            account = {}

        return {
            "equity": account.get("total_equity", 0),
            "cash": account.get("cash", 0),
            "drawdown_pct": account.get("drawdown_pct", 0),
            "unrealized_pnl": account.get("unrealized_pnl", 0),
        }

    def handle_positions(self) -> dict:
        """Current positions."""
        positions = []
        try:
            raw = self.redis.get_all_positions() if hasattr(self.redis, "get_all_positions") else []
            for p in (raw or []):
                positions.append(p if isinstance(p, dict) else {"data": str(p)})
        except Exception:
            pass
        return {"positions": positions}

    def handle_decisions(self) -> dict:
        """Recent OODA decisions."""
        decisions = []
        try:
            raw = self.timescale.query_decisions_by_type("ooda_%", limit=20)
            for d in (raw or []):
                decisions.append(d if isinstance(d, dict) else {"data": str(d)})
        except Exception:
            pass
        return {"decisions": decisions}

    def handle_regime(self) -> dict:
        """Market regime."""
        try:
            symbol = self.config.get("symbol", "BTC/USDC")
            regime = self.redis.get_market_regime(symbol)
            if regime:
                return regime if isinstance(regime, dict) else {"regime": str(regime)}
        except Exception:
            pass
        return {"regime": "unknown", "confidence": 0}

    def handle_health(self) -> dict:
        """System health check."""
        redis_ok = False
        timescale_ok = False

        try:
            self.redis.get_account_state()
            redis_ok = True
        except Exception:
            pass

        try:
            if hasattr(self.timescale, "health_check"):
                self.timescale.health_check()
            timescale_ok = True
        except Exception:
            pass

        ok = redis_ok and timescale_ok
        return {
            "status": "ok" if ok else "degraded",
            "redis": "ok" if redis_ok else "error",
            "timescale": "ok" if timescale_ok else "error",
        }

    def handle_wallets(self) -> dict:
        """Top wallet intelligence."""
        wallets = []
        try:
            raw = self.timescale.query_wallet_scores(limit=10) if hasattr(self.timescale, "query_wallet_scores") else []
            for w in (raw or []):
                wallets.append(w if isinstance(w, dict) else {"data": str(w)})
        except Exception:
            pass
        return {"wallets": wallets}
