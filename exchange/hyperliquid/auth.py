"""Authentication helper for Hyperliquid using an Ethereum private key."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from eth_account import Account as _EthAccount  # type: ignore[import-untyped]
    _HAS_ETH_ACCOUNT = True
except ImportError:
    _HAS_ETH_ACCOUNT = False
    _EthAccount = None


class HyperliquidAuth:
    """Wraps an Ethereum private key for signing Hyperliquid requests.

    Parameters
    ----------
    private_key:
        Hex-encoded Ethereum private key (with or without ``0x`` prefix).
    account_address:
        Optional override for the wallet address.  When ``None`` the address
        is derived from *private_key*.
    """

    def __init__(self, private_key: str, account_address: Optional[str] = None):
        if not _HAS_ETH_ACCOUNT:
            raise ImportError(
                "eth_account is required for HyperliquidAuth. "
                "Install it with: pip install eth-account"
            )

        self._account = _EthAccount.from_key(private_key)
        self._account_address = account_address
        logger.info(
            "HyperliquidAuth initialised for address %s",
            self.address,
        )

    # -- public properties ---------------------------------------------------

    @property
    def wallet(self):
        """Return the underlying ``eth_account.Account`` object."""
        return self._account

    @property
    def address(self) -> str:
        """Return the effective wallet address (override or derived)."""
        if self._account_address is not None:
            return self._account_address
        return self._account.address
