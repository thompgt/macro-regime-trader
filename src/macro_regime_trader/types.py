"""Shared data contracts used across engine, strategy, risk, and broker layers.

Keeping these in one module lets independently-built modules interoperate
without importing each other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Regime(str, Enum):
    SUSTAINED_BULL = "sustained_bull"
    VOLATILE_DISTRIBUTION = "volatile_distribution"
    STRUCTURAL_BEAR = "structural_bear"
    COMPRESSED_LIQUIDITY = "compressed_liquidity"


@dataclass(frozen=True)
class Signal:
    """A strategy-generated intent, prior to risk validation."""

    timestamp: object  # pandas.Timestamp
    regime: Regime
    target_exposure: float  # fraction of equity, 0.0-1.0
    stop_price: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class RiskDecision:
    """RiskManager's verdict on a Signal. Only approved signals reach the broker."""

    approved: bool
    adjusted_exposure: float
    reason: str = ""
    locked: bool = False


@dataclass(frozen=True)
class Fill:
    """Result of an executed (or rejected) order at the broker layer."""

    timestamp: object
    side: str  # "buy" | "sell" | "hold"
    quantity: float
    price: float
    slippage_cost: float
