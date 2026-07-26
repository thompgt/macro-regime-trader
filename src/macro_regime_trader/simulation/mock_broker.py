"""Stateful brokerage simulation.

Simulates a long-only trading account with:

- A virtual starting cash balance (``Settings.starting_balance``).
- Per-trade slippage (``Settings.slippage_pct``) applied against the trader.
- Dynamic, ratchet-only trailing stop-loss processing.
- Financing: interest charged on borrowed cash when approved exposure exceeds
  1x, and yield credited on idle cash, so leverage carries a real cost.

Exposure is bounded below at 0 (no shorting) but may exceed 1x; the strategy
layer caps it at ``Settings.max_leverage``.

Each call to :meth:`MockBroker.step` represents one bar of OHLCV data
(only close-level granularity is assumed, so ``price`` doubles as both the
bar's close and the price at which any stop-loss breach is checked/filled).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from macro_regime_trader.config import Settings, get_settings
from macro_regime_trader.types import Fill


@dataclass
class MockBroker:
    """A simple, stateful, long-only broker simulation.

    Order of operations within :meth:`step`:

    1. Check the active trailing stop (if any) against the bar's price.
       A breach forces full liquidation at this bar's price (net of
       slippage) *before* any exposure rebalancing is considered, and
       overrides whatever ``approved_exposure`` was passed in for this bar.
    2. If no stop was triggered, rebalance the position toward the target
       exposure (``approved_exposure * total_equity``), buying or selling
       the delta quantity at a slippage-adjusted price.
    3. If a new ``stop_price`` was supplied and the resulting exposure is
       still positive, ratchet the trailing stop up to
       ``max(existing_stop, stop_price)`` -- for a long-only engine the
       stop never loosens.
    """

    settings: Settings = field(default_factory=get_settings)

    def __post_init__(self) -> None:
        self._cash: float = self.settings.starting_balance
        self._position_qty: float = 0.0
        self._stop_price: float | None = None
        self._ledger: list[Fill] = []
        self._equity_curve: list[float] = []

    # -- read-only state ----------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def position_qty(self) -> float:
        return self._position_qty

    @property
    def stop_price(self) -> float | None:
        return self._stop_price

    def total_equity(self, price: float) -> float:
        return self._cash + self._position_qty * price

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    @property
    def ledger(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": f.timestamp,
                    "side": f.side,
                    "quantity": f.quantity,
                    "price": f.price,
                    "slippage_cost": f.slippage_cost,
                    "financing": f.financing,
                    "reason": f.reason,
                    "equity_after": equity,
                }
                for f, equity in zip(self._ledger, self._equity_curve, strict=True)
            ],
            columns=[
                "timestamp",
                "side",
                "quantity",
                "price",
                "slippage_cost",
                "financing",
                "reason",
                "equity_after",
            ],
        )

    # -- core simulation ------------------------------------------------

    def step(
        self,
        timestamp: object,
        price: float,
        approved_exposure: float,
        stop_price: float | None,
    ) -> Fill:
        """Advance the simulation by one bar and return the resulting Fill."""

        # 0. Accrue one bar of financing on the balance carried in from the
        #    previous bar, before any trading changes it.
        financing = self._accrue_financing()

        fill: Fill | None = None

        # 1. Trailing stop check takes priority over rebalancing.
        if self._stop_price is not None and self._position_qty > 0.0 and price <= self._stop_price:
            fill = self._liquidate(timestamp, price, reason="stop_loss")
            self._stop_price = None
        else:
            fill = self._rebalance(timestamp, price, approved_exposure)

            # 3. Ratchet the trailing stop up, only while still holding exposure.
            if stop_price is not None and stop_price != 0.0 and self._position_qty > 0.0:
                current = self._stop_price if self._stop_price is not None else float("-inf")
                self._stop_price = max(current, stop_price)

        fill = replace(fill, financing=financing)
        self._ledger.append(fill)
        self._equity_curve.append(self.total_equity(price))
        return fill

    # -- internals -------------------------------------------------------

    def _accrue_financing(self) -> float:
        """Charge interest on borrowed cash, credit yield on idle cash.

        Returns the (signed) amount applied: negative is a cost. Without this,
        leverage would be free and any exposure above 1x would look strictly
        better than it is.
        """
        days = self.settings.trading_days_per_year
        if days <= 0:
            return 0.0

        if self._cash < 0.0:
            amount = self._cash * self.settings.margin_interest_rate / days
        else:
            amount = self._cash * self.settings.cash_yield_rate / days

        self._cash += amount
        return amount

    def _liquidate(self, timestamp: object, price: float, reason: str = "") -> Fill:
        qty = self._position_qty
        fill_price = price * (1 - self.settings.slippage_pct)
        proceeds = qty * fill_price
        slippage_cost = qty * price * self.settings.slippage_pct
        self._cash += proceeds
        self._position_qty = 0.0
        return Fill(
            timestamp=timestamp,
            side="sell",
            quantity=qty,
            price=fill_price,
            slippage_cost=slippage_cost,
            reason=reason,
        )

    def _rebalance(self, timestamp: object, price: float, approved_exposure: float) -> Fill:
        equity = self.total_equity(price)
        current_value = self._position_qty * price
        target_value = approved_exposure * equity
        delta_value = target_value - current_value

        # No-trade band. Holding a constant *exposure fraction* would otherwise
        # require trading every single bar, since the position's value drifts
        # with price even when the target is unchanged -- which bleeds slippage
        # continuously for no change in stance. Only rebalance once actual
        # exposure has drifted from target by more than the band. Full exits are
        # always executed: standing down is never deferred for tidiness.
        band = self.settings.rebalance_band
        if (
            approved_exposure > 0.0
            and equity > 0.0
            and band > 0.0
            and abs(delta_value) / equity < band
        ):
            return Fill(
                timestamp=timestamp, side="hold", quantity=0.0, price=price, slippage_cost=0.0
            )

        if not price or abs(delta_value) / price < 1e-9:
            return Fill(
                timestamp=timestamp, side="hold", quantity=0.0, price=price, slippage_cost=0.0
            )

        if delta_value > 0:
            # Size the buy off the slippage-adjusted fill price so the dollar
            # cost equals delta_value exactly. When approved_exposure exceeds
            # 1.0 the resulting cash balance goes negative -- that is the
            # borrowed margin, and `_accrue_financing` charges interest on it.
            fill_price = price * (1 + self.settings.slippage_pct)
            delta_qty = delta_value / fill_price
            cost = delta_qty * fill_price
            self._cash -= cost
            self._position_qty += delta_qty
            slippage_cost = delta_qty * price * self.settings.slippage_pct
            return Fill(
                timestamp=timestamp,
                side="buy",
                quantity=delta_qty,
                price=fill_price,
                slippage_cost=slippage_cost,
            )
        else:
            # Sized off the raw mark price (not fill_price) so we never sell
            # more than the position actually holds at this valuation.
            sell_qty = -delta_value / price
            fill_price = price * (1 - self.settings.slippage_pct)
            proceeds = sell_qty * fill_price
            self._cash += proceeds
            self._position_qty -= sell_qty
            slippage_cost = sell_qty * price * self.settings.slippage_pct
            return Fill(
                timestamp=timestamp,
                side="sell",
                quantity=sell_qty,
                price=fill_price,
                slippage_cost=slippage_cost,
            )
