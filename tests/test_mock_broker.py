"""Tests for the stateful MockBroker simulation."""

from __future__ import annotations

import pandas as pd
import pytest

from macro_regime_trader.config import Settings
from macro_regime_trader.simulation.mock_broker import MockBroker


@pytest.fixture
def settings() -> Settings:
    return Settings(starting_balance=100_000.0, slippage_pct=0.0004)


def test_initial_state(settings: Settings) -> None:
    broker = MockBroker(settings=settings)
    assert broker.cash == settings.starting_balance
    assert broker.position_qty == 0.0
    assert broker.stop_price is None
    assert broker.equity_curve == []
    assert broker.ledger.empty


def test_buy_to_full_exposure_applies_slippage(settings: Settings) -> None:
    broker = MockBroker(settings=settings)
    price = 100.0
    fill = broker.step(
        timestamp=pd.Timestamp("2024-01-01"),
        price=price,
        approved_exposure=1.0,
        stop_price=None,
    )

    assert fill.side == "buy"
    assert fill.quantity > 0.0
    # Slippage on a buy makes the effective fill price worse (higher) than raw price.
    assert fill.price > price
    assert fill.slippage_cost > 0.0

    # Position should now represent (approximately) full exposure of starting equity.
    assert broker.position_qty == pytest.approx(fill.quantity)
    assert broker.cash < settings.starting_balance
    # Cash spent should roughly consume the whole starting balance.
    assert broker.cash == pytest.approx(settings.starting_balance - fill.quantity * fill.price)


def test_round_trip_buy_then_sell_costs_slippage(settings: Settings) -> None:
    broker = MockBroker(settings=settings)
    price = 100.0

    broker.step(
        timestamp=pd.Timestamp("2024-01-01"),
        price=price,
        approved_exposure=1.0,
        stop_price=None,
    )
    equity_after_buy = broker.total_equity(price)

    sell_fill = broker.step(
        timestamp=pd.Timestamp("2024-01-02"),
        price=price,
        approved_exposure=0.0,
        stop_price=None,
    )

    assert sell_fill.side == "sell"
    assert broker.position_qty == pytest.approx(0.0, abs=1e-9)
    # Effective sell price should be worse (lower) than raw price due to slippage.
    assert sell_fill.price < price

    # Round trip at an unchanged price should cost roughly the slippage, not be exact.
    assert broker.cash < equity_after_buy
    assert broker.cash == pytest.approx(equity_after_buy, rel=0.001)
    assert broker.cash == pytest.approx(
        settings.starting_balance * (1 - 2 * settings.slippage_pct), rel=0.01
    )


def test_trailing_stop_forces_liquidation_regardless_of_exposure(settings: Settings) -> None:
    broker = MockBroker(settings=settings)
    entry_price = 100.0

    # Enter a full position and set an initial trailing stop below entry.
    broker.step(
        timestamp=pd.Timestamp("2024-01-01"),
        price=entry_price,
        approved_exposure=1.0,
        stop_price=90.0,
    )
    assert broker.stop_price == 90.0
    assert broker.position_qty > 0.0

    # Next bar: price breaches the stop. Even though approved_exposure says
    # to stay fully invested, the stop should force a full liquidation.
    breach_price = 85.0
    fill = broker.step(
        timestamp=pd.Timestamp("2024-01-02"),
        price=breach_price,
        approved_exposure=1.0,
        stop_price=None,
    )

    assert fill.side == "sell"
    assert broker.position_qty == pytest.approx(0.0, abs=1e-9)
    assert broker.stop_price is None
    assert fill.price < breach_price  # slippage on the forced sell


def test_stop_ratchets_up_and_never_loosens(settings: Settings) -> None:
    broker = MockBroker(settings=settings)

    broker.step(
        timestamp=pd.Timestamp("2024-01-01"),
        price=100.0,
        approved_exposure=1.0,
        stop_price=90.0,
    )
    assert broker.stop_price == 90.0

    # Price rises, stop should ratchet up.
    broker.step(
        timestamp=pd.Timestamp("2024-01-02"),
        price=110.0,
        approved_exposure=1.0,
        stop_price=100.0,
    )
    assert broker.stop_price == 100.0

    # A lower proposed stop should not loosen the existing (higher) stop.
    broker.step(
        timestamp=pd.Timestamp("2024-01-03"),
        price=115.0,
        approved_exposure=1.0,
        stop_price=95.0,
    )
    assert broker.stop_price == 100.0


def test_hold_when_no_delta_needed(settings: Settings) -> None:
    broker = MockBroker(settings=settings)

    # Zero exposure requested against a flat (all-cash) book requires no trade
    # in either step, regardless of price movement.
    first = broker.step(
        timestamp=pd.Timestamp("2024-01-01"),
        price=100.0,
        approved_exposure=0.0,
        stop_price=None,
    )
    fill = broker.step(
        timestamp=pd.Timestamp("2024-01-02"),
        price=105.0,
        approved_exposure=0.0,
        stop_price=None,
    )

    assert first.side == "hold"
    assert fill.side == "hold"
    assert fill.quantity == 0.0
    assert fill.slippage_cost == 0.0


def test_ledger_and_equity_curve_grow_one_per_step(settings: Settings) -> None:
    broker = MockBroker(settings=settings)

    steps = [
        (pd.Timestamp("2024-01-01"), 100.0, 0.5, None),
        (pd.Timestamp("2024-01-02"), 102.0, 1.0, 90.0),
        (pd.Timestamp("2024-01-03"), 98.0, 1.0, 95.0),
    ]

    for i, (ts, price, exposure, stop) in enumerate(steps, start=1):
        broker.step(timestamp=ts, price=price, approved_exposure=exposure, stop_price=stop)
        assert len(broker.equity_curve) == i
        assert len(broker.ledger) == i

    ledger = broker.ledger
    assert list(ledger.columns) == [
        "timestamp",
        "side",
        "quantity",
        "price",
        "slippage_cost",
        "equity_after",
    ]
    assert len(ledger) == len(steps)
