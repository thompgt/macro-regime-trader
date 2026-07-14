"""Tests for the RiskManager capital-preservation gate."""

from __future__ import annotations

import json

import pytest

from macro_regime_trader.config import get_settings
from macro_regime_trader.core.risk_manager import RiskManager
from macro_regime_trader.types import Regime, RiskDecision, Signal


def make_signal(target_exposure: float = 0.5) -> Signal:
    return Signal(
        timestamp="2024-01-01",
        regime=Regime.SUSTAINED_BULL,
        target_exposure=target_exposure,
        reason="test signal",
    )


def test_approves_normal_signal_with_no_drawdown(tmp_path):
    rm = RiskManager(base_dir=tmp_path)
    signal = make_signal(0.5)

    decision = rm.validate(signal, current_equity=100_000.0)

    assert decision == RiskDecision(
        approved=True, adjusted_exposure=0.5, reason="approved", locked=False
    )


def test_circuit_breaker_trips_halts_then_resumes(tmp_path):
    settings = get_settings()
    rm = RiskManager(settings=settings, base_dir=tmp_path)
    signal = make_signal(0.5)

    # Establish session baseline.
    first = rm.validate(signal, current_equity=100_000.0)
    assert first.approved is True

    # Drop equity more than circuit_breaker_drawdown_pct (2.5%) from session start.
    drawdown_equity = 100_000.0 * (1 - settings.circuit_breaker_drawdown_pct - 0.01)
    tripped = rm.validate(signal, current_equity=drawdown_equity)
    assert tripped.approved is False
    assert tripped.locked is False
    assert tripped.reason == "circuit_breaker_halt"

    # While halted, every call should be vetoed for circuit_breaker_halt_steps calls,
    # regardless of equity recovering.
    for _ in range(settings.circuit_breaker_halt_steps):
        decision = rm.validate(signal, current_equity=100_000.0)
        assert decision.approved is False
        assert decision.reason == "circuit_breaker_halt"
        assert decision.locked is False

    # Halt window has elapsed; healthy equity should resume approving.
    resumed = rm.validate(signal, current_equity=100_000.0)
    assert resumed.approved is True
    assert resumed.adjusted_exposure == 0.5
    assert resumed.reason == "approved"


def test_kill_switch_trips_and_locks_permanently(tmp_path):
    settings = get_settings()
    rm = RiskManager(settings=settings, base_dir=tmp_path)
    signal = make_signal(0.5)

    peak_equity = 100_000.0
    rm.validate(signal, current_equity=peak_equity)

    crash_equity = peak_equity * (1 - settings.kill_switch_drawdown_pct - 0.01)
    tripped = rm.validate(signal, current_equity=crash_equity)

    assert tripped.approved is False
    assert tripped.locked is True
    assert tripped.reason == "kill_switch_triggered"

    lock_file = tmp_path / settings.lock_file_path
    assert lock_file.exists()
    payload = json.loads(lock_file.read_text())
    assert payload["peak_equity"] == peak_equity
    assert payload["current_equity"] == crash_equity

    # Even if equity fully recovers, the lock must persist for all future calls.
    recovered = rm.validate(signal, current_equity=peak_equity * 2)
    assert recovered.approved is False
    assert recovered.locked is True
    assert recovered.adjusted_exposure == 0.0


def test_fresh_manager_honors_pre_existing_lock_file(tmp_path):
    settings = get_settings()
    lock_file = tmp_path / settings.lock_file_path
    lock_file.write_text(
        json.dumps({"reason": "kill_switch_triggered", "peak_equity": 100_000.0})
    )

    rm = RiskManager(settings=settings, base_dir=tmp_path, check_existing_lock=True)
    signal = make_signal(0.5)

    decision = rm.validate(signal, current_equity=100_000.0)

    assert decision.approved is False
    assert decision.locked is True
    assert decision.reason == "system_locked"


def test_manager_without_check_existing_lock_ignores_stray_file(tmp_path):
    settings = get_settings()
    lock_file = tmp_path / settings.lock_file_path
    lock_file.write_text(
        json.dumps({"reason": "kill_switch_triggered", "peak_equity": 100_000.0})
    )

    rm = RiskManager(settings=settings, base_dir=tmp_path)
    signal = make_signal(0.5)

    decision = rm.validate(signal, current_equity=100_000.0)

    assert decision.approved is True
    assert decision.locked is False
