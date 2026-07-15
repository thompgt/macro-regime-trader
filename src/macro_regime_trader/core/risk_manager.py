"""Strict capital preservation layer with absolute veto authority.

The RiskManager sits between the strategy layer and the broker layer. It is a
stateful, sequential validator: each call to :meth:`RiskManager.validate`
advances an internal state machine (peak equity, session-start equity, and a
circuit-breaker halt countdown) and must be called exactly once per
simulation step, in chronological order.

Precedence of checks, evaluated in this order on every call:

1. Permanent lock (kill switch already tripped, or a pre-existing
   ``TRADING_LOCKED.json`` found on disk when ``check_existing_lock=True``).
   Once locked, every subsequent decision is an automatic veto -- no other
   check runs.
2. Peak equity bookkeeping (updated before the kill-switch check so the
   drawdown comparison always uses the freshest peak).
3. Hard kill switch: peak-to-trough drawdown exceeding
   ``kill_switch_drawdown_pct`` permanently locks the system and persists
   ``TRADING_LOCKED.json``.
4. Intra-day circuit breaker halt in progress: countdown decremented, signal
   vetoed.
5. Intra-day circuit breaker trip: session drawdown exceeding
   ``circuit_breaker_drawdown_pct`` starts a new halt.
6. Otherwise the signal is approved unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from macro_regime_trader.config import Settings, get_settings
from macro_regime_trader.types import RiskDecision, Signal


class RiskManager:
    """Sequential, stateful capital-preservation gate for strategy signals.

    Parameters
    ----------
    settings:
        Configuration object. Defaults to ``get_settings()``.
    base_dir:
        Directory the lock file is read from / written to. Defaults to the
        current working directory. Pass ``tmp_path`` in tests to keep stray
        ``TRADING_LOCKED.json`` files out of the repo.
    check_existing_lock:
        If True, the constructor checks ``base_dir / settings.lock_file_path``
        for a pre-existing lock file and, if found, starts the manager in a
        locked state. Defaults to False so a fresh RiskManager in tests never
        picks up a lock file left behind by an unrelated prior run.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        base_dir: str | Path | None = None,
        check_existing_lock: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.lock_file_path = self.base_dir / self.settings.lock_file_path

        self._locked: bool = False
        self._peak_equity: float | None = None
        self._session_start_equity: float | None = None
        self._halt_steps_remaining: int = 0

        if check_existing_lock and self.lock_file_path.exists():
            self._locked = True

    def reset_session(self, current_equity: float) -> None:
        """Reset the session-start equity baseline used by the circuit breaker.

        Call this at the start of each new trading session/day in a
        multi-day backtest. Does NOT clear the kill-switch lock or the
        running peak-equity high-water mark -- those persist for the life of
        the RiskManager (and the kill switch is meant to be permanent).
        """
        self._session_start_equity = current_equity

    def validate(self, signal: Signal, current_equity: float) -> RiskDecision:
        """Validate one strategy signal against the current equity state.

        Must be called once per simulation step, in order, since it mutates
        internal state (peak equity, session baseline, halt countdown).
        """
        if self._locked:
            return RiskDecision(
                approved=False,
                adjusted_exposure=0.0,
                reason="system_locked",
                locked=True,
            )

        if self._session_start_equity is None:
            self._session_start_equity = current_equity

        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity

        kill_switch_drawdown = self._drawdown(self._peak_equity, current_equity)
        if kill_switch_drawdown > self.settings.kill_switch_drawdown_pct:
            self._trip_kill_switch(kill_switch_drawdown, current_equity)
            return RiskDecision(
                approved=False,
                adjusted_exposure=0.0,
                reason="kill_switch_triggered",
                locked=True,
            )

        if self._halt_steps_remaining > 0:
            self._halt_steps_remaining -= 1
            return RiskDecision(
                approved=False,
                adjusted_exposure=0.0,
                reason="circuit_breaker_halt",
                locked=False,
            )

        session_drawdown = self._drawdown(self._session_start_equity, current_equity)
        if session_drawdown > self.settings.circuit_breaker_drawdown_pct:
            self._halt_steps_remaining = self.settings.circuit_breaker_halt_steps
            return RiskDecision(
                approved=False,
                adjusted_exposure=0.0,
                reason="circuit_breaker_halt",
                locked=False,
            )

        return RiskDecision(
            approved=True,
            adjusted_exposure=signal.target_exposure,
            reason="approved",
        )

    @staticmethod
    def _drawdown(baseline: float, current: float) -> float:
        """Fractional decline of ``current`` below ``baseline``.

        Returns 0.0 if ``baseline <= 0`` or ``current >= baseline``.
        """
        if baseline <= 0:
            return 0.0
        return max(0.0, (baseline - current) / baseline)

    def _trip_kill_switch(self, drawdown_pct: float, current_equity: float) -> None:
        self._locked = True
        payload = {
            "reason": "kill_switch_triggered",
            "peak_equity": self._peak_equity,
            "current_equity": current_equity,
            "drawdown_pct": drawdown_pct,
            "kill_switch_drawdown_pct": self.settings.kill_switch_drawdown_pct,
        }
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file_path.write_text(json.dumps(payload, indent=2))
