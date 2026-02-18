"""
gait_scheduler.py — Phase 4: Trot Gait Scheduler
==================================================
Paper: Di Carlo et al., IROS 2018  (Section IV.B, Table I)

Implements the contact schedule for a diagonal trot gait:

    Diagonal pair 1 (FL + RR): stance for first half of each period
    Diagonal pair 2 (FR + RL): stance for second half of each period

Phase offsets [FL, FR, RL, RR]:
    offset = [0.0, 0.5, 0.5, 0.0]

    → At phase ∈ [0, duty):   FL and RR are in stance, FR and RL swing
    → At phase ∈ [duty, 1):  FR and RL are in stance, FL and RR swing

FIX (BUG 7): Gait phase must be measured relative to the activation time t₀,
not absolute simulation time.  activate(t) must be called when the trot first
starts.  Otherwise the gait phase jumps to a mid-cycle value and the first
swing step is incorrectly timed.

Usage::

    gait = TrotGait(period=0.40, duty=0.50)
    gait.activate(sim_time)                   # call once at Phase C start

    contacts = gait.contact_at(sim_time)      # (4,) bool for current instant
    schedule = gait.schedule(t0, dt, K)       # (K, 4) bool for MPC horizon
"""

import numpy as np
from config import NUM_LEGS, dbg


class TrotGait:
    """
    Trot gait contact scheduler.

    Parameters
    ----------
    period : float
        Full gait cycle period [s].  Default: 0.40 s (2.5 Hz)
    duty   : float
        Fraction of each period a foot is in stance.  Default: 0.50 (50%)
    """

    # Phase offsets for each leg [FL, FR, RL, RR]
    # 0.0 = starts in stance from the beginning of the cycle
    # 0.5 = starts in stance from the midpoint of the cycle
    _OFFSETS = np.array([0.0, 0.5, 0.5, 0.0])

    def __init__(self, period: float = 0.40, duty: float = 0.50):
        self.T    = period
        self.duty = duty
        self._t0  = 0.0   # simulation time when gait was activated

    # ── Activation ────────────────────────────────────────────────────────────

    def activate(self, t: float) -> None:
        """
        Record the activation time so that phase is measured relative to it.

        FIX (BUG 7): Without this, phase computation at t=4.5s with T=0.4s
        gives a different cycle position than starting fresh at t=0.
        Always call this when transitioning from Phase B → Phase C.
        """
        self._t0 = t
        dbg(2, f"TrotGait activated at t={t:.3f}s  "
               f"(T={self.T:.3f}s, duty={self.duty:.0%})")

    # ── Contact queries ───────────────────────────────────────────────────────

    def contact_at(self, t: float) -> np.ndarray:
        """
        Returns (4,) bool array: True = foot is in stance at time t.

        Phase for leg i:
            φᵢ(t) = ((t − t₀) / T + offset[i]) mod 1
        Foot is in stance when φᵢ < duty.
        """
        phase = (((t - self._t0) / self.T) + self._OFFSETS) % 1.0
        return phase < self.duty

    def schedule(self, t0: float, dt: float, K: int) -> np.ndarray:
        """
        Returns (K, 4) bool contact schedule for the MPC prediction horizon.

        t0 : start time of the horizon (current sim time)
        dt : MPC timestep [s]
        K  : horizon length (number of steps)
        """
        return np.array([
            self.contact_at(t0 + k * dt)
            for k in range(K)
        ])

    def phase_fraction(self, t: float, leg_idx: int) -> float:
        """
        Returns the normalised phase ∈ [0, 1) for a specific leg at time t.
        0 = start of stance,  duty = end of stance / start of swing.
        """
        return (((t - self._t0) / self.T) + self._OFFSETS[leg_idx]) % 1.0

    def is_stance(self, t: float, leg_idx: int) -> bool:
        """True if leg leg_idx is in stance at time t."""
        return self.phase_fraction(t, leg_idx) < self.duty

    # ── Timing helpers ────────────────────────────────────────────────────────

    @property
    def stance_time(self) -> float:
        """Duration of a single stance phase [s]."""
        return self.T * self.duty

    @property
    def swing_time(self) -> float:
        """Duration of a single swing phase [s]."""
        return self.T * (1.0 - self.duty)

    # ── Debug representation ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"TrotGait(T={self.T:.3f}s, duty={self.duty:.0%}, "
                f"t₀={self._t0:.3f}s, "
                f"stance={self.stance_time*1e3:.0f}ms, "
                f"swing={self.swing_time*1e3:.0f}ms)")