"""
phase2_6_mpc_go2_complete.py — Entry Point
==========================================
Paper: Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3
       Through Convex Model-Predictive Control", IROS 2018.

Running this file executes Phases 2–6 of the convex MPC trot controller
on the Unitree Go2 in MuJoCo.

Module layout (one responsibility per file):
─────────────────────────────────────────────────────────────────────────
  config.py            — All hardware constants, IDs, limits, debug config
  robot_params.py      — Go2Params (tuning) + RobotState (data container)
  state_estimator.py   — Phase 1: State estimation from MuJoCo ground truth
  dynamics.py          — Phase 2: Simplified rigid-body dynamics (Eq. 16–17)
  mpc_solver.py        — Phase 3: Condensed convex QP solver (Eq. 27–32)
  gait_scheduler.py    — Phase 4: Trot gait contact schedule
  swing_controller.py  — Phase 5: Swing leg PD + Raibert heuristic (Eq. 1, 33)
  torque_utils.py      — Phase 6: GRF→τ (Eq. 4), leg_to_ctrl, PDStand, Logger
  mpc_controller.py    — Top-level controller: ties all phases together
  simulation.py        — Simulation harness: verify() + run_with_viewer()
  phase2_6_mpc_go2_complete.py  ← YOU ARE HERE (entry point only)
─────────────────────────────────────────────────────────────────────────

Phase state machine inside Go2MPCController:

    Phase A (0 → 1.5 s)
        Joint-space PD settling.  Robot rises to nominal standing height.

    Phase B (1.5 → 4.5 s)
        All-stance convex MPC.  All four feet treated as in contact.
        Validates MPC can hold height before adding swing dynamics.

    Phase C (4.5 s → ∞,  only if pz ≥ 0.25 m)
        Trot MPC + swing controller.  Diagonal pairs alternate.
        Velocity ramps: 0 → 0.2 → 0.4 → 0.6 m/s.

Bug fixes implemented (11 bugs documented in original file header):
  • BUG 1/2: leg_to_ctrl actuator mapping verified
  • BUG 3:   all-stance Bc fallback when contacts = 0
  • BUG 4:   low-height forced contact detection
  • BUG 5:   all-stance schedule fallback in MPC horizon
  • BUG 6:   Phase C only activates above TROT_READY_HEIGHT
  • BUG 7:   TrotGait phase resets at activation time
  • BUG 8:   swing arc z measured above actual ground plane
  • BUG 9:   qfrc_bias indices verified correct
  • BUG 10:  Bc skew-cross sign confirmed correct
  • BUG 11:  pz_settled auto-detected from PD phase tail

To tweak tuning, edit Go2Params in robot_params.py.
To change XML path, edit XML_PATH in config.py.
"""

# ─── Standard library ────────────────────────────────────────────────────────
import sys
import os

# ─── Make sure we can import sibling modules regardless of working directory ──
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# ─── All phase modules (importing here makes them part of this run) ───────────
import config            # noqa: F401  — constants visible to all
import robot_params      # noqa: F401  — Go2Params, RobotState
import state_estimator   # noqa: F401  — Phase 1
import dynamics          # noqa: F401  — Phase 2
import mpc_solver        # noqa: F401  — Phase 3
import gait_scheduler    # noqa: F401  — Phase 4
import swing_controller  # noqa: F401  — Phase 5
import torque_utils      # noqa: F401  — Phase 6 utilities
import mpc_controller    # noqa: F401  — Main controller
import simulation        # noqa: F401  — Simulation harness

from config import XML_PATH
from simulation import Go2Simulation


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    sim = Go2Simulation(
        xml         = XML_PATH,
        K           = 10,        # horizon: 10 steps × 30 ms = 300 ms lookahead
        mpc_dt      = 0.030,     # MPC update rate: ~33 Hz
        gait_period = 0.40,      # 400 ms trot period (2.5 Hz)
        duty        = 0.50,      # 50% stance fraction
    )

    # ── Step 1: Headless verification (12 seconds) ────────────────────────────
    # Runs Phase A (PD) → Phase B (all-stance MPC) → Phase C (trot MPC).
    # Console prints a table: time, height, orientation, GRFs, contacts, phase.
    # CSV log written to mpc_debug_log.csv for post-run analysis.
    sim.verify(duration=12.0)

    # ── Step 2: Interactive viewer (25 seconds) ───────────────────────────────
    # Opens the MuJoCo passive viewer.
    # Ramps velocity 0 → 0.2 → 0.4 → 0.4 (turning) → 0.6 → 0.8 m/s.
    # Close the window or press Ctrl-C to stop.
    print("\nLaunching interactive viewer (Ctrl-C or close window to stop)…")
    try:
        sim.run_with_viewer(duration=25.0)
    except KeyboardInterrupt:
        sim.ctrl.logger.close()
        print("\nStopped by user.")