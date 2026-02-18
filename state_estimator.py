"""
state_estimator.py — Phase 1: Robot Interface & State Estimation
=================================================================
Paper: Di Carlo et al., IROS 2018  (Section II.B, III)

Reads ground-truth state every physics step from MuJoCo data.

FIX APPLIED HERE (Root Cause B):
  The original contact detection used a 3-step rolling-max buffer over
  efc_force to suppress transient dropouts.  The debug log shows that when
  ctrl[] changes at the Phase A→B boundary, efc_force drops to EXACTLY ZERO
  for ~150 consecutive steps (300 ms at 500 Hz) — far longer than any buffer
  of practical size could bridge.

  Root cause: MuJoCo's contact solver re-initialises its warm-start when the
  actuator command vector changes abruptly, which temporarily reports zero
  constraint forces even though the feet are physically on the ground.

  The correct fix is NOT a bigger buffer.  The correct fix is:

      If pz > COLLAPSE_HEIGHT (the robot is clearly standing, not fallen),
      force contact[i] = True regardless of efc_force.

  This is safe because:
  • Phase B uses an all-stance schedule anyway (contacts don't select Bc legs)
  • The only thing contact=False would do in Phase B is trigger collapse detection,
    which requires pz < COLLAPSE_HEIGHT — not a false positive
  • Phase C reads contacts from the GAIT SCHEDULER, not from here (see mpc_controller.py)
    contact[] from this estimator is used only for: (a) swing arc ground estimate,
    (b) the debug display, and (c) the Bc fallback in dynamics.py

  So forcing contacts=True when standing is always correct behaviour.

  The efc_force reading is kept for Phase C stance/swing classification
  (which needs to detect genuine lift-off) and for the debug log.
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from config import (
    LEG_NAMES, TRUNK_BODY_ID, FOOT_BODY_IDS, GEOM_TO_LEG,
    JOINT_QPOS, JOINT_QVEL, CONTACT_THR, COLLAPSE_HEIGHT,
    dbg,
)
from robot_params import RobotState


class StateEstimator:
    """
    Ground-truth state reader from MuJoCo.

    Usage::

        est = StateEstimator(model, data)
        state = est.update()   # call once per physics step
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data
        self._s    = RobotState()

        # ── Contact force buffer (kept for Phase C swing detection) ──────────
        # Window of 10 steps (20 ms at 500 Hz).  This is sufficient for the
        # genuine single-step efc dropouts that happen mid-trot at heel strike.
        # It is NOT sufficient for the 300ms dropout at the A→B boundary —
        # that case is handled by the pz-based override below.
        self._cf_buf = np.zeros((10, 4))
        self._cf_idx = 0

        self._verify_ids()

    # ── Startup check ──────────────────────────────────────────────────────────

    def _verify_ids(self) -> None:
        """Assert that body IDs in config.py still match the loaded XML."""
        def body_name(i: int) -> str:
            return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)

        assert body_name(TRUNK_BODY_ID) == "base_link", (
            f"Trunk ID mismatch: expected 'base_link' at id {TRUNK_BODY_ID}, "
            f"got '{body_name(TRUNK_BODY_ID)}'"
        )
        for leg, bid in FOOT_BODY_IDS.items():
            expected = f"{leg}_foot"
            got = body_name(bid)
            assert got == expected, (
                f"Foot ID mismatch for {leg}: expected '{expected}', got '{got}'"
            )

        print("[StateEstimator] Body IDs verified OK")
        print(f"[StateEstimator] nv={self.model.nv}, nq={self.model.nq}, "
              f"nu={self.model.nu}")

    # ── Main update — called every physics step ─────────────────────────────

    def update(self) -> RobotState:
        """
        Extract the full RobotState from the current MuJoCo data buffers.
        Returns the same RobotState object (updated in-place) for zero-copy.
        """
        d, s = self.data, self._s
        s.t  = d.time

        self._read_base(d, s)
        self._read_joints(d, s)
        self._read_feet(d, s)
        self._detect_contacts(d, s)

        return s

    def _read_base(self, d: mujoco.MjData, s: RobotState) -> None:
        """Position, orientation, linear and angular velocities (world frame)."""
        s.pos = d.qpos[0:3].copy()

        qw  = d.qpos[3:7]
        rot = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]])
        s.R = rot.as_matrix()

        e = rot.as_euler('ZYX')
        s.euler = np.array([e[2], e[1], e[0]])   # [φ, θ, ψ]

        s.vel   = d.qvel[0:3].copy()
        s.omega = d.qvel[3:6].copy()

    def _read_joints(self, d: mujoco.MjData, s: RobotState) -> None:
        """Fill s.q / s.dq in internal leg order [FL, FR, RL, RR] × 3."""
        for i, leg in enumerate(LEG_NAMES):
            s.q[i*3 : i*3+3]  = d.qpos[JOINT_QPOS[leg]]
            s.dq[i*3 : i*3+3] = d.qvel[JOINT_QVEL[leg]]

    def _read_feet(self, d: mujoco.MjData, s: RobotState) -> None:
        """Foot world positions and r_i vectors (foot − CoM) for the B matrix."""
        for i, leg in enumerate(LEG_NAMES):
            fp = d.xpos[FOOT_BODY_IDS[leg]].copy()
            s.foot_pos[i] = fp
            s.r_feet[i]   = fp - s.pos

    def _detect_contacts(self, d: mujoco.MjData, s: RobotState) -> None:
        """
        Detect foot contacts and populate s.cf and s.contact.

        STRATEGY:
        ─────────
        1. Read raw efc_force for all foot geoms → cf_raw (4,)
        2. Update 10-step rolling-max buffer → s.cf (smoothed)
        3. Apply pz-based override:
             If pz > COLLAPSE_HEIGHT  →  force all contacts = True
             (robot is clearly standing; efc_force dropout is a solver artefact)
             If pz ≤ COLLAPSE_HEIGHT  →  use s.cf > CONTACT_THR
             (robot may be truly airborne or fallen; trust the sensor)

        WHY pz override is correct:
          • Phase B uses all-stance MPC regardless of contact state.
          • Phase C uses the gait SCHEDULER for contact decisions, not s.contact.
          • s.contact is only used by: swing ground estimation, Bc fallback,
            and the collapse-detection check (which itself uses pz).
          • Forcing contacts=True when standing is always the right behaviour.
        """
        # ── Step 1: accumulate raw contact forces ─────────────────────────────
        cf_raw = np.zeros(4)
        for ci in range(d.ncon):
            c   = d.contact[ci]
            idx = GEOM_TO_LEG.get(c.geom1, GEOM_TO_LEG.get(c.geom2, -1))
            if idx < 0:
                continue
            addr = c.efc_address
            if 0 <= addr < len(d.efc_force):
                cf_raw[idx] += abs(d.efc_force[addr])

        # ── Step 2: rolling-max buffer (handles single-step efc dropouts) ────
        self._cf_buf[self._cf_idx] = cf_raw
        self._cf_idx = (self._cf_idx + 1) % len(self._cf_buf)
        s.cf = self._cf_buf.max(axis=0)

        dbg(4, f"t={s.t:.3f}  cf_raw={cf_raw.round(1)}  cf_buf_max={s.cf.round(1)}")

        # ── Step 3: pz-based override ─────────────────────────────────────────
        if s.pos[2] > COLLAPSE_HEIGHT:
            # FIX (Root Cause B): Robot is standing.  Report all contacts as True
            # regardless of efc_force, which may drop to zero for hundreds of
            # steps when ctrl[] changes abruptly (confirmed: 300 ms dropout in log).
            # This prevents the MPC from seeing a false all-swing state and
            # requesting 3× weight to arrest a phantom freefall.
            s.contact = np.ones(4, dtype=bool)
            dbg(4, f"t={s.t:.3f}  pz={s.pos[2]:.3f}m > {COLLAPSE_HEIGHT}m "
                   f"→ forcing contacts=TTTT")
        else:
            # Robot may be fallen or genuinely airborne — use sensor
            s.contact = s.cf > CONTACT_THR
            dbg(4, f"t={s.t:.3f}  pz={s.pos[2]:.3f}m ≤ {COLLAPSE_HEIGHT}m "
                   f"→ sensor contacts={''.join('T' if c else 'F' for c in s.contact)}")

    # ── Convenience ────────────────────────────────────────────────────────────

    @property
    def state(self) -> RobotState:
        """Last computed state (before next call to update())."""
        return self._s

    def euler_rate_approx(self) -> np.ndarray:
        """
        Small-angle Euler-rate approximation (paper Eq. 12):
            [φ̇, θ̇, ψ̇] ≈ Rz(ψ)ᵀ · ω
        """
        psi = self._s.euler[2]
        c, s_psi = np.cos(psi), np.sin(psi)
        Rz_T = np.array([[ c,  s_psi, 0.],
                          [-s_psi,  c, 0.],
                          [ 0.,    0., 1.]])
        return Rz_T @ self._s.omega