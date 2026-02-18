"""
swing_controller.py — Phase 5: Swing Leg Controller

Implements the reference leg_controller.py pattern:
  - Swing: task-space PD with Cartesian inertia decoupling (Lambda)
    tau = J.T @ (KP*(p_des-p_cur) + KD*(v_des-v_cur)) + (C*dq + g)[joints]
  - Stance: tau = J.T @ -contact_force  (ref uses negative sign directly)

The swing trajectory uses minimum-jerk profile (from reference gait.py):
  - Smooth position/velocity/acceleration at lift-off and touchdown
  - Height bump using b(s) = 64*s^3*(1-s)^3

Raibert landing heuristic (from reference gait.py):
  p_td = p_hip + cmd_v * pred_time + k_v * (v_actual - cmd_v)

BUG FIXES applied in this version
-----------------------------------
FIX 1 — Raibert touchdown overcorrection
  BEFORE: p_td[x] = hip_x + vel[0] * pred_time + k_v_x * vel[0]
          This applies (pred_time + k_v_x) * v_actual as one block.
          When vy_drift = -0.51 m/s and pred_time=0.20, k_v_x=0.08:
          correction = -0.51 * (0.20 + 0.08) = -0.143 m  ← 3.5× too large

  AFTER:  p_td[x] = hip_x + cmd_vx_world * pred_time
                           + k_v_x * (vel[0] - cmd_vx_world)
          Nominal drift follows *commanded* velocity; k_v term only damps
          the *error* between actual and commanded velocity.
          At vy_error = -0.51 m/s, k_v_y=0.08: correction = -0.041 m ✓

FIX 2 — Foot velocity uses leg-only Jacobian (ignores body motion)
  BEFORE: Ji = jacp[:, cols]          # 3×3 — leg DOFs only
          v_foot = Ji @ state.dq[i*3:i*3+3]
          Body translational / angular velocity not included → wrong damping
          when body is rolling or pitching.

  AFTER:  v_foot = jacp @ self.data.qvel[:nv]   # full 3×nv Jacobian
          Includes all 18 DOFs (6 body + 12 leg joints).
"""

import numpy as np
import mujoco

from config import (
    LEG_NAMES, NUM_LEGS, FOOT_BODY_IDS, QVEL_COLS,
    HIP_ORIGINS, TORQUE_LIMIT, dbg,
)
from robot_params import RobotState
from gait_scheduler import TrotGait


# Gains matching reference leg_controller.py
KP_SWING = np.diag([400., 400., 400.])
KD_SWING = np.diag([75., 75., 75.])

HEIGHT_SWING = 0.08   # 8cm clearance (reference uses 0.1m)


def _min_jerk(s):
    """Minimum-jerk basis functions and derivatives (reference gait.py)."""
    mj   = 10*s**3 - 15*s**4 + 6*s**5
    dmj  = 30*s**2 - 60*s**3 + 30*s**4
    d2mj = 60*s    - 180*s**2 + 120*s**3
    return mj, dmj, d2mj


def _height_bump(s, h_sw):
    """Smooth vertical bump: b(s)=64*s^3*(1-s)^3 (reference gait.py)."""
    b    = 64 * s**3 * (1 - s)**3
    db   = 192 * s**2 * (1 - s)**2 * (1 - 2*s)
    d2b  = 192 * (2*s*(1-s)**2*(1-2*s) - 2*s**2*(1-s)*(1-2*s) - 2*s**2*(1-s)**2)
    return h_sw * b, h_sw * db, h_sw * d2b


def make_swing_trajectory(p0, pf, t_swing, h_sw=HEIGHT_SWING):
    """
    Minimum-jerk swing trajectory evaluator (directly from reference gait.py).
    Returns callable: t → (pos, vel, acc) all in world frame.
    """
    p0 = np.asarray(p0, dtype=float)
    pf = np.asarray(pf, dtype=float)
    T  = float(t_swing)
    dp = pf - p0

    def eval_at(t):
        s = np.clip(t / T, 0.0, 1.0)
        mj, dmj, d2mj = _min_jerk(s)

        p = p0 + dp * mj
        v = dp * dmj / T
        a = dp * d2mj / (T**2)

        if h_sw != 0.0:
            bz, dbz, d2bz = _height_bump(s, h_sw)
            p[2] += bz
            v[2] += dbz / T
            a[2] += d2bz / (T**2)

        return p, v, a

    return eval_at


def raibert_touchdown(state: RobotState,
                      gait: TrotGait,
                      leg_idx: int,
                      cmd_vx_world: float = 0.0,
                      cmd_vy_world: float = 0.0) -> np.ndarray:
    """
    Raibert heuristic touchdown position.

    Correct formulation (from Di Carlo et al. 2018):
      p_td = hip_pos_world
           + cmd_v   * pred_time          ← nominal: foot follows command
           + k_v     * (v_actual - cmd_v) ← correction: damp velocity error

    Parameters
    ----------
    state        : current robot state
    gait         : TrotGait carrying swing_time and stance_time
    leg_idx      : 0=FL, 1=FR, 2=RL, 3=RR
    cmd_vx_world : desired x-velocity in WORLD frame [m/s]
    cmd_vy_world : desired y-velocity in WORLD frame [m/s]
    """
    t_swing  = gait.swing_time
    t_stance = gait.stance_time
    T        = t_swing + 0.5 * t_stance
    pred_time = T / 2.0

    # Hip position in world frame
    hip_offset_body = HIP_ORIGINS[leg_idx]
    hip_pos_world = state.pos + state.R @ hip_offset_body

    ground_z = 0.02   # slight offset (reference uses 0.02)

    # ── FIX 1: separate nominal drift from error correction ────────────────
    # k_v proportional to full gait period (reference values)
    k_v_x = 0.4 * T   # ≈ 0.08 m·s at T=0.20
    k_v_y = 0.2 * T   # ≈ 0.04 m·s at T=0.20

    # Velocity tracking errors in world frame
    vx_err = state.vel[0] - cmd_vx_world
    vy_err = state.vel[1] - cmd_vy_world

    p_td = np.array([
        hip_pos_world[0] + cmd_vx_world * pred_time + k_v_x * vx_err,
        hip_pos_world[1] + cmd_vy_world * pred_time + k_v_y * vy_err,
        ground_z
    ])
    # ──────────────────────────────────────────────────────────────────────

    return p_td


class SwingController:
    """
    Task-space swing controller matching the reference leg_controller.py.

    For swing legs: PD + feedforward in Cartesian space with inertia decoupling
    For stance legs: direct Jacobian transpose mapping (handled by torque_utils)
    """

    def __init__(self, gait: TrotGait,
                 model: mujoco.MjModel, data: mujoco.MjData):
        self.gait  = gait
        self.model = model
        self.data  = data

        self._swing_start_t  : dict = {}   # leg_idx → float
        self._swing_traj     : dict = {}   # leg_idx → callable
        self._ground_z_est   = 0.02

    def update_ground_estimate(self, state: RobotState) -> None:
        stance_z = [
            state.foot_pos[i][2]
            for i in range(NUM_LEGS) if state.contact[i]
        ]
        if stance_z:
            self._ground_z_est = float(np.mean(stance_z))

    def notify_lift(self, leg_idx: int, t: float,
                    foot_pos_world: np.ndarray,
                    state: RobotState = None,
                    cmd_vx_world: float = 0.0,
                    cmd_vy_world: float = 0.0) -> None:
        """
        Called when a stance foot transitions to swing.

        Parameters
        ----------
        leg_idx       : 0=FL, 1=FR, 2=RL, 3=RR
        t             : current simulation time [s]
        foot_pos_world: current foot position in world frame
        state         : full robot state (used for Raibert heuristic)
        cmd_vx_world  : desired x-velocity in WORLD frame [m/s]  ← NEW
        cmd_vy_world  : desired y-velocity in WORLD frame [m/s]  ← NEW
        """
        self._swing_start_t[leg_idx] = t

        if state is not None:
            # FIX 1 applied here — pass world-frame commands to Raibert
            p_td = raibert_touchdown(state, self.gait, leg_idx,
                                     cmd_vx_world, cmd_vy_world)
        else:
            p_td = foot_pos_world.copy()
            p_td[2] = self._ground_z_est

        self._swing_traj[leg_idx] = make_swing_trajectory(
            foot_pos_world, p_td, self.gait.swing_time, HEIGHT_SWING
        )
        dbg(2, f"Leg {LEG_NAMES[leg_idx]} LIFT-OFF t={t:.3f}s "
               f"foot={foot_pos_world.round(3)} td={p_td.round(3)}")

    def compute_torques(self, state: RobotState,
                        contacts: np.ndarray) -> np.ndarray:
        """
        Compute joint torques for all SWING legs.
        Matches reference leg_controller.py swing computation.
        """
        self.update_ground_estimate(state)

        tau  = np.zeros(12)
        nv   = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        t    = state.t

        for i, leg in enumerate(LEG_NAMES):
            if contacts[i]:
                continue   # stance — handled by grf_to_joint_torques

            bid  = FOOT_BODY_IDS[leg]
            cols = QVEL_COLS[leg]    # still used for bias extraction below

            # Current foot position
            p_cur = self.data.xpos[bid].copy()

            # Full (3 × nv) Jacobian — includes body DOFs 0:6 and all legs
            jacp[:] = 0.
            jacr[:] = 0.
            mujoco.mj_jac(self.model, self.data, jacp, jacr, p_cur, bid)

            # ── FIX 2: full foot velocity (body + leg DOFs) ───────────────
            # BEFORE: Ji = jacp[:, cols]; v_foot = Ji @ state.dq[i*3:i*3+3]
            #   → ignores body translational & angular velocity
            # AFTER:  use all 18 DOFs via the full (3×nv) Jacobian
            v_foot = jacp @ self.data.qvel[:nv]
            # ─────────────────────────────────────────────────────────────

            # 3×3 leg sub-Jacobian (still needed for tau mapping)
            Ji = jacp[:, cols]   # (3, 3)

            # Desired foot state from trajectory
            t_start = self._swing_start_t.get(i, t)
            traj    = self._swing_traj.get(i, None)

            if traj is not None:
                time_since_to = t - t_start
                p_des, v_des, a_des = traj(time_since_to)
            else:
                # No trajectory set yet — hold current position
                p_des = p_cur.copy()
                v_des = np.zeros(3)
                a_des = np.zeros(3)

            pos_error = p_des - p_cur
            vel_error = v_des - v_foot

            # Task-space PD force
            F_task = KP_SWING @ pos_error + KD_SWING @ vel_error

            # Map to joint torques + gravity/Coriolis compensation
            # qfrc_bias[cols] = (C*dq + g) for the 3 leg joints
            tau[i*3 : i*3+3] = Ji.T @ F_task + self.data.qfrc_bias[cols]

            dbg(4, f"Swing {leg}: |Δp|={np.linalg.norm(pos_error):.4f}m  "
                   f"|F|={np.linalg.norm(F_task):.1f}N  "
                   f"|vf|={np.linalg.norm(v_foot):.3f}m/s")

        return tau