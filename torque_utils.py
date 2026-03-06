"""
torque_utils.py — GRF→torques, PD stand, reference, logger

KEY FIX: Stance torque sign.
Reference (leg_controller.py):
    tau_cmd = J_foot_world.T @ -contact_force

This is tau = -J^T * f_world, NOT +J^T * f_world.
The negative sign is required because:
- MPC gives reaction forces the GROUND exerts on the foot (upward = +z)
- To generate this ground reaction, the joint must push DOWN
- For the knee: J_z_knee < 0 at nominal stance → -J^T*f gives EXTENSION
"""

from __future__ import annotations

import os
import csv
from typing import TYPE_CHECKING

import numpy as np
import mujoco

from config import (
    LEG_NAMES, NUM_LEGS, FOOT_BODY_IDS, QVEL_COLS,
    CTRL_SLICE, TORQUE_LIMIT, DEBUG_LEVEL, LOG_FILE,
    dbg,
)
from robot_params import RobotState, GO2

if TYPE_CHECKING:
    pass


def grf_to_joint_torques(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    state: RobotState,
    grf_world: np.ndarray,
) -> np.ndarray:
    """
    Convert MPC ground reaction forces to joint torques.
    
    Reference implementation (leg_controller.py stance):
        tau = J.T @ -contact_force
    
    Note: qfrc_bias added to compensate for leg mass which the 
    single-rigid-body MPC ignores (~1.5kg/leg).
    """
    nv   = model.nv
    tau  = np.zeros(12)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    for i, leg in enumerate(LEG_NAMES):
        foot_bid = FOOT_BODY_IDS[leg]
        cols     = QVEL_COLS[leg]

        jacp[:] = 0.; jacr[:] = 0.
        mujoco.mj_jac(model, data, jacp, jacr,
                      data.xpos[foot_bid], foot_bid)
        Ji = jacp[:, cols]   # (3, 3)

        # Reference: tau = J.T @ -contact_force + (C*dq+g)[joints]
        # The negative is essential — see module docstring.
        tau[i*3 : i*3+3] = -Ji.T @ grf_world[i] + data.qfrc_bias[cols]

    return tau


def leg_to_ctrl(tau_leg: np.ndarray) -> np.ndarray:
    """
    Reorder torques from internal leg order to MuJoCo actuator order.
    Internal: [FL=0, FR=1, RL=2, RR=3] × 3
    MuJoCo:   [FR=0-2, FL=3-5, RR=6-8, RL=9-11]
    """
    out = np.zeros(12)
    out[0:3]  = tau_leg[3:6]    # FR
    out[3:6]  = tau_leg[0:3]    # FL
    out[6:9]  = tau_leg[9:12]   # RR
    out[9:12] = tau_leg[6:9]    # RL
    return out


def make_reference(
    x0: np.ndarray,
    cmd: np.ndarray,
    pz_des: float,
    dt: float,
    K: int,
    pz_error_integral: float = 0.0,
    pitch_ref: float = 0.0,
    terrain=None,
    nominal_clearance: float = 0.0,
) -> np.ndarray:
    """
    Build reference state trajectory Xref ∈ R^{13K}.
    State order: [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]

    pitch_ref : desired pitch angle [rad].  0 for flat ground;
                on stairs set to arctan(Δz/hip_span) so the MPC
                doesn't waste authority fighting the natural incline.
    terrain   : TerrainEstimator or None.  When provided (non-flat mode),
                pitch_ref and pz_des are computed per horizon step using
                the projected body x-position.  This lets the MPC
                anticipate upcoming terrain changes instead of reacting
                after the fact.
    nominal_clearance : CoM height above terrain (used with terrain).
    """
    vx, vy, wz = cmd
    Xref = np.zeros(13 * K)

    # Small integral correction
    pz_corrected = pz_des + np.clip(pz_error_integral * 0.1, -0.05, 0.05)

    # FIX 9: per-step terrain-aware reference.
    # Use delta formulation: step 0 uses the controller's smoothed
    # pitch_ref and pz_des, then future steps add the terrain CHANGE
    # from the current position.  This ensures consistency between the
    # IIR-smoothed current reference and the projected future values.
    use_terrain = (terrain is not None and not terrain.is_flat()
                   and nominal_clearance > 0)
    if use_terrain:
        pz_base    = terrain.target_pz(x0[3], nominal_clearance)
        pitch_base = terrain.pitch_ref(x0[3])

    for k in range(K):
        t = (k + 1) * dt
        r = np.zeros(13)

        proj_x = x0[3] + vx * t

        if use_terrain:
            # Delta from current position — preserves IIR smoothing at k=0,
            # while letting future steps anticipate terrain changes.
            delta_pitch = terrain.pitch_ref(proj_x) - pitch_base
            r[1] = pitch_ref + delta_pitch  # pitch_ref is already smoothed
            delta_pz = terrain.target_pz(proj_x, nominal_clearance) - pz_base
            r[5] = pz_corrected + delta_pz
        else:
            r[1] = pitch_ref
            r[5] = pz_corrected

        r[0]  = 0.0                   # roll  → 0
        r[2]  = x0[2] + wz * t       # yaw
        r[3]  = proj_x               # px
        r[4]  = x0[4] + vy * t       # py
        r[6]  = 0.0                   # ωx
        r[7]  = 0.0                   # ωy
        r[8]  = wz                    # ωz
        r[9]  = vx                    # vx
        r[10] = vy                    # vy
        r[11] = 0.0                   # vz
        r[12] = -9.81                 # −g

        Xref[k*13 : (k+1)*13] = r

    return Xref


class PDStand:
    """High-gain joint-space PD controller for Phase A and recovery."""

    Q_DES = np.array([0.0, 0.7, -1.4])   # nominal standing angles [rad]

    KP = np.array([80., 80., 80.])
    KD = np.array([ 8.,  8.,  8.])

    def compute(self, state: RobotState) -> np.ndarray:
        ctrl = np.zeros(12)
        leg_order = [("FR", 1), ("FL", 0), ("RR", 3), ("RL", 2)]
        for leg_name, leg_idx in leg_order:
            sl  = CTRL_SLICE[leg_name]
            q   = state.q[leg_idx*3 : leg_idx*3+3]
            dq  = state.dq[leg_idx*3 : leg_idx*3+3]
            ctrl[sl] = np.clip(
                self.KP * (self.Q_DES - q) - self.KD * dq,
                -TORQUE_LIMIT, TORQUE_LIMIT,
            )
        return ctrl


class DebugLogger:
    FIELDS = [
        "t", "phase", "pz", "roll_deg", "pitch_deg", "yaw_deg",
        "vx", "vy", "vz", "wx", "wy",
        "contact_FL", "contact_FR", "contact_RL", "contact_RR",
        "cf_FL", "cf_FR", "cf_RL", "cf_RR",
        "fz_FL", "fz_FR", "fz_RL", "fz_RR",
        "fx_FL", "fx_FR", "fx_RL", "fx_RR",
        "ctrl_max", "ctrl_min",
        "total_fz", "weight_ratio",
        "pz_error", "trot_active", "mpc_active",
        # Foot positions (world frame) — added for stair debugging
        "foot_x_FL", "foot_y_FL", "foot_z_FL",
        "foot_x_FR", "foot_y_FR", "foot_z_FR",
        "foot_x_RL", "foot_y_RL", "foot_z_RL",
        "foot_x_RR", "foot_y_RR", "foot_z_RR",
    ]

    def __init__(self, filename: str = LOG_FILE, enabled: bool = True):
        self.enabled  = enabled
        self.filename = filename
        self._rows: list = []
        self._last_flush = 0

    def log(self, state: RobotState, grf: np.ndarray, ctrl: np.ndarray,
            phase: str, pz_des: float,
            trot_active: bool, mpc_active: bool,
            contacts_sched: "np.ndarray | None" = None) -> None:
        if not self.enabled:
            return

        r, p, y   = np.degrees(state.euler)
        total_fz  = float(grf[:, 2].sum())
        weight    = GO2.mass * GO2.g

        # Use gait-scheduled contacts when available (Phase C).
        # state.contact is forced all-True by the pz-override in StateEstimator
        # (a stability fix), so it cannot be used to show the gait pattern.
        if contacts_sched is None:
            contacts_sched = state.contact

        row = {
            "t"           : round(state.t, 4),
            "phase"       : phase,
            "pz"          : round(state.pos[2], 4),
            "roll_deg"    : round(r, 2),
            "pitch_deg"   : round(p, 2),
            "yaw_deg"     : round(y, 2),
            "vx"          : round(state.vel[0], 4),
            "vy"          : round(state.vel[1], 4),
            "vz"          : round(state.vel[2], 4),
            "wx"          : round(state.omega[0], 4),
            "wy"          : round(state.omega[1], 4),
            "contact_FL"  : int(contacts_sched[0]),
            "contact_FR"  : int(contacts_sched[1]),
            "contact_RL"  : int(contacts_sched[2]),
            "contact_RR"  : int(contacts_sched[3]),
            "cf_FL"       : round(state.cf[0], 2),
            "cf_FR"       : round(state.cf[1], 2),
            "cf_RL"       : round(state.cf[2], 2),
            "cf_RR"       : round(state.cf[3], 2),
            "fz_FL"       : round(grf[0, 2], 2),
            "fz_FR"       : round(grf[1, 2], 2),
            "fz_RL"       : round(grf[2, 2], 2),
            "fz_RR"       : round(grf[3, 2], 2),
            "fx_FL"       : round(grf[0, 0], 2),
            "fx_FR"       : round(grf[1, 0], 2),
            "fx_RL"       : round(grf[2, 0], 2),
            "fx_RR"       : round(grf[3, 0], 2),
            "ctrl_max"    : round(float(np.max(np.abs(ctrl))), 2),
            "ctrl_min"    : round(float(np.min(ctrl)), 2),
            "total_fz"    : round(total_fz, 2),
            "weight_ratio": round(total_fz / weight, 3),
            "pz_error"    : round(pz_des - state.pos[2], 4),
            "trot_active" : int(trot_active),
            "mpc_active"  : int(mpc_active),
            # Foot world-frame positions — 4 legs × (x, y, z)
            # state.foot_pos shape: (4, 3)
            "foot_x_FL"   : round(float(state.foot_pos[0, 0]), 4),
            "foot_y_FL"   : round(float(state.foot_pos[0, 1]), 4),
            "foot_z_FL"   : round(float(state.foot_pos[0, 2]), 4),
            "foot_x_FR"   : round(float(state.foot_pos[1, 0]), 4),
            "foot_y_FR"   : round(float(state.foot_pos[1, 1]), 4),
            "foot_z_FR"   : round(float(state.foot_pos[1, 2]), 4),
            "foot_x_RL"   : round(float(state.foot_pos[2, 0]), 4),
            "foot_y_RL"   : round(float(state.foot_pos[2, 1]), 4),
            "foot_z_RL"   : round(float(state.foot_pos[2, 2]), 4),
            "foot_x_RR"   : round(float(state.foot_pos[3, 0]), 4),
            "foot_y_RR"   : round(float(state.foot_pos[3, 1]), 4),
            "foot_z_RR"   : round(float(state.foot_pos[3, 2]), 4),
        }
        self._rows.append(row)

        if len(self._rows) - self._last_flush >= 200:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        mode = "a" if os.path.exists(self.filename) else "w"
        with open(self.filename, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if mode == "w":
                writer.writeheader()
            writer.writerows(self._rows[self._last_flush:])
        self._last_flush = len(self._rows)

    def close(self) -> None:
        self._flush()
        if self.enabled and os.path.exists(self.filename):
            print(f"[DebugLogger] Saved {len(self._rows)} rows → {self.filename}")