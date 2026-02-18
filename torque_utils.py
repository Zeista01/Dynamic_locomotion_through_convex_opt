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
) -> np.ndarray:
    """
    Build reference state trajectory Xref ∈ R^{13K}.
    State order: [φ, θ, ψ,  px, py, pz,  ωx, ωy, ωz,  vx, vy, vz,  −g]
    """
    vx, vy, wz = cmd
    Xref = np.zeros(13 * K)

    # Small integral correction
    pz_corrected = pz_des + np.clip(pz_error_integral * 0.1, -0.05, 0.05)

    for k in range(K):
        t = (k + 1) * dt
        r = np.zeros(13)

        r[0]  = 0.0                   # roll  → 0
        r[1]  = 0.0                   # pitch → 0
        r[2]  = x0[2] + wz * t       # yaw
        r[3]  = x0[3] + vx * t       # px
        r[4]  = x0[4] + vy * t       # py
        r[5]  = pz_corrected          # pz
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
    ]

    def __init__(self, filename: str = LOG_FILE, enabled: bool = True):
        self.enabled  = enabled
        self.filename = filename
        self._rows: list = []
        self._last_flush = 0

    def log(self, state: RobotState, grf: np.ndarray, ctrl: np.ndarray,
            phase: str, pz_des: float,
            trot_active: bool, mpc_active: bool) -> None:
        if not self.enabled:
            return

        r, p, y   = np.degrees(state.euler)
        total_fz  = float(grf[:, 2].sum())
        weight    = GO2.mass * GO2.g

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
            "contact_FL"  : int(state.contact[0]),
            "contact_FR"  : int(state.contact[1]),
            "contact_RL"  : int(state.contact[2]),
            "contact_RR"  : int(state.contact[3]),
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
