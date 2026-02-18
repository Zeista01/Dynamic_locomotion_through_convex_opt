"""
config.py — Hardware constants, MuJoCo IDs, joint slices, geometry, debug config
=================================================================================
Paper: Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3
       Through Convex Model-Predictive Control", IROS 2018.

All magic numbers live here. If the Go2 XML changes (body IDs shift,
actuator order changes, etc.) this is the ONLY file you need to edit.
"""

import numpy as np

# =============================================================================
# File paths
# =============================================================================
XML_PATH = "/home/stanny/unitree_ws/unitree_mujoco/unitree_robots/go2/scene.xml"

# =============================================================================
# Debug configuration
#   0 = silent
#   1 = warnings / errors only
#   2 = phase transitions + key events  (recommended for normal use)
#   3 = every MPC solve
#   4 = every control step (very verbose)
# =============================================================================
DEBUG_LEVEL = 2
LOG_CSV     = True
LOG_FILE    = "mpc_debug_log.csv"


def dbg(level: int, msg: str) -> None:
    """Conditional print gated by DEBUG_LEVEL."""
    if level <= DEBUG_LEVEL:
        print(f"[DBG{level}] {msg}")


# =============================================================================
# Leg ordering used throughout ALL modules
#   Internal leg index:  FL=0, FR=1, RL=2, RR=3
# =============================================================================
LEG_NAMES = ["FL", "FR", "RL", "RR"]
NUM_LEGS  = 4

# =============================================================================
# MuJoCo body IDs  (verified from diagnose_go2_model.py)
#   [1]=base_link, [5]=FL_foot, [9]=FR_foot, [13]=RL_foot, [17]=RR_foot
# =============================================================================
TRUNK_BODY_ID = 1
FOOT_BODY_IDS = {
    "FL": 5,
    "FR": 9,
    "RL": 13,
    "RR": 17,
}

# =============================================================================
# Foot geom IDs for contact detection
#   Named sphere geoms: FL=125, FR=137, RL=149, RR=161  (type=2)
#   Reverse map: geom_id → leg_index  (0=FL, 1=FR, 2=RL, 3=RR)
# =============================================================================
FOOT_GEOM_IDS = {"FL": 125, "FR": 137, "RL": 149, "RR": 161}
GEOM_TO_LEG   = {125: 0, 137: 1, 149: 2, 161: 3}

# =============================================================================
# qpos / qvel index slices
#   qpos[0:3]  = free-joint position  (x, y, z)
#   qpos[3:7]  = free-joint quaternion (w, x, y, z)
#   qpos[7:19] = leg joints in XML order: FL, FR, RL, RR × (hip, thigh, calf)
#
#   qvel[0:3]  = free-joint linear velocity  (world frame)
#   qvel[3:6]  = free-joint angular velocity (world frame)
#   qvel[6:18] = leg joint velocities in same XML order
# =============================================================================
JOINT_QPOS = {
    "FL": slice(7,  10),
    "FR": slice(10, 13),
    "RL": slice(13, 16),
    "RR": slice(16, 19),
}
JOINT_QVEL = {
    "FL": slice(6,  9),
    "FR": slice(9,  12),
    "RL": slice(12, 15),
    "RR": slice(15, 18),
}

# =============================================================================
# qvel column indices for mj_jac full nv-vector
#   body occupies dofs 0-5 (lin 0:3, ang 3:6), then legs 6:18
# =============================================================================
QVEL_COLS = {
    "FL": [6, 7, 8],
    "FR": [9, 10, 11],
    "RL": [12, 13, 14],
    "RR": [15, 16, 17],
}

# =============================================================================
# Actuator (ctrl) order in MuJoCo: FR(0-2), FL(3-5), RR(6-8), RL(9-11)
#   Note: this is DIFFERENT from our internal leg order [FL,FR,RL,RR].
#   The leg_to_ctrl() function in torque_utils.py handles the remapping.
# =============================================================================
CTRL_SLICE = {
    "FR": slice(0, 3),
    "FL": slice(3, 6),
    "RR": slice(6, 9),
    "RL": slice(9, 12),
}

# =============================================================================
# Hardware limits
# =============================================================================
TORQUE_LIMIT       = 33.5   # Nm per joint
# BUG FIX: raised from 1.0 N to 10.0 N.
# The debug log shows cf drops to exactly 0.0 on the first step after
# MPC activation (t=1.504) even though all feet are physically on the
# ground. This is a transient efc_force dropout when ctrl[] changes
# abruptly at the Phase A→B transition. At 1 N the contact state flips
# to NNNN, making Bc an all-zeros matrix for one step and destroying the
# first QP solve. 10 N is safely above MuJoCo solver noise while still
# detecting genuine lift-off (swing feet read ≈0 N during flight).
CONTACT_THR        = 10.0   # N  (was 1.0)

# =============================================================================
# Go2 nominal geometry
#   Hip ab/ad offset, link lengths [m]
#   Hip origins in body frame [FL, FR, RL, RR]  (x-fwd, y-left, z-up)
# =============================================================================
L_HIP   = 0.08085
L_THIGH = 0.213
L_CALF  = 0.213

HIP_ORIGINS = np.array([
    [ 0.1934,  0.1125, 0.0],   # FL
    [ 0.1934, -0.1125, 0.0],   # FR
    [-0.1934,  0.1125, 0.0],   # RL
    [-0.1934, -0.1125, 0.0],   # RR
])

# =============================================================================
# Height thresholds
# =============================================================================
NOMINAL_HEIGHT    = 0.32    # m  — target trot CoM height (below 4-leg stance ~0.338m)
COLLAPSE_HEIGHT   = 0.20    # m  — trigger PD recovery
TROT_READY_HEIGHT = 0.30    # m  — minimum height before activating trot