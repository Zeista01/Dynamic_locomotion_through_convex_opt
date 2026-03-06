"""
terrain_estimator.py — Terrain Height Estimation & Stair Geometry
=================================================================

Supports the up-down stair scene (scene_stairs_uniform.xml):
  5 ascending steps  (6 cm riser / 28 cm tread, x = 2.00 → 3.40 m)
  Top platform       (1.0 m, x = 3.40 → 4.40 m, z = 0.30 m)
  5 descending steps (mirror, x = 4.40 → 5.80 m)
  Flat exit zone     (x > 5.80 m)

FIXES (previous session):
  pitch_ref():  return 0 when robot_x < STAIR_START_X to prevent +8 deg
                phantom pitch on flat approach.
  target_pz():  return nominal_clearance when robot_x < STAIR_START_X
                to prevent premature pz_des raise on flat approach.
  swing_clearance(): terrain-height-adaptive (IIR-tracked).

FIXES (this session — descending stair support):
  stair_height() and stair_height_smooth(): extended from ascent-only
    to handle the top platform and the descending stair section.
  snap_to_tread(): extended to clamp footholds onto descending treads.
  pitch_ref(): descent naturally handled — front_z < rear_z on the
    descending section gives negative pitch (nose-down) which is the
    correct body-tilt target for descent. No extra guard needed since
    the front hip physically touches the lower step exactly when CoM
    crosses PLATFORM_END_X (no pre-activation artefact).
  target_pz(): descent naturally handled — avg_z tracks down as both
    hips move onto lower steps.
"""

import numpy as np
from config import NUM_LEGS, CONTACT_THR, dbg
from robot_params import RobotState


# ── Stair geometry (verified against scene_stairs_uniform.xml) ────────────────
# XML box geoms: pos = center, size = half-extents.  top_z = center_z + half_z.
#
# Ascending:  step1 x=[2.00,2.28] top=0.06m  ...  step5 x=[3.12,3.40] top=0.30m
# Platform:   x=[3.40, 4.40]  top=0.30m
# Descending: step6 x=[4.40,4.68] top=0.24m  ...  step10 x=[5.52,5.80] top=0.03m

STAIR_START_X    = 2.00    # ascending stairs begin
STAIR_TREAD      = 0.28    # tread width [m]
STAIR_RISER      = 0.06    # riser height [m]
STAIR_COUNT      = 5       # number of ascending steps
PLATFORM_START_X = 3.40    # = STAIR_START_X + STAIR_COUNT * STAIR_TREAD
PLATFORM_END_X   = 4.40    # top platform ends / descent begins
STAIR_COUNT_DOWN = 5       # number of descending steps (mirror)
DESCENT_END_X    = 5.80    # = PLATFORM_END_X + STAIR_COUNT_DOWN * STAIR_TREAD
LANDING_Z        = STAIR_COUNT * STAIR_RISER   # = 0.30 m

FOOT_RADIUS  = 0.022
HIP_X_OFFSET = 0.1934


class TerrainEstimator:

    SWING_CLEARANCE_FLAT    = 0.08
    SWING_CLEARANCE_TERRAIN = 0.18
    TOUCHDOWN_OFFSET        = 0.015

    def __init__(self, mode: str = "flat", smoothing: float = 0.70):
        assert mode in ("flat", "stairs", "uneven"), \
            f"Unknown terrain mode '{mode}'."
        self.mode   = mode
        self._alpha = smoothing
        self._terrain_z  = 0.0
        self._foot_z     = np.zeros(4)
        self._foot_seen  = np.zeros(4, dtype=bool)

    # ── Discrete stair height (step function) ─────────────────────────────────

    @staticmethod
    def stair_height(x: float) -> float:
        """
        Returns terrain surface height at world x-coordinate.
        Step-function (no smoothing) — used for failure detection and
        touchdown z initialisation.

        x < 2.00          : 0.00m  (flat approach)
        2.00 <= x < 3.40  : ascending 0.06 to 0.30m
        3.40 <= x < 4.40  : 0.30m  (top platform)
        4.40 <= x < 5.80  : descending 0.24 to 0.06m
        x >= 5.80          : 0.00m  (flat exit)
        """
        if x < STAIR_START_X:
            return 0.0

        # Ascending
        if x < PLATFORM_START_X:
            dx   = x - STAIR_START_X
            step = int(dx / STAIR_TREAD)
            if step >= STAIR_COUNT:
                return LANDING_Z
            return (step + 1) * STAIR_RISER

        # Top platform
        if x < PLATFORM_END_X:
            return LANDING_Z

        # Descending
        if x < DESCENT_END_X:
            dx   = x - PLATFORM_END_X
            step = int(dx / STAIR_TREAD)
            if step >= STAIR_COUNT_DOWN:
                return 0.0
            return LANDING_Z - (step + 1) * STAIR_RISER

        return 0.0

    # ── Smooth stair height (for pitch_ref / target_pz) ───────────────────────

    @staticmethod
    def stair_height_smooth(x: float) -> float:
        """
        Continuous-ramp approximation. RAMP_WIDTH = 0.08m: linear interpolation
        starts 8 cm before each riser so pitch/height references transition
        smoothly rather than as a step change.

        Descending mirrors ascending: pre-ramp DOWN 8 cm before each descent edge.

        Callers (pitch_ref, target_pz) guard against the pre-stair ramp
        artefact by returning neutral values when robot_x < STAIR_START_X.
        """
        RAMP_WIDTH = 0.08

        # Before ascending stairs
        if x < STAIR_START_X - RAMP_WIDTH:
            return 0.0
        if x < STAIR_START_X:
            alpha = (x - (STAIR_START_X - RAMP_WIDTH)) / RAMP_WIDTH
            return alpha * STAIR_RISER

        # Ascending stairs
        if x < PLATFORM_START_X:
            dx        = x - STAIR_START_X
            step      = int(dx / STAIR_TREAD)
            if step >= STAIR_COUNT:
                return LANDING_Z
            tread_pos = dx - step * STAIR_TREAD
            base_z    = (step + 1) * STAIR_RISER
            nrs = STAIR_TREAD - RAMP_WIDTH
            if tread_pos > nrs and step + 1 < STAIR_COUNT:
                alpha = (tread_pos - nrs) / RAMP_WIDTH
                return base_z + alpha * STAIR_RISER
            return base_z

        # Top platform
        if x < PLATFORM_END_X:
            return LANDING_Z

        # Descending stairs — mirror of ascending logic, ramps DOWN
        if x < DESCENT_END_X:
            dx        = x - PLATFORM_END_X
            step      = int(dx / STAIR_TREAD)
            if step >= STAIR_COUNT_DOWN:
                return 0.0
            tread_pos = dx - step * STAIR_TREAD
            base_z    = LANDING_Z - (step + 1) * STAIR_RISER
            nrs = STAIR_TREAD - RAMP_WIDTH
            if tread_pos > nrs and step + 1 < STAIR_COUNT_DOWN:
                alpha = (tread_pos - nrs) / RAMP_WIDTH
                return base_z - alpha * STAIR_RISER
            return base_z

        return 0.0

    # ── Terrain-geometry queries ───────────────────────────────────────────────

    def foot_touchdown_z(self, foot_world_x: float) -> float:
        if self.mode == "flat":
            return 0.02
        return self.stair_height(foot_world_x) + FOOT_RADIUS

    def snap_to_tread(self, x_td: float) -> float:
        """
        Clamp foothold x to the safe centre of the tread it lands on.
        MARGIN = 0.04 m clearance from each riser edge.
        Handles ascending, platform, and descending sections.
        """
        if self.mode == "flat":
            return x_td

        MARGIN = 0.04

        # Before ascending stairs
        if x_td < STAIR_START_X:
            if x_td > STAIR_START_X - MARGIN:
                return STAIR_START_X + MARGIN
            return x_td

        # Ascending stairs
        if x_td < PLATFORM_START_X:
            dx   = x_td - STAIR_START_X
            step = int(dx / STAIR_TREAD)
            if step >= STAIR_COUNT:
                return x_td
            tread_lo = STAIR_START_X + step * STAIR_TREAD + MARGIN
            tread_hi = STAIR_START_X + (step + 1) * STAIR_TREAD - MARGIN
            return float(np.clip(x_td, tread_lo, tread_hi))

        # Top platform — free placement
        if x_td < PLATFORM_END_X:
            return x_td

        # Descending stairs
        if x_td < DESCENT_END_X:
            dx   = x_td - PLATFORM_END_X
            step = int(dx / STAIR_TREAD)
            if step >= STAIR_COUNT_DOWN:
                return x_td
            tread_lo = PLATFORM_END_X + step * STAIR_TREAD + MARGIN
            tread_hi = PLATFORM_END_X + (step + 1) * STAIR_TREAD - MARGIN
            return float(np.clip(x_td, tread_lo, tread_hi))

        return x_td

    def pitch_ref(self, robot_x: float) -> float:
        """
        Desired body pitch [rad] for the MPC reference.
        Positive = nose-up, negative = nose-down. Clipped to +-0.21 rad (~12 deg).

        Ascending (2.0-3.4m):  +8.8 deg  (nose-up to match stair slope)
        Platform  (3.4-4.4m):   0 deg    (level)
        Descending(4.4-5.8m): -8.8 deg  (nose-down to match descent slope)

        PHYSICAL REASON for negative pitch on descent:
          On the descending section, front feet land on lower steps while rear
          feet are still on higher steps. If the body stays level, the CoM
          shifts forward of the support polygon, creating a nose-dive torque
          that the MPC must fight reactively. By commanding -8.8 deg pitch_ref,
          the MPC proactively distributes forces to maintain this geometry,
          keeping the CoM centred over the support polygon throughout descent.

        FIX (flat-approach bug, previous session):
          stair_height_smooth() has an 8cm pre-ramp before STAIR_START_X.
          At terrain-switch x=1.80m, this gives front_z=0.055m on flat ground,
          creating a +8.1 deg pitch target against zero physical slope.
          Guard: return 0.0 when robot_x < STAIR_START_X.
          No equivalent guard needed for descent: front hip naturally touches
          the first descent step at CoM x=4.207m (=PLATFORM_END_X-HIP_X_OFFSET),
          so there is no pre-activation artefact.
        """
        if self.mode == "flat":
            return 0.0

        # Guard: no phantom pitch before ascending stairs
        if robot_x < STAIR_START_X:
            return 0.0

        front_z  = self.stair_height_smooth(robot_x + HIP_X_OFFSET)
        rear_z   = self.stair_height_smooth(robot_x - HIP_X_OFFSET)
        dz       = front_z - rear_z
        hip_span = 2.0 * HIP_X_OFFSET
        return float(np.clip(np.arctan2(dz, hip_span), -0.21, 0.21))

    def target_pz(self, robot_x: float, nominal_clearance: float) -> float:
        """
        Desired CoM height = average terrain height under both hips + clearance.

        Ascending: pz_des rises from 0.338 to 0.638m (0.30m terrain + clearance).
        Platform:  pz_des constant at 0.638m.
        Descending:pz_des falls back from 0.638 to 0.338m as terrain drops.
        Flat exit: pz_des = nominal_clearance (~0.338m).

        The mpc_controller applies an asymmetric rate limiter on pz_des changes
        (UP: 0.12 m/s, DOWN: 0.04 m/s) so this target is tracked smoothly.

        FIX (flat-approach bug, previous session):
          pre-ramp artefact raised avg_z by 0.028m at x=1.80m on flat ground.
          Guard: return nominal_clearance when robot_x < STAIR_START_X.
        """
        if self.mode == "flat":
            return nominal_clearance

        if robot_x < STAIR_START_X:
            return nominal_clearance

        front_z = self.stair_height_smooth(robot_x + HIP_X_OFFSET)
        rear_z  = self.stair_height_smooth(robot_x - HIP_X_OFFSET)
        avg_z   = 0.5 * (front_z + rear_z)
        return avg_z + nominal_clearance

    # ═══════════════════════════════════════════════════════════════════════════
    # IIR terrain tracking
    # ═══════════════════════════════════════════════════════════════════════════

    def update(self, state: RobotState) -> None:
        """
        Update IIR terrain height estimate from stance foot z-positions.

        During ascent:  terrain_z tracks UP   (0.0 → 0.30 m).
        During descent: terrain_z tracks DOWN (0.30 → 0.0 m).

        The IIR filter (alpha=0.70) prevents single-step spikes from
        causing large force impulses in the swing clearance calculation.
        """
        if self.mode == "flat":
            return
        for i in range(NUM_LEGS):
            if state.cf[i] > CONTACT_THR:
                self._foot_z[i]    = state.foot_pos[i][2]
                self._foot_seen[i] = True
        if self._foot_seen.any():
            tracked_z = self._foot_z[self._foot_seen]
            raw = max(0.0, float(np.mean(tracked_z)) - FOOT_RADIUS)
            self._terrain_z = (self._alpha * self._terrain_z
                               + (1.0 - self._alpha) * raw)
        dbg(4, f"TerrainEst [{self.mode}]: "
               f"z_est={self._terrain_z:.4f}m  "
               f"foot_z={self._foot_z.round(3)}")

    def reset(self) -> None:
        self._terrain_z    = 0.0
        self._foot_z[:]    = 0.0
        self._foot_seen[:] = False

    def set_mode(self, mode: str) -> None:
        assert mode in ("flat", "stairs", "uneven"), \
            f"Unknown terrain mode '{mode}'."
        self.mode = mode

    # ── Query methods ─────────────────────────────────────────────────────────

    @property
    def terrain_z(self) -> float:
        return self._terrain_z

    def touchdown_z(self) -> float:
        if self.mode == "flat":
            return 0.02
        return self._terrain_z + self.TOUCHDOWN_OFFSET

    def swing_clearance(self) -> float:
        """
        Adaptive swing clearance based on IIR terrain height.

        Ascent:   terrain_z rises 0→0.30m, clearance scales 0.09→0.18m.
        Descent:  terrain_z falls 0.30→0m, clearance scales 0.18→0.09m.

        The adaptive behaviour is correct for both directions: the foot needs
        clearance above the TARGET surface (which the IIR tracks), not above
        the surface it is departing from.

        flat mode: always 0.08m (unchanged from original).
        """
        if self.mode == "flat":
            return self.SWING_CLEARANCE_FLAT

        t_z = max(0.0, self._terrain_z)

        ramp_range = 0.10
        extra = 0.01 + min(1.0, t_z / ramp_range) * (
            self.SWING_CLEARANCE_TERRAIN - self.SWING_CLEARANCE_FLAT - 0.01
        )
        return self.SWING_CLEARANCE_FLAT + extra

    def is_flat(self) -> bool:
        return self.mode == "flat"