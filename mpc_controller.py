"""
mpc_controller.py — Main MPC Controller
========================================

KEY FIXES in this version
--------------------------
(From previous session — still present)
  1. DO NOT reset pz_des to NOMINAL_HEIGHT at Phase C.
     The settled height (~0.338m) IS the correct target.
  2. Phase C warmup extended to 300ms.
  3. _prev_contacts correctly seeded from gait at Phase C activation.
  4. Swing notify_lift only called AFTER warmup ends.
  5. Phase B warmup high-frequency MPC for 100ms at B entry.

(New fixes — this session)
  6. BUG FIX — Raibert: command velocities propagated to swing controller.
     notify_lift() now receives cmd_vx_world / cmd_vy_world so that the
     Raibert heuristic can separate nominal drift from velocity-error
     correction.  Without this the touchdown always uses cmd_v=0,
     causing the 3.5× lateral overcorrection that drives sideways drift.

  7. BUG FIX — World-frame command rotation.
     cmd_vx / cmd_vy are in the BODY frame (aligned with the robot nose).
     The Raibert formula operates in the WORLD frame.
     We rotate:
         [vx_w]   [cos ψ  -sin ψ] [cmd_vx]
         [vy_w] = [sin ψ   cos ψ] [cmd_vy]
     where ψ = yaw = state.euler[2].
     Without this, a robot commanded to move in a diagonal direction
     after yawing will place feet incorrectly even when cmd_vy=0.
"""

import time
import numpy as np
import mujoco
from support_plane import SupportPlane

from config import (
    LEG_NAMES, NUM_LEGS, TORQUE_LIMIT,
    NOMINAL_HEIGHT, COLLAPSE_HEIGHT, TROT_READY_HEIGHT,
    LOG_CSV, DEBUG_LEVEL,
    dbg,
)
from robot_params import RobotState, Go2Params, GO2
from state_estimator import StateEstimator
from dynamics import RigidBodyDynamics
from mpc_solver import ConvexMPC
from gait_scheduler import TrotGait
from swing_controller import SwingController
from terrain_estimator import TerrainEstimator
from torque_utils import (
    grf_to_joint_torques, leg_to_ctrl, make_reference, PDStand, DebugLogger,
)


class Go2MPCController:

    STAND_DURATION     = 1.5   # Phase A: PD settling
    STAND_MPC_DURATION = 2.0   # Phase B: all-stance MPC

    _PHASE_B_WARMUP_DURATION = 0.10   # 100ms high-freq MPC at Phase B start

    # 300ms = 1.5 full gait cycles at T=0.4s, 50% duty
    # Robot needs time to stabilize at the new GRF mode before first swing
    _PHASE_C_WARMUP_DURATION = 0.30   # increased from 0.20s

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 K: int = 10, mpc_dt: float = 0.030,
                 gait_period: float = 0.40, duty: float = 0.50,
                 params: Go2Params = GO2,
                 terrain_mode: str = "flat"):
        """
        Parameters
        ----------
        terrain_mode : str
            "flat"   — original flat-ground behaviour, unchanged (default).
            "stairs" — terrain-aware: adaptive pz_des, larger swing clearance,
                       early-touchdown detection.
            "uneven" — same as "stairs" (future extension).
        """
        self.model  = model
        self.data   = data
        self.K      = K
        self.mpc_dt = mpc_dt
        self.p      = params

        # Terrain estimator — always starts in "flat" mode so Phase A/B/C
        # settling is identical to the working flat-ground controller.
        # Call switch_to_terrain_mode("stairs") mid-run when the robot
        # approaches the stairs.  Store the requested mode so reset() can
        # inform callers, but the estimator itself starts flat.
        self._terrain_mode_target = terrain_mode
        self.terrain = TerrainEstimator(mode="flat")

        self.estimator = StateEstimator(model, data)
        self.dyn       = RigidBodyDynamics(params)
        self.support_plane = SupportPlane()
        self.mpc       = ConvexMPC(K, params)
        self.gait      = TrotGait(gait_period, duty)
        # Pass terrain estimator to swing controller so it can use
        # terrain-aware touchdown z, swing clearance, and early touchdown.
        self.swing     = SwingController(self.gait, model, data,
                                         terrain=self.terrain)
        self.pd        = PDStand()
        self.logger    = DebugLogger(enabled=LOG_CSV)

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.pz_des = NOMINAL_HEIGHT
        # Nominal CoM clearance above terrain (set from settled height in Phase A).
        # On flat ground this equals pz_des; on stairs: pz_des = terrain_z + this.
        self._nominal_clearance: float = NOMINAL_HEIGHT

        _fz0 = params.mass * params.g / NUM_LEGS
        self._grf            = np.array([[0., 0., _fz0]] * NUM_LEGS, dtype=float)
        self._last_mpc       = -999.0
        self._phase_b_start  = None
        self._mpc_active     = False
        self._trot_active    = False
        self._pz_settled     = None
        self._solve_ms: list = []
        self._prev_contacts  = np.ones(4, dtype=bool)

        self._phase          = "A"
        self._recovery_mode  = False
        self._recovery_end   = 0.0
        self._contacts_sched = np.ones(4, dtype=bool)   # logged contact schedule

        self._pz_integral    = 0.0
        self._pz_last_t      = 0.0
        self._phase_c_start  = None
        self._warmup_done    = False   # track when Phase C warmup ends
        self._pitch_ref_smooth = 0.0   # FIX 8: IIR-smoothed pitch reference

        # ── Stairs-specific Q weights ────────────────────────────────────────
        # Tuned for scene_stairs_uniform.xml (6 cm riser / 28 cm tread):
        #   pitch Q  100 (5× flat) — aggressively track incline reference;
        #                             debug log shows 7-8° pitch oscillation
        #                             pre-stairs that creates fragility.
        #   roll  Q  25  (2.5× flat) — tighter lateral stability on steps.
        #   pz    Q  100 (↑ from 80) — strong height tracking so the MPC
        #                               raises the body promptly per step.
        #   ωy    Q  3   (↑ from 1)  — damp pitch rate oscillations that
        #                               cause the fatal nose-down pitch-over.
        #   vz    Q  5   (↑ from 3, = flat) — the robot must actively
        #                  control vertical velocity during ascent, not
        #                  allow free vertical drift.
        # State order: [φ, θ, ψ, px, py, pz, ωx, ωy, ωz, vx, vy, vz, -g]
        # FIXED WEIGHTS (v3) — root-cause analysis from second failure run:
        #
        # RC1: py=1, vy=2 was NEVER fixed from original — zero lateral authority.
        #      Diagonal stance (FL on step, RR on flat) creates lateral roll torque.
        #      With py=1/vy=2, the MPC exerts ~0.16N vs 149N robot — negligible.
        #      vy exploded to 0.38 m/s, causing full rollover at t=7.59s.
        #      FIX: py=15, vy=15 — strong lateral position/velocity tracking.
        #
        # RC2: pitch=100 with diagonal stance generates huge anti-pitch moments
        #      that manifest as roll couples in 2-leg stance (FL+RR).
        #      This directly caused the vy→0.38 m/s tumble at first step contact.
        #      FIX: pitch=50 — sufficient to track terrain slope without overcorrecting.
        #
        # RC3: vx=2 too weak — robot surged to 0.19→0.37 m/s despite cmd=0.08 m/s.
        #      Forward surge means each step is hit at 2-4× the intended speed,
        #      leaving no time for the MPC to regulate height/pitch per step.
        #      FIX: vx=8 — strong enough to prevent forward surge on incline.
        #
        # State order: [φ, θ, ψ, px, py, pz, ωx, ωy, ωz, vx, vy, vz, -g]
        # TUNED from log analysis (run ending t=9.5, roll=53°):
        #
        # pitch=50→80: pitch diverged +2°→+16° in 1.3s (t=6→7.3).
        #   At 50, the MPC penalty for 16° pitch error is 50*(0.28)²=3.9.
        #   At 80, it's 80*(0.28)²=6.3 — 60% stronger anti-pitch torque.
        #   Not 100+ because that caused roll coupling in previous runs.
        #
        # ωy=5→12: pitch rate reached 10°/s = 0.17 rad/s sustained.
        #   At ωy=5, penalty = 5*(0.17)²=0.15 — negligible damping.
        #   At ωy=12, penalty = 12*(0.17)²=0.35 — meaningful damping
        #   that prevents pitch from accelerating past ±10°.
        #
        # roll=25→35: roll hit 10° at t=8.1 with only 2 diagonal stance
        #   feet.  Stronger roll tracking helps reject the geometric roll
        #   torque from feet at different heights.
        #
        # vz=5→8: on stairs the robot must actively control vertical
        #   velocity.  Debug log shows vz oscillating ±0.15 m/s, which
        #   couples into pitch through the moment arm.
        #
        # TUNED from log analysis (pitch moment analysis at t=6.15):
        #
        # ROOT CAUSE: With pz Q=80 and pitch Q=80, the MPC pushes harder
        # on front feet (which are higher on stairs) to maintain height.
        # This creates a +20 Nm nose-up pitch moment that accelerates
        # the pitch divergence.  The MPC's pitch penalty is insufficient
        # to counteract the height-driven force distribution.
        #
        # FIX: Make pitch Q DOMINANT over pz Q (2:1 ratio).
        #   pitch=120, pz=60: the MPC now prioritizes pitch correction
        #   over height tracking.  This means the body may sag 1-2cm
        #   during transitions, but pitch stays within ±10° which prevents
        #   the cascade that leads to rollover.
        #
        # ωy=15: very strong pitch-rate damping.  Debug log showed pitch
        #   rate reaching 10°/s = 0.17 rad/s sustained for >1s.
        #   At ωy=15, penalty = 15*(0.17)² = 0.44 per tick, providing
        #   meaningful deceleration of pitch rotation.
        #
        # vx=12: robot surged from 0.08 to 0.21 m/s at t=6.15.
        #   Each step is hit at 2.6× intended speed, giving the MPC
        #   only 40% of the intended time to regulate per step.
        #
        # State order: [φ, θ, ψ, px, py, pz, ωx, ωy, ωz, vx, vy, vz, -g]
        #
        # TUNED from third run log analysis (roll failure at t=9.1):
        #
        # RC: Roll is the actual kill mode, NOT pitch.  At t=8.83, diagonal
        #   stance FL(step1)=139N vs RR(flat)=46N creates +9.3 Nm roll torque
        #   (87 rad/s² → 4°/tick).  With roll Q=35 vs pitch Q=120, the MPC
        #   sacrificed roll to correct pitch.  Roll diverged to -9.3° in one
        #   tick, then 30° → crash.
        #
        # FIX: roll Q 35→60 — EQUAL priority with pz (both 60).
        #   The MPC now distributes forces to zero roll torque FIRST, then
        #   uses remaining authority for pitch.  This prevents the roll
        #   divergence cascade that caused all 3 previous failures.
        #
        # ωx: 1→8 — strong roll-rate damping.  Log showed roll rate hitting
        #   15°/s (0.26 rad/s) before divergence.  At ωx=8: penalty =
        #   8*(0.26)² = 0.54 per tick, providing meaningful deceleration.
        #
        # vx: 12→25 — robot surged from cmd=0.05 to actual=0.21 m/s (4.2×).
        #   At vx=25: penalty for 0.16 m/s error = 25*(0.16)² = 0.64,
        #   comparable to pitch error penalty = 120*(0.14)² = 2.35.
        #   This prevents the speed surge that leaves <1s per tread.
        #
        # FIX: pz 60→100.  In the first successful climb (t=35s, x=2.43),
        # pitch was 10-14° on the flat approach (x=1.6→2.0).  Root cause:
        # pz=60 (down from flat-Q 150) gave weak height tracking, letting
        # the body sag at the front, which pitched up.  At pz=100, height
        # tracking is 67% of flat-Q strength — strong enough to prevent sag
        # while still allowing pitch Q=120 to dominate on stairs (1.2:1 ratio).
        # Stairs Q: with corrected composite inertia, less aggressive weights
        # are needed since the MPC now correctly models angular authority.
        # FIX: roll Q 40→60, ωx 5→10 — the primary crash mode on stairs is
        # ROLL divergence from asymmetric GRF (front feet on step push harder
        # than rear feet on flat → roll torque).  With roll=40 the MPC penalty
        # for 5° roll error was only 40*(0.087)²=0.30 — negligible vs pz penalty.
        # At roll=60: 60*(0.087)²=0.45 — comparable to pitch penalty, ensuring
        # the MPC distributes forces to zero roll torque FIRST.
        # pitch Q 80→65: with corrected inertia the MPC generates sufficient
        # pitch correction at lower Q.  High pitch Q drives asymmetric forces
        # that couple into roll on diagonal stance.
        # FIX: pz 80→50 — high pz Q forces the MPC to push hard on front feet
        # (which are on the step) to maintain height, creating a pitch-up torque
        # that drove pitch to +14°.  At pz=50, the robot sags 1-2cm during step
        # transitions but pitch stays within ±10°.
        # pitch 65→85 — with duty=0.85 giving 70% 4-foot overlap, roll coupling
        # from pitch correction is much reduced, so we can safely increase pitch Q.
        # Stairs Q weights — pz-DOMINANT over pitch (2:1 ratio).
        # Previous: pitch=85, pz=50 (pitch-dominant) → MPC sacrificed height
        # for pitch correction → height oscillation → pitch/roll cascade → crash.
        # Fix: pz=100 (=flat), pitch=50.  Height stability prevents the
        # vertical bouncing that couples into pitch and then roll.
        # roll=80, ωx=15: roll divergence is the kill mode on stairs.
        # ωy=25: strong pitch-rate damping prevents overshoot past pitch_ref.
        # TUNED from run analysis — pitch divergence to +14° on platform:
        #
        # ROOT CAUSE 1: pitch Q=25 too weak on platform (penalty 1.09 for 12°).
        # ROOT CAUSE 2: pitch_ref capped at 5.2° while terrain slope is 8.8°,
        #   creating a permanent 3.6° "phantom error" during ascent.  The MPC
        #   wasted force correcting terrain-induced pitch, causing roll coupling.
        # ROOT CAUSE 3: the wasted correction accumulated extra pitch momentum
        #   that carried onto the platform where Q=25 couldn't damp it.
        #
        # FIX STRATEGY: reduce pitch ERROR during ascent (via pitch_ref cap
        #   increase in terrain_estimator.py), then moderate Q handles platform.
        #
        # pitch Q 25→35: conservative increase. Q=55 caused stumble at step 5
        #   (t=35s) from aggressive pitch correction during 2-foot diagonal
        #   stance.  Q=35 provides 40% more correction on platform (penalty
        #   35×0.14²=0.69 for 8° error) without destabilizing ascent.
        #   Roll Q=120 remains 3.4× dominant.
        #
        # ωy 15→20: 33% more pitch-rate damping.  At stair riser impact,
        #   pitch rate ~15°/s (0.26 rad/s): penalty 15×0.26²=1.01 → 20×0.26²
        #   =1.35.  Prevents pitch overshoot entering the platform.
        #
        # State order: [φ, θ, ψ, px, py, pz, ωx, ωy, ωz, vx, vy, vz, -g]
        #
        # Phase-adaptive Q: contradictory requirements between ascent and platform.
        #   Ascent: high ωy (20) needed to damp step-impact perturbations.
        #   Platform: low ωy (12) needed for pitch correction bandwidth.
        # Solution: two Q vectors blended by terrain slope (|pitch_ref|).
        # stairs_Q: used during ascent/descent (|pitch_ref| > 0).
        self._stairs_Q = np.array([
           150.,  50.,  5.,     # roll, pitch, yaw
             1.,  15., 100.,    # px, py, pz
            30.,  20.,  5.,     # ωx, ωy, ωz
            15.,  20.,  8.,     # vx, vy, vz
             0.,                # -g
        ], dtype=float)
        # platform_Q: for flat sections in stairs mode (|pitch_ref| ≈ 0).
        # Tuning iteration results (platform phase):
        #   ωy=12, roll=40: mean=1.6° (good), |max|=11.7° (oscillations)
        #   ωy=20 (stairs): mean=7.0° (overdamped, no recovery)
        # Split the difference: ωy=15, roll=60, ωx=8 for post-stair damping
        # while preserving pitch correction bandwidth.
        self._platform_Q = np.array([
            60.,  85.,  8.,     # roll (↑ damp oscillations), pitch (↑ 55→85), yaw (drift ctrl)
             1.,   1., 100.,    # px, py, pz
             8.,  15.,  3.,     # ωx (↑ roll damp), ωy (15: balanced), ωz
            15.,   8.,  5.,     # vx, vy, vz
             0.,                # -g
        ], dtype=float)

        print(f"\n[MPC Controller]  K={K}  dt={mpc_dt*1e3:.0f}ms  "
              f"T_gait={gait_period:.2f}s  duty={duty:.0%}  "
              f"mass={params.mass}kg")
        print(f"  Phase A: PD stand {self.STAND_DURATION:.1f}s  "
              f"→  Phase B: all-stance MPC {self.STAND_MPC_DURATION:.1f}s  "
              f"→  Phase C: trot MPC")
        print(f"  f ∈ [{params.f_min},{params.f_max}]N  "
              f"α={params.alpha:.1e}  μ={params.mu}")
        print("MPC mass:", self.p.mass)
        print("MuJoCo total mass:", float(self.model.body_mass.sum()))
        trunk_id = 1   # base_link (from config.py)
        print("MuJoCo trunk mass:", float(self.model.body_mass[trunk_id]))
        print("MuJoCo trunk inertia:", self.model.body_inertia[trunk_id])
        print(f"  Terrain mode: '{self.terrain.mode}'  "
              f"swing_cl={self.terrain.swing_clearance():.3f}m  "
              f"td_z={self.terrain.touchdown_z():.3f}m")




    # ── Helper: rotate body-frame commands into world frame ──────────────────
    def _cmd_world_frame(self, state: RobotState):
        """
        Returns (cmd_vx_world, cmd_vy_world).

        cmd_vx / cmd_vy are expressed in the body frame (robot nose = +x).
        Raibert and any world-frame planner need these in the world frame,
        so we rotate by current yaw ψ = state.euler[2].

            [vx_w]   [cos ψ  -sin ψ] [cmd_vx]
            [vy_w] = [sin ψ   cos ψ] [cmd_vy]
        """
        psi   = state.euler[2]
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)
        vx_w  =  c_psi * self.cmd_vx - s_psi * self.cmd_vy
        vy_w  =  s_psi * self.cmd_vx + c_psi * self.cmd_vy
        return float(vx_w), float(vy_w)

    def step(self) -> RobotState:
        state = self.estimator.update()
        t     = state.t
        pz    = state.pos[2]

        # Update terrain estimate from current stance foot positions.
        # In flat mode this is a no-op; no behaviour change.
        self.terrain.update(state)

        if t > self._pz_last_t:
            dt_step           = t - self._pz_last_t
            self._pz_integral += (self.pz_des - pz) * dt_step
            self._pz_integral  = np.clip(self._pz_integral, -0.5, 0.5)
            self._pz_last_t    = t

        # ── Recovery mode ─────────────────────────────────────────────────────
        if self._recovery_mode:
            ctrl = self.pd.compute(state)
            if t > self._recovery_end:
                dbg(2, f"[t={t:.2f}s] Recovery complete (pz={pz:.3f}m)")
                self._recovery_mode  = False
                self._trot_active    = False
                self._mpc_active     = False
                self._phase          = "B"
                self._pz_integral    = 0.0
                self._warmup_done    = False
            self._apply(ctrl)
            self.logger.log(state, self._grf, ctrl, "RECOVERY",
                            self.pz_des, False, False)
            return state

        # ── Phase A: PD settling ───────────────────────────────────────────────
        if t < self.STAND_DURATION:
            self._phase = "A"
            ctrl = self.pd.compute(state)
            if t > self.STAND_DURATION - 0.8:
                self._pz_settled = (
                    pz if self._pz_settled is None
                    else 0.98 * self._pz_settled + 0.02 * pz
                )
            self._apply(ctrl)
            self.logger.log(state, self._grf, ctrl, "A_PD",
                            self.pz_des, False, False)
            return state

        # ── Collapse detection ─────────────────────────────────────────────────
        if pz < COLLAPSE_HEIGHT and self._mpc_active:
            dbg(1, f"[t={t:.2f}s] COLLAPSE pz={pz:.3f}m → PD recovery for 2 s")
            self._recovery_mode = True
            self._recovery_end  = t + 2.0
            ctrl = self.pd.compute(state)
            self._apply(ctrl)
            return state

        # ── First time entering Phase B ───────────────────────────────────────
        if not self._mpc_active:
            self._phase = "B"
            if self._pz_settled is not None and self._pz_settled > COLLAPSE_HEIGHT:
                # FIX: Use the settled height as pz_des for ALL phases.
                # Do NOT reset to NOMINAL_HEIGHT at Phase C — it causes the
                # robot to drop during the warmup phase.
                self.pz_des = float(self._pz_settled)
                # Store as nominal clearance above terrain.
                # Subtract terrain_z so the foot geometry offset (~0.022m) cancels:
                #   target_pz = terrain_z + (pz_settled - terrain_z) = pz_settled  ← flat ground (no drift)
                #   target_pz = terrain_z_step + (pz_settled - terrain_z_flat)      ← on steps (correct)
                self._nominal_clearance = float(self._pz_settled) - self.terrain.terrain_z
            print(f"\n[t={t:.2f}s] ── Phase B: ALL-STANCE MPC  "
                  f"(pz_des={self.pz_des:.3f}m, settled={self._pz_settled})")
            self._mpc_active  = True
            self._last_mpc    = -999.0
            self._pz_integral = 0.0
            self._phase_b_start = t
            _fz0 = self.p.mass * self.p.g / NUM_LEGS
            self._grf = np.array([[0., 0., _fz0]] * NUM_LEGS, dtype=float)

        # ── Phase C gate ──────────────────────────────────────────────────────
        if (not self._trot_active
                and t >= self.STAND_DURATION + self.STAND_MPC_DURATION):
            if pz >= TROT_READY_HEIGHT:
                pitch_deg = abs(np.degrees(state.euler[1]))
                if pitch_deg > 2.0:
                    dbg(2, f"[t={t:.2f}s] Trot delayed: |pitch|={pitch_deg:.1f}° > 2°")
                else:
                    print(f"\n[t={t:.2f}s] ── Phase C: TROT MPC  "
                          f"(pz={pz:.3f}m ≥ {TROT_READY_HEIGHT}m)")
                    self._trot_active   = True
                    self._phase         = "C"
                    self._phase_c_start = t
                    self._warmup_done   = False
                    self.gait.activate(t)

                    # FIX: Keep pz_des = settled height, NOT NOMINAL_HEIGHT.
                    # The original code reset to 0.32m here, causing the MPC
                    # to reduce upward forces and drop the robot 18mm before
                    # first swing. The settled height IS the correct target.
                    # pz_des is already correct from Phase B.
                    # self.pz_des = NOMINAL_HEIGHT  ← REMOVED THIS BUG

                    self._pz_integral = 0.0
                    print(f"  [Phase C CHECK] terrain.mode='{self.terrain.mode}'  "
                          f"swing_cl={self.terrain.swing_clearance():.3f}m  "
                          f"td_z={self.terrain.touchdown_z():.3f}m  "
                          f"terrain_z={self.terrain.terrain_z:.4f}m  "
                          f"nominal_cl={self._nominal_clearance:.3f}m")

                    # Seed _prev_contacts from actual gait phase
                    self._prev_contacts = self.gait.contact_at(t).copy()

                    # Pre-warm GRF for initial stance feet (all-stance warmup)
                    # During warmup all 4 feet are in stance
                    fz_per_foot = self.p.mass * self.p.g / NUM_LEGS
                    fz_per_foot = np.clip(fz_per_foot, self.p.f_min, self.p.f_max)
                    self._grf = np.array([[0., 0., fz_per_foot]] * NUM_LEGS, dtype=float)
                    self._last_mpc = -999.0
                    dbg(2, f"Phase C start: pz_des={self.pz_des:.3f}m (settled height kept)")
            else:
                dbg(2, f"[t={t:.2f}s] Trot delayed: pz={pz:.3f} < "
                        f"{TROT_READY_HEIGHT}m")

        # ── MPC solve (rate-limited, with Phase B warmup) ─────────────────────
        in_phase_b_warmup = (
            self._phase_b_start is not None and
            not self._trot_active and
            (t - self._phase_b_start) < self._PHASE_B_WARMUP_DURATION
        )
        time_since_mpc = t - self._last_mpc
        should_solve   = in_phase_b_warmup or (time_since_mpc >= self.mpc_dt)

        if should_solve:
            t0 = time.perf_counter()
            self._run_mpc(state)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            self._solve_ms.append(elapsed_ms)
            self._last_mpc = t

        # ── Phase B: all-stance torques ───────────────────────────────────────
        if not self._trot_active:
            tau_stance = grf_to_joint_torques(self.model, self.data,
                                               state, self._grf)
            ctrl = leg_to_ctrl(tau_stance)

            if DEBUG_LEVEL >= 2 and int(t * 10) % 5 == 0:
                ratio = self._grf[:, 2].sum() / (self.p.mass * self.p.g)
                dbg(2, f"Phase B  t={t:.2f}  pz={pz:.3f}/{self.pz_des:.3f}  "
                       f"GRF_ratio={ratio:.2f}  "
                       f"c={''.join('Y' if c else 'N' for c in state.contact)}")

        # ── Phase C: trot (MPC stance + PD swing) ────────────────────────────
        else:
            self._phase = "C"

            # Check if warmup period has ended
            in_phase_c_warmup = (
                self._phase_c_start is not None and
                (t - self._phase_c_start) < self._PHASE_C_WARMUP_DURATION
            )

            # When warmup just ended, reset swing controller
            if not in_phase_c_warmup and not self._warmup_done:
                self._warmup_done = True
                # Re-seed _prev_contacts from actual gait phase at warmup end
                # so the first lift-off notification is correct
                self._prev_contacts = self.gait.contact_at(t).copy()
                dbg(2, f"[t={t:.2f}s] Phase C warmup done. Trot starting. "
                       f"contacts={self._prev_contacts}")

            if in_phase_c_warmup:
                # All feet in stance during warmup — no swing
                contacts_sched = np.ones(4, dtype=bool)
            else:
                contacts_sched = self.gait.contact_at(t)

            # Safety: if schedule is all-swing, fallback
            if contacts_sched.sum() == 0:
                contacts_sched = np.ones(4, dtype=bool)

            # Store for logger so plots show true gait schedule, not the
            # pz-override state.contact (which is always all-True when standing)
            self._contacts_sched = contacts_sched.copy()

            # ── FIX 6 & 7: compute world-frame commands ONCE per step ────────
            # Rotate body-frame cmd_vx/cmd_vy → world frame using current yaw.
            # This is used for Raibert touchdown placement so the foot lands
            # in the right spot even when the robot has rotated.
            cmd_vx_world, cmd_vy_world = self._cmd_world_frame(state)
            # ─────────────────────────────────────────────────────────────────

            # Detect lift-off transitions (only after warmup)
            if not in_phase_c_warmup:
                for i in range(NUM_LEGS):
                    if not contacts_sched[i] and self._prev_contacts[i]:
                        # FIX 6: pass world-frame commands so Raibert can
                        # separate nominal drift from velocity-error correction
                        self.swing.notify_lift(
                            i, t, state.foot_pos[i], state,
                            cmd_vx_world=cmd_vx_world,
                            cmd_vy_world=cmd_vy_world,
                        )
            self._prev_contacts = contacts_sched.copy()

            # Swing torques (swing legs only)
            tau_swing = self.swing.compute_torques(state, contacts_sched)

            # Zero swing-leg forces
            grf_stance = self._grf.copy()
            for i in range(NUM_LEGS):
                if not contacts_sched[i]:
                    grf_stance[i] = 0.0

            tau_stance = grf_to_joint_torques(self.model, self.data,
                                               state, grf_stance)
            tau_leg = np.zeros(12)
            for i in range(NUM_LEGS):
                s = slice(i*3, i*3+3)
                tau_leg[s] = tau_stance[s] if contacts_sched[i] else tau_swing[s]
            ctrl = leg_to_ctrl(tau_leg)

            if DEBUG_LEVEL >= 2 and int(t * 10) % 5 == 0:
                sched = ''.join('S' if c else 'W' for c in contacts_sched)
                real  = ''.join('Y' if c else 'N' for c in state.contact)
                dbg(2, f"Phase C  t={t:.2f}  pz={pz:.3f}  "
                       f"sched={sched}  real={real}  "
                       f"vx={state.vel[0]:.2f}  vy={state.vel[1]:.3f}  "
                       f"cmd_w=({cmd_vx_world:.2f},{cmd_vy_world:.2f})")

        ctrl_clipped = np.clip(ctrl, -TORQUE_LIMIT, TORQUE_LIMIT)
        self._apply(ctrl_clipped)
        self.logger.log(state, self._grf, ctrl_clipped, self._phase,
                        self.pz_des, self._trot_active, self._mpc_active,
                        contacts_sched=self._contacts_sched)
        return state

    def _apply(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        # mj_forward recomputes derived quantities (contacts, sensor data,
        # kinematics) so the state estimator reads fresh values on the next
        # step() call.  Without this, the estimator reads stale pre-step
        # data, causing trajectory divergence on stairs (roll crash at t≈37s).
        # The interactive viewer calls mj_forward via viewer.sync(), which
        # is why the viewer run succeeded while headless failed.
        mujoco.mj_forward(self.model, self.data)

    def _run_mpc(self, state: RobotState) -> None:
        x0  = state.mpc_x()
        psi = state.euler[2]
        t   = state.t

        # ── Contact schedule ─────────────────────────────────────
        if not self._trot_active:
            schedule = np.ones((self.K, NUM_LEGS), dtype=bool)
        else:
            in_c_warmup = (
                self._phase_c_start is not None and
                (t - self._phase_c_start) < self._PHASE_C_WARMUP_DURATION
            )
            if in_c_warmup:
                schedule = np.ones((self.K, NUM_LEGS), dtype=bool)
            else:
                schedule = self.gait.schedule(t, self.mpc_dt, self.K)
                if schedule.sum() == 0:
                    schedule[:] = True

        # ── Support plane update ─────────────────────────────────
        if self.terrain.is_flat():
            support_plane = None
        else:
            self.support_plane.update(state.foot_pos, state.contact)
            support_plane = self.support_plane

        # ── Terrain adaptive references ──────────────────────────
        pitch_ref = 0.0
        Q_active  = None

        if not self.terrain.is_flat():

            target_pz = self.terrain.target_pz(
                state.pos[0], self._nominal_clearance)

            # Rate limiter for pz_des changes.  When the robot crosses a stair
            # edge, target_pz jumps +3cm (half-riser, since we average front
            # and rear terrain).  The rate limit prevents force impulses that
            # couple into pitch/roll.
            # FIX: increased 3→5 cm/s.  At 3 cm/s, a 3cm half-riser jump
            # takes 1.0s to track.  Debug log shows pz_err growing to +0.016
            # at t=7.4 (pz lagging by 16mm), causing the MPC to generate
            # excessive upward force that couples into pitch oscillation.
            # At 5 cm/s, the same 3cm jump tracks in 0.6s — fast enough to
            # prevent lag while still smooth enough to avoid force spikes.
            # ── FIX B: asymmetric pz_des rate limiter ────────────────────────
            # BEFORE (too slow ascending → OVERFORCE → roll crash):
            #   _PZ_RATE = 0.05 m/s = 1.25mm/tick  (up AND down)
            #   A 30mm riser takes 24 ticks = 600ms → sustained OVERFORCE.
            #   At t=23.05: pz_des=0.428, pz=0.391 (37mm lag) → cf[FL]=365N
            #   → roll torque=40Nm → Δroll=4.7°/tick → crash at roll=25°.
            #
            # AFTER: separate rates for ascent and descent.
            #   UP:   0.12 m/s = 3.0mm/tick → 30mm riser in 10 ticks (250ms)
            #   DOWN: 0.04 m/s = 1.0mm/tick → gentle descent, no sudden drop
            #   OVERFORCE window: 600ms → 250ms (58% reduction).
            # FIX: reduced UP rate 0.12→0.06 m/s.  At 0.12, a 30mm half-riser
            # jump tracks in 250ms, creating a sustained pz error that drives
            # asymmetric GRF (front feet push harder → roll torque).
            # At 0.06, the same jump tracks in 500ms — gentler force transients.
            _PZ_RATE_UP   = 0.06   # m/s — ascending (was 0.12)
            _PZ_RATE_DOWN = 0.04   # m/s — descending
            pz_error = target_pz - self.pz_des
            if pz_error > 0:
                _max_d = _PZ_RATE_UP   * self.mpc_dt
            else:
                _max_d = _PZ_RATE_DOWN * self.mpc_dt
            self.pz_des += float(np.clip(pz_error, -_max_d, _max_d))

            # FIX: pitch_ref IIR α=0.5→0.3 (30% old, 70% new).
            # With α=0.5, the smoothed pitch_ref lags the geometric slope by
            # 3-6° for ~75ms after crossing a step edge.  During this lag, the
            # MPC generates anti-pitch torques (pitch Q=80 × 6°² = large) that
            # actively push the robot backward, opposing the climb.
            # α=0.3 settles in ~2 ticks (50ms), reducing the lag to 1-2°.
            # The geometric pitch_ref from terrain_estimator is deterministic
            # (not noisy), so aggressive tracking is safe.
            raw_pitch_ref = self.terrain.pitch_ref(state.pos[0])
            self._pitch_ref_smooth = (0.3 * self._pitch_ref_smooth
                                      + 0.7 * raw_pitch_ref)
            pitch_ref = self._pitch_ref_smooth

            # Phase-adaptive Q selection based on position (not pitch_ref).
            # pitch_ref oscillates as hips cross step edges, causing Q weight
            # oscillation that destabilizes roll during ascent.
            # Position-based selection is stable and deterministic.
            # Terrain layout: ascent [2.0,3.4], platform [3.4,4.4], descent [4.4,5.8]
            x = state.pos[0]
            if x > 1.90:
                if 3.30 <= x <= 4.50:
                    Q_active = self._platform_Q   # top platform: low ωy for recovery
                elif x > 5.70:
                    Q_active = self._platform_Q   # post-descent flat: low ωy
                else:
                    Q_active = self._stairs_Q     # slopes: high ωy for damping

        cmd  = np.array([self.cmd_vx, self.cmd_vy, self.cmd_wz])
        # FIX 9: pass terrain for per-step pitch_ref and pz_des in horizon.
        Xref = make_reference(
            x0, cmd, self.pz_des,
            self.mpc_dt, self.K,
            self._pz_integral,
            pitch_ref=pitch_ref,
            terrain=self.terrain if not self.terrain.is_flat() else None,
            nominal_clearance=self._nominal_clearance,
        )

        # ── Terrain-aware dynamics ───────────────────────────────
        Ad_list, Bd_list = [], []

        for k in range(self.K):
            A_c = self.dyn.Ac(psi, support_plane=support_plane)

            B_c = self.dyn.Bc(
                state.r_feet,
                psi,
                schedule[k],
                support_plane=support_plane
            )

            Ad, Bd = self.dyn.discretise(A_c, B_c, self.mpc_dt)

            Ad_list.append(Ad)
            Bd_list.append(Bd)

        Aqp, Bqp = self.mpc.condense(Ad_list, Bd_list)

        H, g = self.mpc.cost(
            Aqp, Bqp,
            x0, Xref,
            Q_override=Q_active
        )

        C, lb, ub = self.mpc.constraints(
            schedule,
            support_plane=support_plane
        )

        U_opt = self.mpc.solve(H, g, C, lb, ub)

        new_grf = U_opt[0:12].reshape(4, 3)

        # ── GRF smoothing ────────────────────────────────────────
        # GRF smoothing: β controls how fast the QP solution is applied.
        # Flat: β=0.75 (proven stable from flat-ground runs).
        # Stairs: β=0.65 (was 0.85 → too slow, was 0.50 → too fast).
        # At β=0.85: stale forces persist 7+ ticks, causing ±50% wr oscillation.
        # At β=0.50: force changes too fast, causing pitch/roll oscillation.
        # β=0.65: tracks QP changes in 4 ticks (~100ms), ±30% wr oscillation.
        beta = 0.85 if self.terrain.is_flat() else 0.50

        if self._trot_active and not np.all(~schedule[0]):
            n_stance = int(schedule[0].sum())
            fz_seed  = np.clip(
                self.p.mass * self.p.g / max(n_stance, 1),
                self.p.f_min,
                self.p.f_max
            )
            for i in range(NUM_LEGS):
                # FIX: threshold 0.5→0.7.  On stairs, legs transitioning from
                # swing carry _grf[i,2]≈0.  With β=0.50 smoothing and seed at
                # 50%, the first-tick stance force is ~50% of needed (37N vs 75N).
                # This creates the 100ms underforce window (wr=0.64) visible in
                # the log at t=7.5.  Seeding at 70% ensures legs start at full
                # target force immediately, eliminating the underforce spike.
                if schedule[0, i] and self._grf[i, 2] < fz_seed * 0.7:
                    self._grf[i] = np.array([0.0, 0.0, fz_seed])

        self._grf = beta * new_grf + (1.0 - beta) * self._grf

        if DEBUG_LEVEL >= 3:
            total_fz = self._grf[:, 2].sum()
            weight   = self.p.mass * self.p.g
            if total_fz < 0.5 * weight:
                dbg(1, f"[WARN t={t:.3f}s] Σfz={total_fz:.1f}N < "
                       f"0.5·W={0.5*weight:.1f}N — robot may fall")    

    def switch_to_terrain_mode(self, mode: str = "stairs") -> None:
        """
        Activate terrain-aware mode mid-run (e.g. when robot approaches stairs).

        Call this once from the simulation loop when the robot's x-position
        crosses the stair approach threshold.  At that moment we:
          1. Switch the terrain estimator from "flat" to the new mode so it
             begins tracking stance foot z-positions.
          2. Compute _nominal_clearance from the current settled pz and
             the current (still ≈ 0) terrain estimate, so that
               target_pz = terrain_z + _nominal_clearance ≈ pz_des
             keeps the robot at its current height as the estimator warms up.
        """
        if self.terrain.mode == mode:
            return   # already active
        self.terrain.set_mode(mode)
        # BUG FIX (nominal_clearance): use the SETTLED height, not pz_des.
        # When terrain switches before Phase B is complete (or during Phase A),
        # pz_des may still be NOMINAL_HEIGHT=0.320 instead of the settled
        # ~0.338m.  Using pz_des=0.320 sets nominal_clearance=0.320, causing
        # target_pz = terrain_z + 0.320 = 0.320 on flat ground.  The MPC then
        # tracks pz_des DOWN from 0.338 to 0.320 throughout Phase B (GRF_ratio
        # drops to 0.91, confirmed in log).  Use _pz_settled if available
        # (Phase A/B did converge), otherwise fall back to pz_des.
        ref_pz = (float(self._pz_settled)
                  if self._pz_settled is not None and self._pz_settled > COLLAPSE_HEIGHT
                  else self.pz_des)
        # Re-anchor clearance to ref_pz so no step change on switch.
        # BUG FIX: removed +0.02 buffer that pushed pz_des 2cm above settled
        # height.  On flat ground this forced the MPC to generate extra upward
        # force during the entire approach, consuming control authority that
        # should be reserved for stair response.
        self._nominal_clearance = ref_pz - self.terrain.terrain_z
        self.pz_des = ref_pz   # ensure pz_des also reflects settled height
        src = "settled" if (self._pz_settled is not None and self._pz_settled > COLLAPSE_HEIGHT) else "pz_des"
        print(f"\n[switch_to_terrain_mode] → '{mode}'  "
              f"pz_des={self.pz_des:.3f}m (from {src})  "
              f"terrain_z={self.terrain.terrain_z:.3f}m  "
              f"nominal_clearance={self._nominal_clearance:.3f}m")

    @property
    def grf(self) -> np.ndarray:
        return self._grf.copy()

    def stats(self) -> dict:
        if not self._solve_ms:
            return {}
        a = np.array(self._solve_ms)
        return {
            "n"      : len(a),
            "mean_ms": float(a.mean()),
            "max_ms" : float(a.max()),
            "min_ms" : float(a.min()),
            **self.mpc.stats,
        }