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
                 params: Go2Params = GO2):

        self.model  = model
        self.data   = data
        self.K      = K
        self.mpc_dt = mpc_dt
        self.p      = params

        self.estimator = StateEstimator(model, data)
        self.dyn       = RigidBodyDynamics(params)
        self.mpc       = ConvexMPC(K, params)
        self.gait      = TrotGait(gait_period, duty)
        self.swing     = SwingController(self.gait, model, data)
        self.pd        = PDStand()
        self.logger    = DebugLogger(enabled=LOG_CSV)

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.pz_des = NOMINAL_HEIGHT

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

        self._pz_integral    = 0.0
        self._pz_last_t      = 0.0
        self._phase_c_start  = None
        self._warmup_done    = False   # track when Phase C warmup ends

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
                        self.pz_des, self._trot_active, self._mpc_active)
        return state

    def _apply(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def _run_mpc(self, state: RobotState) -> None:
        x0  = state.mpc_x()
        psi = state.euler[2]
        t   = state.t

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

        cmd  = np.array([self.cmd_vx, self.cmd_vy, self.cmd_wz])
        Xref = make_reference(x0, cmd, self.pz_des,
                              self.mpc_dt, self.K, self._pz_integral)

        Ad_list, Bd_list = [], []
        for k in range(self.K):
            A_c = self.dyn.Ac(psi)
            B_c = self.dyn.Bc(state.r_feet, psi, schedule[k])
            Ad, Bd = self.dyn.discretise(A_c, B_c, self.mpc_dt)
            Ad_list.append(Ad)
            Bd_list.append(Bd)

        Aqp, Bqp  = self.mpc.condense(Ad_list, Bd_list)
        H, g      = self.mpc.cost(Aqp, Bqp, x0, Xref)
        C, lb, ub = self.mpc.constraints(schedule)
        U_opt     = self.mpc.solve(H, g, C, lb, ub)

        self._grf = U_opt[0:12].reshape(4, 3)
        new_grf = U_opt[0:12].reshape(4, 3)
        # Exponential smoothing (critical for stable trot)
        beta = 0.6
        self._grf = beta * new_grf + (1 - beta) * self._grf

        if DEBUG_LEVEL >= 3:
            total_fz = self._grf[:, 2].sum()
            weight   = self.p.mass * self.p.g
            if total_fz < 0.5 * weight:
                dbg(1, f"[WARN t={t:.3f}s] Σfz={total_fz:.1f}N < "
                       f"0.5·W={0.5*weight:.1f}N — robot may fall")

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