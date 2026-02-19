"""
simulation.py — MuJoCo Simulation Harness
==========================================
Paper: Di Carlo et al., IROS 2018

Wraps the MuJoCo model + Go2MPCController into a convenient simulation class
with two run modes:

    verify(duration)         — headless run with console logging table
    run_with_viewer(duration) — interactive MuJoCo passive viewer

The velocity command profile ramps up gradually so the MPC has time to
settle at each speed before the next step.

Usage (from phase2_6_mpc_go2_complete.py)::

    sim = Go2Simulation(xml=XML_PATH, K=10, mpc_dt=0.030,
                        gait_period=0.40, duty=0.50)
    sim.verify(duration=12.0)
    sim.run_with_viewer(duration=25.0)
"""

import numpy as np
import mujoco
import mujoco.viewer

from config import XML_PATH, LEG_NAMES, LOG_CSV, LOG_FILE
from robot_params import RobotState, Go2Params, GO2
from mpc_controller import Go2MPCController


class Go2Simulation:
    """
    Simulation harness combining MuJoCo model + Go2MPCController.

    Parameters
    ----------
    xml         : str   — path to MuJoCo XML scene file
    K           : int   — MPC horizon length
    mpc_dt      : float — MPC update interval [s]
    gait_period : float — trot cycle period [s]
    duty        : float — trot stance fraction
    params      : Go2Params — physical + tuning parameters
    """

    def __init__(self, xml: str = XML_PATH,
                 K: int = 10, mpc_dt: float = 0.030,
                 gait_period: float = 0.40, duty: float = 0.50,
                 params: Go2Params = GO2):

        print(f"\nLoading model: {xml}")
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data  = mujoco.MjData(self.model)

        self.ctrl  = Go2MPCController(
            self.model, self.data,
            K=K, mpc_dt=mpc_dt,
            gait_period=gait_period, duty=duty,
            params=params,
        )

        print(f"Physics dt = {self.model.opt.timestep*1e3:.1f} ms  "
              f"nv={self.model.nv}  nq={self.model.nq}  nu={self.model.nu}")
        self._log: list = []   # lightweight run summary rows

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        import os
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        """Reset MuJoCo state and controller internal state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        c = self.ctrl
        c._last_mpc      = -999.0
        c._mpc_active    = False
        c._trot_active   = False
        c._pz_settled    = None
        c._prev_contacts = np.ones(4, dtype=bool)
        c._solve_ms.clear()
        c.mpc._U_prev    = None
        c._phase         = "A"
        c._recovery_mode = False
        c._pz_integral   = 0.0
        c._pz_last_t     = 0.0
        c._warmup_done   = False
        c._phase_c_start = None
        c._phase_b_start = None
        # BUG FIX: re-apply gravity pre-warm on reset so runs after the first
        # don't start from the stale end-state _grf of the previous run.
        _fz0 = c.p.mass * c.p.g / 4.0
        c._grf = np.array([[0., 0., _fz0]] * 4, dtype=float)
        # Reset contact smoother buffer
        c.estimator._cf_buf[:] = 0.0
        c.estimator._cf_idx    = 0
        self._log.clear()

    # ── Command interface ──────────────────────────────────────────────────────

    def set_cmd(self, vx: float = 0., vy: float = 0.,
                wz: float = 0., pz: float = None) -> None:
        """Update velocity commands. pz=None keeps current height setpoint."""
        c = self.ctrl
        c.cmd_vx = vx
        c.cmd_vy = vy
        c.cmd_wz = wz
        if pz is not None:
            c.pz_des = pz

    # ── Console table helpers ─────────────────────────────────────────────────

    def _print_header(self) -> None:
        hdr = (f"  {'t':>5}  {'pz':>5}  "
               f"{'roll':>6} {'pitch':>6} {'yaw':>6}  "
               f"{'vx':>5} {'vy':>5}  "
               f"{'fzFL':>6} {'fzFR':>6} {'fzRL':>6} {'fzRR':>6}  "
               f"contacts  phase")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

    def _print_row(self, s: RobotState) -> None:
        r, p, y = np.degrees(s.euler)
        g = self.ctrl.grf
        c = "  ".join(f"{n}:{'Y' if v else 'N'}"
                      for n, v in zip(LEG_NAMES, s.contact))
        row = (f"  {s.t:5.2f}  {s.pos[2]:5.3f}  "
               f"{r:6.1f} {p:6.1f} {y:6.1f}  "
               f"{s.vel[0]:5.2f} {s.vel[1]:5.2f}  "
               f"{g[0,2]:6.1f} {g[1,2]:6.1f} {g[2,2]:6.1f} {g[3,2]:6.1f}  "
               f"{c}  {self.ctrl._phase}")
        print(row)
        self._log.append((s.t, s.pos[2], r, p, y, s.vel[0]))

    # ── Run modes ─────────────────────────────────────────────────────────────

    def verify(self, duration: float = 15.0) -> None:
        """
        Headless run with console output.

        Ramps forward velocity: 0 m/s → 0.2 → 0.4 → 0.6 m/s.
        Phase timings (after fixes):
          Phase A: 0 – 2.0s   (PD settle)
          Phase B: 2.0 – 5.0s (all-stance MPC)
          Phase C: 5.0s+      (trot, once pz ≥ 0.30m)
        Velocity commands start at t=7s to give Phase C 2s to stabilise.
        """
        self.reset()
        self.set_cmd()   # start with zero velocity

        print("\n" + "=" * 74)
        print("  PHASES 2–6: Convex MPC Trot  (Di Carlo et al., IROS 2018)")
        print("=" * 74)
        self._print_header()

        # Velocity profile shifted to match new Phase timing:
        # Phase A=1.5s, Phase B=2.0s → trot starts at t=3.5s.
        # Wait 2s for trot to stabilise before commanding velocity (t=5.5s).
        profile = [
            (0.,   dict(vx=0.0)),
            (5.5,  dict(vx=0.2)),
            (8.0,  dict(vx=0.4)),
            (11.0, dict(vx=0.6)),
        ]
        ci      = 0
        last_pr = -1.0

        while self.data.time < duration:
            t = self.data.time
            while ci < len(profile) and t >= profile[ci][0]:
                self.set_cmd(**profile[ci][1])
                print(f"\n  ── t={t:.1f}s  cmd={profile[ci][1]}")
                ci += 1

            state = self.ctrl.step()

            if t - last_pr >= 0.2:
                self._print_row(state)
                last_pr = t

        self._print_verify_summary()
        self.ctrl.logger.close()

    def run_with_viewer(self, duration: float = 28.0) -> None:
        """
        Interactive MuJoCo passive viewer run.

        Ramps velocity more aggressively to demonstrate full trot.
        Close the viewer window or press Ctrl-C to stop early.
        """
        self.reset()
        self.set_cmd()

        # Shifted for new Phase timing (trot at 3.5s, wait 2s for stabilisation)
        profile = [
            (0.,   dict(vx=0.0,  wz=0.0)),
            (5.5,  dict(vx=0.2,  wz=0.0)),
            (8.0,  dict(vx=0.4,  wz=0.0)),
            (12.0, dict(vx=0.4,  wz=0.3)),   # turning test
            (16.0, dict(vx=0.6,  wz=0.0)),
            (20.0, dict(vx=0.8,  wz=0.0)),
        ]
        ci = 0

        hdr = (f"{'t':>6} {'pz':>5}  "
               f"{'roll':>5} {'pitch':>5} {'yaw':>5}  "
               f"{'vx':>5} {'vy':>5}  "
               f"{'fzFL':>6} {'fzFR':>6} {'fzRL':>6} {'fzRR':>6}  "
               f"contacts  phase")
        print(hdr)
        print("-" * len(hdr))

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # ── Chase camera: follow robot base ───────────────────────────
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = 1      # base_link (from config.py)
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -20
            last_p = -1.0
            while viewer.is_running() and self.data.time < duration:
                t = self.data.time
                while ci < len(profile) and t >= profile[ci][0]:
                    self.set_cmd(**profile[ci][1])
                    print(f"── t={t:.1f}s  {profile[ci][1]}")
                    ci += 1

                state = self.ctrl.step()

                if t - last_p >= 0.2:
                    r, p, y = np.degrees(state.euler)
                    g       = self.ctrl.grf
                    c = " ".join(f"{n}:{'Y' if v else 'N'}"
                                 for n, v in zip(LEG_NAMES, state.contact))
                    print(f"{t:6.2f} {state.pos[2]:5.3f}  "
                          f"{r:5.1f} {p:5.1f} {y:5.1f}  "
                          f"{state.vel[0]:5.2f} {state.vel[1]:5.2f}  "
                          f"{g[0,2]:6.1f} {g[1,2]:6.1f} "
                          f"{g[2,2]:6.1f} {g[3,2]:6.1f}  "
                          f"{c}  {self.ctrl._phase}")
                    last_p = t

                viewer.sync()

        st = self.ctrl.stats()
        print(f"\nSolves={st.get('n', 0)}  "
              f"mean={st.get('mean_ms', 0):.1f}ms  "
              f"QP fails={st.get('fails', 0)}")
        self.ctrl.logger.close()

    # ── Summary ───────────────────────────────────────────────────────────────

    def _print_verify_summary(self) -> None:
        st = self.ctrl.stats()
        print("\n" + "=" * 74)
        print(f"  MPC solves : {st.get('n', 0)}  "
              f"mean={st.get('mean_ms', 0):.1f}ms  "
              f"max={st.get('max_ms', 0):.1f}ms  "
              f"QP fails={st.get('fails', 0)}")
        if self._log:
            pz = [e[1] for e in self._log]
            print(f"  Height     : mean={np.mean(pz):.3f}  "
                  f"min={np.min(pz):.3f}  max={np.max(pz):.3f} m")
        print("=" * 74)
