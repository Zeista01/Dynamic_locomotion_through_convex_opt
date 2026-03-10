"""
stairs_climb.py — Stair-Climbing Entry Point
============================================

Runs the SAME convex MPC controller as flat_trot.py with
an important architectural change:

  Phase 1 (flat ground, x < STAIRS_APPROACH_X):
      terrain_mode = "flat"  — IDENTICAL to the working flat-ground trot.
      The controller is completely unchanged from phase2_6_mpc_go2_complete.py.

  Phase 2 (approaching stairs, x >= STAIRS_APPROACH_X = 1.20 m):
      ctrl.switch_to_terrain_mode("stairs") is called ONCE.
      From this point the terrain estimator begins tracking stance foot z.
      Three additions activate:
        1. Adaptive pz_des  — tracks terrain_z + nominal_clearance.
        2. Terrain-aware swing — larger arc clearance (8 cm → 18 cm),
                                 touchdown z from terrain estimate.
        3. Early touchdown   — swing foot frozen if step contact detected early.

Scene
-----
scene_stairs_uniform.xml  — 5 uniform steps, 6 cm riser × 28 cm tread, 2 m wide.
Stairs start at x = 2.0 m.  Robot starts at x = 1.6 m.
STAIRS_APPROACH_X = 1.20 m → terrain mode activates ~0.8 m before first step.

Command profile
---------------
t = 0-5.5 s   : vx = 0    (Phase A PD settle -> Phase B MPC -> Phase C trot)
t = 5.5 s+    : vx = 0.03 m/s  forward (slow approach for stable stair climbing)
"""

import sys
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np
import mujoco

from simulation import Go2Simulation, VideoRecorder

# The stairs scene must live alongside go2.xml so MuJoCo resolves
# mesh paths (base_0.obj etc.) relative to the robot's own directory.
STAIRS_XML = "/home/stanny/unitree_ws/unitree_mujoco/unitree_robots/go2/scene_stairs_uniform.xml"

# Activate terrain-aware mode when robot x-position crosses this threshold.
# FIX: moved from 1.5 to 1.0 m — gives a full 1.0 m (5+ gait cycles at
# 0.20 m/s) for the terrain estimator, swing clearance, and stair Q
# weights to settle before the robot reaches the stairs at x=2.0m.
# FIX: STAIRS_APPROACH_X 1.65→1.20.  Debug log shows the robot crosses
# x=1.65 nearly simultaneously with the vx command at t=5.5, leaving <1s
# for the stairs_Q weights to settle before stair contact.  At x=1.20,
# terrain mode activates ~2.7s before the front feet reach x=2.0
# (at vx=0.03 m/s from x=1.2: (2.0-1.2)/0.03 ≈ 27s, but with drift and
# approach, effectively 5-10 gait cycles of Q settling time).
STAIRS_APPROACH_X = 1.80  # m
# FIX: raised from 1.20 → 1.80m.
# Robot spawns at x=1.60m. With the old threshold=1.20, terrain mode
# fired immediately at trot-ready (~t=3.8s) because x=1.60 > 1.20 already.
# The swing_clearance jumped to 0.18m on flat ground, causing +9°–+11°
# pitch oscillation for the entire 17-second flat approach.
# With 1.80m, terrain mode fires when x≥1.80 — only 0.20m before stairs.
# At vx=0.04 m/s that's 5 seconds of warm-up with flat parameters (0.08m
# clearance, flat-Q), followed by gradual switch. The adaptive clearance
# fix in terrain_estimator.py further ensures clearance starts at 0.09m
# (not 0.18m) when terrain_z≈0.


# ── Diagnostic helper ────────────────────────────────────────────────────────

def diag_print(t, state, ctrl, terrain_on):
    """Print a detailed diagnostic line for debugging."""
    r, p, y = np.degrees(state.euler)
    x       = state.pos[0]
    pz      = state.pos[2]
    vx      = state.vel[0]
    vz      = state.vel[2]
    g       = ctrl.grf
    te      = ctrl.terrain

    # Gait schedule: which legs does the gait say should be in stance?
    if ctrl._trot_active:
        sched = ctrl.gait.contact_at(t)
        sched_str = "".join("S" if s else "W" for s in sched)
    else:
        sched_str = "SSSS"

    # Contact forces (from rolling-max buffer)
    cf_str = " ".join(f"{state.cf[i]:6.1f}" for i in range(4))

    # MPC GRF z-components
    fz_str = " ".join(f"{g[i,2]:6.1f}" for i in range(4))

    # Terrain info
    mode    = te.mode
    tz      = te.terrain_z
    td_z    = te.touchdown_z()
    sw_cl   = te.swing_clearance()
    pz_des  = ctrl.pz_des

    # Early touchdown flags from swing controller
    et = ctrl.swing._early_touchdown
    et_str = "".join("T" if et[i] else "." for i in range(4))

    # Flags
    flags = []
    if abs(p) > 3.0:
        flags.append(f"PITCH={p:+.1f}")
    if abs(r) > 3.0:
        flags.append(f"ROLL={r:+.1f}")
    if pz < 0.28:
        flags.append("PZ_LOW!")
    if pz < 0.25:
        flags.append("COLLAPSING!")
    total_fz = g[:, 2].sum()
    weight   = ctrl.p.mass * ctrl.p.g
    wr       = total_fz / weight
    if wr < 0.7:
        flags.append(f"UNDERFORCE={wr:.2f}")
    if wr > 1.4:
        flags.append(f"OVERFORCE={wr:.2f}")
    flag_str = "  ".join(flags) if flags else "OK"

    print(f"  {t:6.2f}  x={x:+.2f}  pz={pz:.3f}/{pz_des:.3f}  "
          f"rpy=[{r:+5.1f},{p:+5.1f},{y:+5.1f}]  "
          f"vx={vx:+.2f} vz={vz:+.3f}  "
          f"sched={sched_str}  "
          f"cf=[{cf_str}]  "
          f"fz=[{fz_str}]  "
          f"wr={wr:.2f}  "
          f"mode={mode} tz={tz:.3f} td_z={td_z:.3f} sw_cl={sw_cl:.2f}  "
          f"et={et_str}  "
          f"ph={ctrl._phase}  "
          f"[{flag_str}]")


def diag_header():
    """Print header for diagnostic output."""
    print("\n" + "=" * 180)
    print("  DIAGNOSTIC TABLE")
    print("  Columns: t | x | pz/pz_des | rpy | vx vz | sched(S=stance,W=swing) | "
          "cf[FL FR RL RR] | fz[FL FR RL RR] | wr(weight_ratio) | "
          "terrain(mode tz td_z sw_cl) | et(early_td) | phase | flags")
    print("=" * 180)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Start with duty=0.60 to ensure some 4-foot overlap during diagonal
    # transitions (20% overlap = 0.08s).  duty=0.50 had zero overlap,
    # causing zero-contact moments that crashed at low speed.
    # Switch to stairs gait (T=0.60/duty=0.85) at terrain switch.
    sim = Go2Simulation(
        xml          = STAIRS_XML,
        K            = 25,        # 25 steps x 25 ms = 625 ms lookahead
        mpc_dt       = 0.025,
        gait_period  = 0.40,      # 2.5 Hz trot
        duty         = 0.60,      # 60% stance — 20% 4-foot overlap
        terrain_mode = "stairs",  # stored as target; controller always starts flat
    )

    print("\n" + "=" * 74)
    print("  Stair-Climbing: Convex MPC + Terrain Estimator (late-activate)")
    print("  Scene:", STAIRS_XML)
    print("  Steps: 5 x 6 cm riser / 28 cm tread  (total rise = 30 cm)")
    print(f"  Terrain mode activates at x = {STAIRS_APPROACH_X} m  "
          f"(stairs start at x = 2.0 m)")
    print("=" * 74)

    # ── Headless verification ────────────────────────────────────────────────
    # FIX: 0.30 m/s was too fast — the robot had <1s per step tread,
    # not enough time for the MPC to adjust height/pitch per step.
    # At 0.20 m/s each 28cm tread takes ~1.4s (3.5 gait cycles),
    # giving the MPC sufficient time to converge on each step.
    # FIX 4: Reduced approach speed 0.20 → 0.15 m/s.
    # At 0.20 m/s each 28 cm tread lasts ~1.4 s (3.5 gait cycles).
    # At 0.15 m/s it lasts ~1.9 s (4.7 gait cycles), giving the MPC
    # roughly 35% more time per tread to converge on height/pitch changes.
    # This is especially important on the first riser where the terrain
    # estimator is still warming up.
    # Command profile:
    #   0-5.5s   : vx=0.0  (Phase A PD settle → Phase B MPC → Phase C trot warmup)
    #   5.5s+    : vx=0.20 m/s
    #
    # FIX (tertiary root cause — vx too slow):
    # The previous vx=0.04 m/s was set to avoid pitch oscillation, but the
    # oscillation was caused by the pitch_ref/target_pz bugs above, NOT by speed.
    # At 0.04 m/s the robot spent ~30 s in the approach with +12-14° pitch,
    # which was itself the failure mode (not the speed).
    #
    # With the pitch_ref and target_pz fixes applied, the flat approach is stable
    # at any reasonable trot speed.  We restore 0.20 m/s:
    #   • Each 28 cm tread lasts 0.28/0.20 = 1.4 s = 2.8 gait cycles → sufficient
    #     MPC settling time per step (same calculation that was already in comments).
    #   • At vx=0.04 the robot barely moved (actual vx 0.04-0.08 m/s over 30 s).
    #     Momentum assists balance: a moving trot is MORE stable than a near-static
    #     one because the stance-swap inertia damps roll/pitch perturbations.
    #   • The Raibert foothold heuristic was designed for normal trot speeds.
    #     At near-zero speed, the foothold defaults to directly below the hip,
    #     maximising roll/pitch sensitivity to foot height asymmetry.
    # Two-stage speed profile:
    #   Fast (0.20 m/s) on flat approach — good trot dynamics, stable.
    #   Slow (0.10 m/s) on stairs — each 28cm tread takes 2.8s (5+ gait cycles),
    #   giving the Raibert heuristic time to recover from riser impacts.
    #   At 0.20 m/s a riser hit creates a lateral impulse that builds vy>0.5 m/s
    #   before the stance swap can arrest it → roll divergence → crash (confirmed
    #   in log: vy→1.0 m/s, roll→135° at t=13.5s, x=2.55m on step 2).
    #   t=8.0s: robot is at x≈1.90m (0.10m before stairs) at 0.20 m/s.
    #   Total stair time at 0.10 m/s: 1.4m ascend + 1.0m platform + 1.4m descend
    #   = 38s from t≈9s → done by t≈47s (well within 70s window).
    # FIX: constant slow speed throughout. The previous 0.20→0.10 deceleration
    # at t=8.0 coincided exactly with stair contact (x=1.99), creating a combined
    # pitch impulse (deceleration torque + step impact) that drove pitch to +10°.
    # At vx=0.08 each 28cm tread lasts 3.5s (7 gait cycles) — ample time for
    # the MPC to converge height/pitch per step.
    # vx=0.10: minimum stable trot speed with duty=0.50.  Below this,
    # diagonal transitions have zero-contact moments that are unstable.
    # At 0.10 m/s each 28cm tread takes 2.8s = 4.7 gait cycles.
    profile = [
        (0.0,  dict(vx=0.0)),
        (5.5,  dict(vx=0.10)),
    ]

    sim.reset()
    sim.set_cmd(vx=0.0)
    sim.data.qpos[0] = 1.6     # base x-position
    sim.data.qvel[:] = 0.0     # zero velocities for clean start
    mujoco.mj_forward(sim.model, sim.data)


    # Verify terrain mode is flat after reset
    print(f"\n[CHECK] After reset: terrain.mode = '{sim.ctrl.terrain.mode}'  "
          f"(MUST be 'flat')")
    print(f"[CHECK] swing_clearance = {sim.ctrl.terrain.swing_clearance():.3f}m  "
          f"(MUST be 0.080 for flat)")
    print(f"[CHECK] touchdown_z = {sim.ctrl.terrain.touchdown_z():.3f}m  "
          f"(MUST be 0.020 for flat)")
    assert sim.ctrl.terrain.is_flat(), \
        "BUG: terrain mode is NOT flat after reset!"

    ci          = 0
    last_pr     = -1.0
    terrain_on  = False
    # Duration must be long enough for full ascend + platform + descend.
    # At vx_cmd=0.10 (effective ~0.07 m/s), distance = 4.2m (x=1.6→5.8),
    # time ≈ 60s + 9s approach = 69s.  Use 85s for margin.
    duration    = 85.0

    diag_header()

    # ── Video recorder (headless MP4) ──────────────────────────────────────
    os.makedirs("results/stairs", exist_ok=True)
    rec = sim.make_recorder("results/stairs/stairs_climb.mp4", fps=30,
                            cam_distance=3.5, cam_azimuth=160, cam_elevation=-18)

    while sim.data.time < duration:
        t     = sim.data.time
        state = sim.ctrl.step()
        x     = state.pos[0]
        rec.capture(t)

        # Apply velocity commands from profile
        while ci < len(profile) and t >= profile[ci][0]:
            sim.set_cmd(**profile[ci][1])
            print(f"\n  >> t={t:.1f}s  CMD CHANGE: {profile[ci][1]}  "
                  f"terrain_mode={sim.ctrl.terrain.mode}  "
                  f"terrain_z={sim.ctrl.terrain.terrain_z:.3f}m")
            ci += 1

        # Switch to terrain-aware mode once, when robot approaches stairs AND
        # Phase C trot is already stable.
        #
        # BUG FIX (premature terrain switch): The robot spawns at x≈1.60m,
        # which is already > STAIRS_APPROACH_X=1.0m.  Without the trot guard,
        # the terrain switch fires at t=0 BEFORE Phase A settling, causing:
        #   • nominal_clearance = NOMINAL_HEIGHT=0.320 (not settled 0.338)
        #   • pz_des drifts DOWN from 0.338 to 0.315 throughout Phase B
        #   • GRF_ratio stuck at 0.91 (9% underforce) — confirmed in log
        #   • _stairs_Q active during settling, creating pitch oscillations
        # Waiting for _trot_active ensures Phase B has converged and Phase C
        # warmup has run before we activate terrain-aware behaviour.
        trot_ready = sim.ctrl._trot_active and sim.ctrl._warmup_done
        if not terrain_on and trot_ready and x >= STAIRS_APPROACH_X:
            print(f"\n  >> TERRAIN SWITCH at t={t:.2f}s  x={x:.2f}m  "
                  f"(trot stable, pz={state.pos[2]:.3f}m)")
            sim.ctrl.switch_to_terrain_mode("stairs")
            terrain_on = True
            # Switch gait from flat-stable (T=0.40/duty=0.50) to stairs
            # gait (T=0.60/duty=0.85).  High duty gives 70% 4-foot overlap
            # for roll/pitch stability on asymmetric step heights.
            sim.ctrl.gait.T    = 0.60
            sim.ctrl.gait.duty = 0.85
            sim.ctrl.gait.activate(t)  # re-anchor phase to current time
            print(f"     gait: T=0.40/duty=0.60 → T=0.60/duty=0.85")
            print(f"     swing_clearance: 0.08 -> {sim.ctrl.terrain.swing_clearance():.2f}m")
            print(f"     touchdown_z: 0.02 -> {sim.ctrl.terrain.touchdown_z():.3f}m")

        # Print diagnostic every 0.5s (less noisy than 0.2s)
        if t - last_pr >= 0.5:
            diag_print(t, state, sim.ctrl, terrain_on)
            last_pr = t

        # Early warning: detect failure in progress
        # FIX: on stairs the CoM naturally rises (up to +30cm at step 5),
        # so check height RELATIVE to terrain, not absolute.
        # Also allow more pitch (±25°) since the terrain slope is ~9°.
        pz = state.pos[2]
        terrain_z = sim.ctrl.terrain.stair_height(x) if terrain_on else 0.0
        pz_above_terrain = pz - terrain_z
        pitch = abs(np.degrees(state.euler[1]))
        if pz_above_terrain < 0.20 and t > 2.0:
            print(f"\n  !! FAILURE: pz={pz:.3f}m  terrain_z={terrain_z:.3f}m  "
                  f"clearance={pz_above_terrain:.3f}m < 0.20m at t={t:.2f}s  "
                  f"pitch={pitch:.1f} — robot is collapsing!")
            break
        if pitch > 25.0 and t > 4.0:
            print(f"\n  !! FAILURE: pitch={pitch:.1f} > 25 at t={t:.2f}s  "
                  f"pz={pz:.3f}m — robot is toppling!")
            break
        # FIX 6: detect roll failure (the primary crash mode on stairs).
        # Debug log showed roll diverged to -22.4° before crash — undetected
        # because only pitch was checked.
        roll = abs(np.degrees(state.euler[0]))
        if roll > 35.0 and t > 4.0:
            print(f"\n  !! FAILURE: roll={roll:.1f} > 35 at t={t:.2f}s  "
                  f"pz={pz:.3f}m — robot is rolling over!")
            break

    rec.close(sim_time=sim.data.time)

    sim._print_verify_summary()
    sim.ctrl.logger.close()

    # ── Interactive viewer ───────────────────────────────────────────────────
    print("\nLaunching interactive viewer (Ctrl-C or close window to stop)...")
    try:
        sim.reset()
        # Spawn near stairs for viewer run
        sim.data.qpos[0] = 1.6
        sim.data.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)

        # Verify terrain mode is flat after reset (this was the #1 bug)
        print(f"\n[CHECK] Viewer run: terrain.mode = '{sim.ctrl.terrain.mode}'  "
              f"(MUST be 'flat')")
        print(f"[CHECK] swing_clearance = {sim.ctrl.terrain.swing_clearance():.3f}m")
        print(f"[CHECK] touchdown_z = {sim.ctrl.terrain.touchdown_z():.3f}m")
        assert sim.ctrl.terrain.is_flat(), \
            "BUG: terrain mode is NOT flat after reset for viewer run!"

        import mujoco.viewer

        profile_v = [
            (0.0,  dict(vx=0.0)),
            (5.5,  dict(vx=0.10)),
        ]
        ci2         = 0
        terrain_on2 = False

        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = 1
            viewer.cam.distance    = 3.5
            viewer.cam.azimuth     = 160
            viewer.cam.elevation   = -18
            last_p = -1.0

            while viewer.is_running() and sim.data.time < duration:
                t     = sim.data.time
                state = sim.ctrl.step()
                x     = state.pos[0]

                while ci2 < len(profile_v) and t >= profile_v[ci2][0]:
                    sim.set_cmd(**profile_v[ci2][1])
                    print(f">> t={t:.1f}s  CMD: {profile_v[ci2][1]}  "
                          f"terrain_mode={sim.ctrl.terrain.mode}  "
                          f"terrain_z={sim.ctrl.terrain.terrain_z:.3f}m")
                    ci2 += 1

                trot_ready2 = sim.ctrl._trot_active and sim.ctrl._warmup_done
                if not terrain_on2 and trot_ready2 and x >= STAIRS_APPROACH_X:
                    print(f"\n>> TERRAIN SWITCH at t={t:.2f}s  x={x:.2f}m  "
                          f"(trot stable, pz={state.pos[2]:.3f}m)")
                    sim.ctrl.switch_to_terrain_mode("stairs")
                    terrain_on2 = True
                    sim.ctrl.gait.T    = 0.60
                    sim.ctrl.gait.duty = 0.85
                    sim.ctrl.gait.activate(t)

                if t - last_p >= 0.5:
                    diag_print(t, state, sim.ctrl, terrain_on2)
                    last_p = t

                viewer.sync()

        st = sim.ctrl.stats()
        print(f"\nSolves={st.get('n',0)}  "
              f"mean={st.get('mean_ms',0):.1f}ms  "
              f"QP fails={st.get('fails',0)}")
        sim.ctrl.logger.close()

    except KeyboardInterrupt:
        sim.ctrl.logger.close()
        print("\nStopped by user.")
    except Exception as e:
        # Viewer may fail in headless environments (no DISPLAY).
        # The headless simulation run above already completed successfully.
        print(f"\n[INFO] Viewer could not launch ({type(e).__name__}: {e}).")
        print("       Run locally with a display to see the animation.")