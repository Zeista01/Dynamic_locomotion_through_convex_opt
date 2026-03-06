# Dynamic Locomotion via Convex MPC — Unitree Go2 in MuJoCo

Implementation of centroidal dynamics-based convex QP MPC for dynamic quadruped locomotion, based on:

> **"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"**
> Jared Di Carlo, Patrick M. Wensing, Benjamin Katz, Gerardo Bledt, Sangbae Kim — *IROS 2018*

Deployed on the **Unitree Go2** quadruped in **MuJoCo**, achieving:
- Stable **diagonal trot** on flat ground at up to 0.6 m/s
- **Staircase ascent and descent** (6 cm riser / 28 cm tread, 5 steps) with terrain-adaptive height control

---

## Results

| Scenario | Description |
|---|---|
| Flat trot | Stable diagonal trot, velocity ramp 0 → 0.6 m/s |
| Stair climb | 5-step ascent + descent, adaptive CoM height and pitch |

*See [`docs/`](docs/) for detailed plots and analysis.*

---

## Mathematical Overview

The controller solves a receding-horizon **convex QP** over a simplified single rigid-body model of the robot.

### State and Dynamics

The MPC state is the 13-dimensional centroidal state:

$$
\mathbf{x} = [\phi,\ \theta,\ \psi,\ p_x,\ p_y,\ p_z,\ \omega_x,\ \omega_y,\ \omega_z,\ v_x,\ v_y,\ v_z,\ {-g}]^\top \in \mathbb{R}^{13}
$$

The continuous-time linearised dynamics are:

$$
\dot{\mathbf{x}} = A_c(\psi)\,\mathbf{x} + B_c(\mathbf{r}_i, \psi, \mathbf{c})\,\mathbf{u}
$$

where $\mathbf{u} = [f_1^\top,\ldots,f_4^\top]^\top \in \mathbb{R}^{12}$ are the ground reaction forces (GRFs), and the $B$ matrix encodes:

$$
B_c \ni \begin{bmatrix} \hat{I}^{-1}(\psi)\,[r_i]_\times \\ \frac{1}{m}\,I_3 \end{bmatrix}
\quad \text{for each stance leg } i
$$

### Condensed QP

After zero-order-hold discretisation and horizon condensing, the QP is:

$$
\min_{\mathbf{U}} \quad \frac{1}{2}\mathbf{U}^\top H\,\mathbf{U} + \mathbf{g}^\top \mathbf{U}
$$

$$
H = 2(B_{qp}^\top L\, B_{qp} + \alpha I), \qquad \mathbf{g} = 2\,B_{qp}^\top L\,(A_{qp}\mathbf{x}_0 - \mathbf{X}_{ref})
$$

subject to linearised pyramid friction constraints for each stance foot:

$$
f_{z,\min} \leq f_z \leq f_{z,\max}, \qquad |f_x| \leq \mu f_z, \qquad |f_y| \leq \mu f_z
$$

and zero-force equality constraints for swing legs.

### Torque Mapping

Joint torques are recovered from MPC GRFs via the foot Jacobian:

$$
\boldsymbol{\tau} = -J_i^\top\, f_i + (C\dot{q} + g)_{\text{joints}}
$$

The negative sign follows the MIT Cheetah reference: GRFs are ground-on-foot upward forces, so the joint must push downward.

### Swing Trajectory

Swing legs follow a **minimum-jerk** trajectory with a cubic height bump:

$$
z_{\text{bump}}(s) = h \cdot 64s^3(1-s)^3, \qquad s \in [0,1]
$$

Touchdown position is predicted by the **Raibert heuristic**:

$$
p_{\text{td}} = p_{\text{hip}} + v_{\text{cmd}}\,t_{\text{pred}} + k_v(v_{\text{actual}} - v_{\text{cmd}})
$$

---

## System Architecture

```
flat_trot.py / stairs_climb.py          ← entry points
       │
       ▼
mpc_controller.py  (Go2MPCController)   ← top-level state machine
  ├── state_estimator.py                ← Phase 1: state from MuJoCo
  ├── dynamics.py                       ← Phase 2: A_c, B_c, ZOH
  ├── mpc_solver.py                     ← Phase 3: condensed QP
  ├── gait_scheduler.py                 ← Phase 4: contact schedule
  ├── swing_controller.py               ← Phase 5: swing leg PD + Raibert
  ├── torque_utils.py                   ← Phase 6: GRF→τ, logger
  ├── support_plane.py                  ← SVD plane fit (stairs)
  └── terrain_estimator.py              ← height map + pitch ref (stairs)
```

### Controller Phase State Machine

```
t = 0 s       ┌───────────────────────────────────────────┐
              │  Phase A — Joint-space PD settling         │
t = 1.5 s     ├───────────────────────────────────────────┤
              │  Phase B — All-stance MPC warmup           │
t = 3.5 s     ├───────────────────────────────────────────┤
              │  Phase C — Trot MPC + swing controller     │
              │  (active for the rest of simulation)       │
              └───────────────────────────────────────────┘

If pz drops below COLLAPSE_HEIGHT (0.20 m) → RECOVERY (Phase A restart)
```

---

## Repository Layout

```
MPC_CBF_CC/
├── flat_trot.py            Entry point — flat ground trot
├── stairs_climb.py         Entry point — staircase traversal
├── mpc_controller.py       Top-level controller & phase logic
├── dynamics.py             Centroidal A/B matrices, ZOH discretisation
├── mpc_solver.py           Condensed QP builder + quadprog/OSQP solver
├── gait_scheduler.py       Trot contact schedule (phase offsets)
├── swing_controller.py     Swing leg PD + minimum-jerk arc + Raibert
├── state_estimator.py      MuJoCo ground-truth state reader
├── torque_utils.py         GRF→τ, PDStand, reference builder, CSV logger
├── support_plane.py        SVD support-plane fit (terrain mode)
├── terrain_estimator.py    Stair height map, pitch reference, pz target
├── robot_params.py         Go2 physical parameters, Q weights, RobotState
├── simulation.py           MuJoCo simulation harness
├── config.py               All hardware constants, IDs, paths
├── plot_flat_trot.py       Response plots — flat trot scenario
├── plot_stairs.py          Response plots — stair scenario
├── data/                   CSV debug logs (auto-generated on each run)
├── results/                Saved plots and videos
└── docs/                   Theory, design, and results documentation
    ├── theory.md           Full mathematical derivation
    ├── system_design.md    Engineering manual
    └── stair_climbing.md   Terrain-adaptive extension
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mujoco` | Physics simulation |
| `numpy`, `scipy` | Numerical linear algebra, matrix exponential |
| `quadprog` | Primary QP solver (preferred) |
| `osqp` | Fallback QP solver |
| `matplotlib` | Result plots |
| `pandas` | CSV log analysis |

Install:

```bash
pip install mujoco numpy scipy quadprog osqp matplotlib pandas
```

> **quadprog** is preferred — it is faster and has lower overhead than OSQP for small QPs. If it is unavailable, OSQP is used automatically.

---

## How to Run

### 1. Set the XML path

In [config.py](config.py), set `XML_PATH` to point to your Go2 MuJoCo scene:

```python
XML_PATH = "/path/to/unitree_mujoco/unitree_robots/go2/scene.xml"
```

For stairs, set `STAIRS_XML` in [stairs_climb.py](stairs_climb.py):

```python
STAIRS_XML = "/path/to/unitree_robots/go2/scene_stairs_uniform.xml"
```

### 2. Flat-ground trot

```bash
cd MPC_CBF_CC
python flat_trot.py
```

The MuJoCo viewer opens. The robot:
1. Settles from rest (Phase A, ~1.5 s)
2. Activates all-stance MPC (Phase B, ~2 s)
3. Begins trotting at 0.0 → 0.6 m/s (Phase C)

### 3. Staircase climbing

```bash
python stairs_climb.py
```

The robot trots on flat ground then transitions to terrain-aware mode when it crosses `STAIRS_APPROACH_X`. It ascends, traverses the top platform, and descends the mirror staircase.

### 4. Plot results

```bash
python plot_flat_trot.py    # flat trot diagnostic plots
python plot_stairs.py       # staircase scenario plots
```

Plots include: CoM height tracking, body orientation, GRF distribution, contact schedule, velocity tracking, and weight ratio.

---

## Key Tuning Parameters

All tuning lives in [robot_params.py](robot_params.py) and [config.py](config.py).

| Parameter | Location | Default | Description |
|---|---|---|---|
| `Q` (13-vector) | `robot_params.py` | see file | State cost weights |
| `alpha` | `robot_params.py` | `3e-4` | GRF regularisation |
| `mu` | `robot_params.py` | `0.7` | Friction coefficient |
| `f_min / f_max` | `robot_params.py` | `5 / 200 N` | GRF bounds |
| `NOMINAL_HEIGHT` | `config.py` | `0.32 m` | Target CoM height |
| `TORQUE_LIMIT` | `config.py` | `33.5 Nm` | Per-joint torque limit |
| `gait_period` | `flat_trot.py` | `0.40 s` | Trot cycle period |
| `duty` | `flat_trot.py` | `0.50` | Stance fraction |
| `K` | `flat_trot.py` | `10` | MPC horizon steps |
| `mpc_dt` | `flat_trot.py` | `0.030 s` | MPC timestep |

---

## Debugging

Enable verbose logging by setting `DEBUG_LEVEL` in [config.py](config.py):

```python
DEBUG_LEVEL = 2   # phase transitions + key events (recommended)
DEBUG_LEVEL = 3   # every MPC solve
DEBUG_LEVEL = 4   # every control step (very verbose)
```

CSV logs are written to `data/mpc_debug_log.csv` on every run (when `LOG_CSV = True`).

Key columns to inspect:

| Column | Description |
|---|---|
| `pz` vs `pz_error` | CoM height tracking |
| `weight_ratio` | ΣFz / mg — should be ≈ 1.0 in steady state |
| `contact_FL/FR/RL/RR` | Gait-scheduled contacts (1 = stance, 0 = swing) |
| `pitch_deg` | Body pitch — should track terrain slope on stairs |
| `phase` | Current controller phase (A_PD / B / C / RECOVERY) |

---

## Documentation

| Document | Contents |
|---|---|
| [docs/theory.md](docs/theory.md) | Full centroidal dynamics derivation, QP condensing, friction cones, swing trajectory, support-plane math |
| [docs/system_design.md](docs/system_design.md) | Phase state machine, module responsibilities, data flow, tuning guide |
| [docs/stair_climbing.md](docs/stair_climbing.md) | Terrain-adaptive extensions: height estimation, pitch reference, support-plane integration, swing clearance |

---

## Reference

```bibtex
@inproceedings{diCarlo2018,
  title     = {Dynamic Locomotion in the {MIT} Cheetah 3 Through
               Convex Model-Predictive Control},
  author    = {Di Carlo, Jared and Wensing, Patrick M. and Katz, Benjamin
               and Bledt, Gerardo and Kim, Sangbae},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots
               and Systems (IROS)},
  year      = {2018},
}
```
