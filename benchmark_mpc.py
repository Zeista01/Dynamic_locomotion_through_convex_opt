"""
benchmark_mpc.py — MPC QP Solve-Time Profiler
==============================================

Measures wall-clock time for each component of the MPC pipeline
using the real ConvexMPC solver and Go2 dynamics, without MuJoCo.

Scenarios
---------
  A  All-stance (Phase B / warmup)       — 4 stance feet,  20 constraints/step
  B  Trot 2+2 (Phase C steady-state)     — 2 stance feet,  13 constraints/step
  C  Diagonal sparse (single-leg stance) — 1 stance foot,   8 constraints/step

For each scenario: N_SOLVES QP solves are timed end-to-end:
  build A_d/B_d  →  condense  →  build cost  →  build constraints  →  QP solve

Output
------
  • Console table  (mean / max / P95 / P99 / min, all in ms)
  • results/benchmark_solve_time.png  (3-panel figure)
  • results/benchmark_solve_time_detailed.png  (full breakdown)

Usage
-----
  cd /home/stanny/Dynamic_locomotion_through_convex_opt/cvx_opt
  python benchmark_mpc.py
"""

import time
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.linalg

# ── Import project modules ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from robot_params import Go2Params, GO2
from mpc_solver import ConvexMPC
from dynamics import RigidBodyDynamics

# ── Benchmark configuration ──────────────────────────────────────────────────
K        = 10          # MPC horizon steps
MPC_DT   = 0.030      # s  — 30 ms MPC timestep (matches flat_trot.py)
CTRL_HZ  = 100        # Hz — simulation control loop
N_SOLVES = 2000       # solves per scenario (first 200 discarded as warmup)
WARMUP   = 200        # solves to discard
SEED     = 42

rng = np.random.default_rng(SEED)

# ── Realistic foot positions (trot geometry at nominal stance) ──────────────
# Hip origins from config.py  ± small random perturbation
HIP_ORIGINS = np.array([
    [ 0.1934,  0.1125, 0.0],
    [ 0.1934, -0.1125, 0.0],
    [-0.1934,  0.1125, 0.0],
    [-0.1934, -0.1125, 0.0],
])
FOOT_Z = -0.32   # nominal foot z in body frame (CoM at 0.32 m)


def make_foot_positions(n: int = 1) -> np.ndarray:
    """Return (n, 4, 3) array of perturbed foot positions."""
    base = HIP_ORIGINS.copy()
    base[:, 2] = FOOT_Z
    feet = np.tile(base, (n, 1, 1))
    feet += rng.uniform(-0.03, 0.03, size=feet.shape)
    return feet


def make_state(n: int = 1) -> np.ndarray:
    """Return (n, 13) array of realistic MPC states."""
    x = np.zeros((n, 13))
    x[:, 5] = 0.32 + rng.uniform(-0.01, 0.01, n)   # pz
    x[:, 9] = rng.uniform(0.0, 0.6, n)              # vx
    x[:, 12] = -9.81
    return x


def make_schedule(pattern: str, K: int) -> np.ndarray:
    """
    Build a (K, 4) boolean contact schedule.

    Patterns
    --------
    "all"       — all legs in stance every step
    "trot_fl_rr"— FL+RR in stance (diagonal trot, first half)
    "trot_fr_rl"— FR+RL in stance (diagonal trot, second half)
    "single"    — only FL in stance
    """
    sched = np.zeros((K, 4), dtype=bool)
    if pattern == "all":
        sched[:] = True
    elif pattern == "trot_fl_rr":
        sched[:, 0] = True   # FL
        sched[:, 3] = True   # RR
    elif pattern == "trot_fr_rl":
        sched[:, 1] = True   # FR
        sched[:, 2] = True   # RL
    elif pattern == "single":
        sched[:, 0] = True   # FL only
    return sched


# ── Scenarios ────────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name"   : "All-Stance\n(Phase B / warmup)",
        "label"  : "All-Stance",
        "pattern": "all",
        "color"  : "#2196F3",
    },
    {
        "name"   : "Diagonal Trot 2+2\n(Phase C steady-state)",
        "label"  : "Diagonal Trot",
        "pattern": "trot_fl_rr",
        "color"  : "#4CAF50",
    },
    {
        "name"   : "Single-Leg Stance\n(sparse contact)",
        "label"  : "Single-Leg",
        "pattern": "single",
        "color"  : "#FF9800",
    },
]


def run_scenario(scenario: dict,
                 dyn: RigidBodyDynamics,
                 mpc: ConvexMPC,
                 n_solves: int = N_SOLVES,
                 warmup: int = WARMUP) -> dict:
    """
    Run n_solves timed QP solves and return timing arrays in ms.
    Returns dict with keys: total, condense, cost, constraints, solve.
    """
    pattern = scenario["pattern"]
    schedule = make_schedule(pattern, K)

    foot_arr  = make_foot_positions(n_solves)
    state_arr = make_state(n_solves)
    psi_arr   = rng.uniform(-0.3, 0.3, n_solves)

    t_total       = np.zeros(n_solves)
    t_dynamics    = np.zeros(n_solves)
    t_condense    = np.zeros(n_solves)
    t_cost        = np.zeros(n_solves)
    t_constraints = np.zeros(n_solves)
    t_solve       = np.zeros(n_solves)

    for i in range(n_solves):
        r_feet = foot_arr[i]
        x0     = state_arr[i]
        psi    = psi_arr[i]

        contacts = schedule[0]
        Xref     = np.tile(x0, K)   # trivial reference (sufficient for timing)

        t0 = time.perf_counter()

        # Phase 1: build discrete A/B matrices
        t1 = time.perf_counter()
        Ad_list, Bd_list = [], []
        for k in range(K):
            A_c = dyn.Ac(psi)
            B_c = dyn.Bc(r_feet, psi, schedule[k])
            Ad, Bd = dyn.discretise(A_c, B_c, MPC_DT)
            Ad_list.append(Ad)
            Bd_list.append(Bd)
        t2 = time.perf_counter()
        t_dynamics[i] = (t2 - t1) * 1e3

        # Phase 2: condense into Aqp, Bqp
        t1 = time.perf_counter()
        Aqp, Bqp = mpc.condense(Ad_list, Bd_list)
        t2 = time.perf_counter()
        t_condense[i] = (t2 - t1) * 1e3

        # Phase 3: build cost matrices
        t1 = time.perf_counter()
        H, g = mpc.cost(Aqp, Bqp, x0, Xref)
        t2 = time.perf_counter()
        t_cost[i] = (t2 - t1) * 1e3

        # Phase 4: build constraint matrices
        t1 = time.perf_counter()
        C, lb, ub = mpc.constraints(schedule)
        t2 = time.perf_counter()
        t_constraints[i] = (t2 - t1) * 1e3

        # Phase 5: solve QP
        t1 = time.perf_counter()
        U = mpc.solve(H, g, C, lb, ub)
        t2 = time.perf_counter()
        t_solve[i] = (t2 - t1) * 1e3

        t_total[i] = (t2 - t0) * 1e3

    # Discard warmup solves
    sl = slice(warmup, n_solves)
    return {
        "total"      : t_total[sl],
        "dynamics"   : t_dynamics[sl],
        "condense"   : t_condense[sl],
        "cost"       : t_cost[sl],
        "constraints": t_constraints[sl],
        "solve"      : t_solve[sl],
    }


def percentile_stats(arr: np.ndarray) -> dict:
    return {
        "n"   : len(arr),
        "mean": float(arr.mean()),
        "min" : float(arr.min()),
        "p50" : float(np.percentile(arr, 50)),
        "p95" : float(np.percentile(arr, 95)),
        "p99" : float(np.percentile(arr, 99)),
        "max" : float(arr.max()),
        "std" : float(arr.std()),
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────
PLOT_STYLE = {
    "font.family"   : "sans-serif",
    "font.size"     : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"     : True,
    "grid.alpha"    : 0.35,
    "grid.linestyle": "--",
}


def plot_main(results: list, scenario_meta: list, out_path: str) -> None:
    """
    3-panel figure:
      (a) Histogram of total solve time per scenario
      (b) Violin / box plot comparing distributions
      (c) Time-series of rolling mean over the benchmark run
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Convex MPC QP Solve-Time Benchmark  |  K={K}, dt={MPC_DT*1e3:.0f} ms\n"
        f"Go2 (15.2 kg) · quadprog · {N_SOLVES - WARMUP} solves/scenario",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── (a) Histogram ─────────────────────────────────────────────────────────
    ax = axes[0]
    for sc, res, meta in zip(SCENARIOS, results, scenario_meta):
        data = res["total"]
        ax.hist(data, bins=60, density=True, alpha=0.65,
                color=sc["color"], label=sc["label"],
                edgecolor="none")
        mean_ms = data.mean()
        ax.axvline(mean_ms, color=sc["color"], linewidth=1.8, linestyle="--")

    # 100 Hz budget line
    budget = 1000.0 / CTRL_HZ
    ax.axvline(budget, color="red", linewidth=1.5, linestyle=":",
               label=f"100 Hz budget ({budget:.0f} ms)")

    ax.set_xlabel("Total solve time (ms)")
    ax.set_ylabel("Probability density")
    ax.set_title("(a) Solve-time distribution")
    ax.legend(fontsize=9, framealpha=0.8)

    # ── (b) Violin + scatter ───────────────────────────────────────────────────
    ax = axes[1]
    labels  = [sc["label"] for sc in SCENARIOS]
    colors  = [sc["color"] for sc in SCENARIOS]
    data_all = [res["total"] for res in results]

    vp = ax.violinplot(data_all, positions=range(len(SCENARIOS)),
                       showmedians=True, showextrema=False)

    for patch, color in zip(vp["bodies"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    # Overlay P95 / P99 markers
    for j, (res, color) in enumerate(zip(results, colors)):
        p95 = np.percentile(res["total"], 95)
        p99 = np.percentile(res["total"], 99)
        ax.plot(j, p95, "^", color=color, markersize=8, zorder=5,
                label="P95" if j == 0 else None)
        ax.plot(j, p99, "D", color=color, markersize=8, zorder=5,
                label="P99" if j == 0 else None)

    ax.axhline(budget, color="red", linewidth=1.5, linestyle=":",
               label=f"100 Hz budget ({budget:.0f} ms)")

    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total solve time (ms)")
    ax.set_title("(b) Distribution comparison")
    ax.legend(fontsize=9, framealpha=0.8, loc="upper right")

    # ── (c) Time-series rolling mean ──────────────────────────────────────────
    ax = axes[2]
    ROLL = 50
    for sc, res in zip(SCENARIOS, results):
        data = res["total"]
        n    = len(data)
        roll = np.convolve(data, np.ones(ROLL) / ROLL, mode="valid")
        ax.plot(np.arange(ROLL - 1, n), roll,
                color=sc["color"], linewidth=1.5, label=sc["label"])

    ax.axhline(budget, color="red", linewidth=1.5, linestyle=":",
               label=f"100 Hz budget ({budget:.0f} ms)")
    ax.set_xlabel("Solve index")
    ax.set_ylabel("Rolling mean solve time (ms)")
    ax.set_title(f"(c) Temporal stability  (window={ROLL})")
    ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def plot_breakdown(results: list, out_path: str) -> None:
    """
    Stacked bar showing mean time for each pipeline phase per scenario.
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))

    phases  = ["dynamics", "condense", "cost", "constraints", "solve"]
    p_labels= ["Dyn. A/B", "Condense", "Cost H/g", "Constraints", "QP solve"]
    p_colors= ["#607D8B", "#9C27B0", "#F44336", "#FF9800", "#4CAF50"]

    x = np.arange(len(SCENARIOS))
    width = 0.45
    bottoms = np.zeros(len(SCENARIOS))

    for phase, label, color in zip(phases, p_labels, p_colors):
        means = np.array([res[phase].mean() for res in results])
        bars  = ax.bar(x, means, width, bottom=bottoms,
                       label=label, color=color, alpha=0.85, edgecolor="white")
        # Annotate each bar segment if big enough
        for rect, m, b in zip(bars, means, bottoms):
            if m > 0.05:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        b + m / 2, f"{m:.3f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += means

    # Total annotation on top
    for i, (sc, res) in enumerate(zip(SCENARIOS, results)):
        total = res["total"].mean()
        ax.text(i, total + 0.02, f"{total:.3f} ms",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    budget = 1000.0 / CTRL_HZ
    ax.axhline(budget, color="red", linewidth=1.5, linestyle=":",
               label=f"100 Hz budget ({budget:.0f} ms)")

    ax.set_xticks(x)
    ax.set_xticklabels([sc["label"] for sc in SCENARIOS], fontsize=10)
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        f"MPC Pipeline Phase Breakdown  |  K={K}, dt={MPC_DT*1e3:.0f} ms\n"
        f"mean over {N_SOLVES - WARMUP} solves per scenario",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def print_table(results: list) -> None:
    """Print a formatted statistics table to stdout."""
    HDR_WIDTH = 22
    COL_W     = 10

    cols = ["mean", "p50", "p95", "p99", "max", "std"]
    col_labels = ["mean", "p50", "p95", "p99", "max", "std"]

    header = (f"{'Scenario':<{HDR_WIDTH}}" +
              "".join(f"{c:>{COL_W}}" for c in col_labels) + "  [all ms]")
    sep = "─" * len(header)

    print()
    print("═" * len(header))
    print("  Convex MPC QP Solve-Time Benchmark")
    print(f"  K={K}  dt={MPC_DT*1e3:.0f}ms  solver=quadprog  "
          f"N={N_SOLVES - WARMUP} solves/scenario")
    print(f"  Control loop budget at 100 Hz = {1000/CTRL_HZ:.1f} ms")
    print("═" * len(header))
    print()
    print(header)
    print(sep)

    for sc, res in zip(SCENARIOS, results):
        st = percentile_stats(res["total"])
        row = (f"{sc['label']:<{HDR_WIDTH}}" +
               "".join(f"{st[c]:>{COL_W}.3f}" for c in cols))
        print(row)

    print(sep)
    print()
    print("  Pipeline phase breakdown (mean ms):")
    print()
    phases = ["dynamics", "condense", "cost", "constraints", "solve", "total"]
    phase_labels = {
        "dynamics"    : "  Build A/B",
        "condense"    : "  Condense",
        "cost"        : "  Cost H/g",
        "constraints" : "  Constraints",
        "solve"       : "  QP solve",
        "total"       : "  TOTAL",
    }
    ph_header = (f"  {'Phase':<16}" +
                 "".join(f"{sc['label']:>{COL_W}}" for sc in SCENARIOS))
    print(ph_header)
    print("  " + "─" * (len(ph_header) - 2))

    for ph in phases:
        label = phase_labels[ph]
        row = f"  {label:<16}"
        for res in results:
            row += f"{res[ph].mean():>{COL_W}.3f}"
        print(row)

    print()
    budget = 1000.0 / CTRL_HZ
    for sc, res in zip(SCENARIOS, results):
        pct_budget = res["total"].mean() / budget * 100
        pct_p99    = np.percentile(res["total"], 99) / budget * 100
        print(f"  {sc['label']}: uses {pct_budget:.1f}% of 100 Hz budget "
              f"(P99 = {pct_p99:.1f}%)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[benchmark_mpc.py]  Initialising …")

    dyn = RigidBodyDynamics(GO2)
    mpc = ConvexMPC(K, GO2)

    results = []
    scenario_meta = []

    for sc in SCENARIOS:
        label = sc["label"].replace("\n", " ")
        print(f"\n  Running scenario: {label!r} …", flush=True)
        res = run_scenario(sc, dyn, mpc, N_SOLVES, WARMUP)
        results.append(res)
        scenario_meta.append(percentile_stats(res["total"]))
        mean_ms = res["total"].mean()
        max_ms  = res["total"].max()
        print(f"    mean={mean_ms:.3f} ms  max={max_ms:.3f} ms  "
              f"(N={len(res['total'])})")

    print_table(results)

    os.makedirs("results", exist_ok=True)
    plot_main(results, scenario_meta,
              "results/benchmark_solve_time.png")
    plot_breakdown(results,
                   "results/benchmark_solve_time_breakdown.png")

    print("\n[done]  Plots saved to results/")


if __name__ == "__main__":
    main()
