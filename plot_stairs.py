"""
plot_stairs.py — Publication-Quality Stair-Climbing Controller Response
========================================================================

Post-process and visualise controller response from stairs_climb.py.
Reads:  data/mpc_debug_log.csv
Saves:  results/stairs/

Produces 4 publication-ready figures:

  Figure 1 — Terrain-Adaptive Height & Orientation
      Panel A: CoM z (actual vs desired) overlaid with terrain step profile
      Panel B: Body pitch — tracks stair slope ref (+8.8° ascent / –8.8° descent)
      Panel C: Body roll — lateral stability metric
      Panel D: Robot x-position (progress through stair scene)

  Figure 2 — Foot Placement on Stairs
      Per-leg foot z-position overlaid on terrain step geometry.
      Shows that each foot lands on the correct tread, not the riser.

  Figure 3 — Ground Reaction Forces
      Per-leg vertical GRF.  Asymmetry between front/rear during step
      transitions is the key stair-negotiation signature.

  Figure 4 — Contact Schedule & Weight Balance
      Trot contact pattern + total Fz / mg ratio.
      Shows gait continuity throughout the full stair traverse.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

# ── File paths ────────────────────────────────────────────────────────────────
LOG_FILE = "data/mpc_debug_log.csv"
SAVE_DIR = "results/stairs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Stair geometry (mirrors terrain_estimator.py) ─────────────────────────────
STAIR_START_X    = 2.00
STAIR_TREAD      = 0.28
STAIR_RISER      = 0.06
STAIR_COUNT      = 5
PLATFORM_START_X = 3.40
PLATFORM_END_X   = 4.40
STAIR_COUNT_DOWN = 5
DESCENT_END_X    = 5.80
LANDING_Z        = 0.30
MASS             = 15.206408
G                = 9.81

# ── Colour palette — dark academic / publication ───────────────────────────────
BG        = "#0d1117"        # near-black background
PANEL_BG  = "#161b22"        # slightly lighter panel
GRID_COL  = "#21262d"        # subtle grid
BORDER    = "#30363d"        # axes border
TEXT_COL  = "#e6edf3"        # primary text
DIM_TEXT  = "#8b949e"        # secondary labels

# Leg colours — carefully chosen for colourblind safety
LEG_COLORS = {
    "FL": "#58a6ff",   # blue
    "FR": "#3fb950",   # green
    "RL": "#f78166",   # coral
    "RR": "#d2a8ff",   # lavender
}
ACCENT      = "#ffa657"   # orange — highlight / desired trajectory
TERRAIN_COL = "#484f58"   # muted grey for terrain geometry
PHASE_ALPHA = 0.10

PHASE_COLS = {
    "A":        "#388bfd",
    "B":        "#d29922",
    "C":        "#2ea043",
    "RECOVERY": "#f85149",
}

# ── Stair zone labels (x ranges) ──────────────────────────────────────────────
ZONES = [
    (0.0,            STAIR_START_X,    "Flat\napproach",  "#388bfd"),
    (STAIR_START_X,  PLATFORM_START_X, "Ascent\n(5 steps)", "#2ea043"),
    (PLATFORM_START_X, PLATFORM_END_X, "Top\nplatform",  "#d29922"),
    (PLATFORM_END_X, DESCENT_END_X,   "Descent\n(5 steps)", "#f85149"),
    (DESCENT_END_X,  99.0,            "Flat\nexit",      "#388bfd"),
]


# ═════════════════════════════════════════════════════════════════════════════
# Terrain helper
# ═════════════════════════════════════════════════════════════════════════════

def stair_height(x: float) -> float:
    """Step-function terrain height at world x-coordinate."""
    if x < STAIR_START_X:
        return 0.0
    if x < PLATFORM_START_X:
        step = int((x - STAIR_START_X) / STAIR_TREAD)
        return min(step + 1, STAIR_COUNT) * STAIR_RISER
    if x < PLATFORM_END_X:
        return LANDING_Z
    if x < DESCENT_END_X:
        step = int((x - PLATFORM_END_X) / STAIR_TREAD)
        return max(LANDING_Z - (step + 1) * STAIR_RISER, 0.0)
    return 0.0


def terrain_profile(x_min=1.5, x_max=6.5, n=2000):
    """Vectorised terrain profile for plotting."""
    xs = np.linspace(x_min, x_max, n)
    zs = np.array([stair_height(x) for x in xs])
    return xs, zs


def stair_pitch_ref(x: float) -> float:
    """Expected body pitch reference at robot x (degrees)."""
    HIP_X = 0.1934
    if x < STAIR_START_X:
        return 0.0
    front_z = stair_height(x + HIP_X)
    rear_z  = stair_height(x - HIP_X)
    dz      = front_z - rear_z
    return float(np.degrees(np.clip(np.arctan2(dz, 2.0 * HIP_X), -0.21, 0.21)))


# ═════════════════════════════════════════════════════════════════════════════
# Data loading & filtering
# ═════════════════════════════════════════════════════════════════════════════

def load_and_filter():
    df = pd.read_csv(LOG_FILE)
    df = df.sort_values("t").reset_index(drop=True)

    # Stair-specific physical bounds:
    #   pz can reach ~0.65m (0.30 terrain + 0.35 clearance at step 5)
    #   pitch up to ±15° is expected on the slope
    #   roll should stay < 35° — wider tolerance than flat
    mask = (
        (df["pz"]        > 0.10) &
        (df["pz"]        < 0.75) &
        (df["roll_deg"].abs()  < 35) &
        (df["vx"].abs()        < 2.0)
    )
    df_valid = df[mask].copy()

    # Clip to before any RECOVERY phase (crash detection)
    t_collapse = df_valid[df_valid["phase"] == "RECOVERY"]["t"].min()
    if pd.notna(t_collapse):
        df_valid = df_valid[df_valid["t"] <= t_collapse + 1.0]

    # We need robot x from foot data — use mean front-foot x as proxy
    if "foot_x_FL" in df_valid.columns and "foot_x_FR" in df_valid.columns:
        df_valid["robot_x"] = 0.5 * (df_valid["foot_x_FL"] + df_valid["foot_x_FR"]) - 0.1934
    else:
        # Fallback: won't have x, use time as proxy
        df_valid["robot_x"] = np.nan

    print(f"[Filter] kept {len(df_valid)}/{len(df)} rows  "
          f"t = {df_valid['t'].iloc[0]:.2f} → {df_valid['t'].iloc[-1]:.2f} s")

    df_c = df_valid[df_valid["phase"] == "C"].copy()
    print(f"[Filter] Phase C: {len(df_c)} rows  "
          f"x = {df_c['robot_x'].min():.2f} → {df_c['robot_x'].max():.2f} m")
    return df_valid, df_c


# ═════════════════════════════════════════════════════════════════════════════
# Style helpers
# ═════════════════════════════════════════════════════════════════════════════

def style_ax(ax, title, xlabel="Time [s]", ylabel="", title_size=10):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=title_size,
                 pad=5, fontweight="semibold", loc="left")
    ax.set_xlabel(xlabel, color=DIM_TEXT, fontsize=8)
    ax.set_ylabel(ylabel, color=DIM_TEXT, fontsize=8)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(color=GRID_COL, linewidth=0.5, linestyle="-", alpha=1.0)
    ax.set_axisbelow(True)


def draw_zone_bands(ax, df, x_col="t"):
    """
    Shade time axis by stair zone inferred from robot_x.
    Falls back gracefully if robot_x is unavailable.
    """
    if "robot_x" not in df.columns or df["robot_x"].isna().all():
        return
    times = df[x_col].values
    rxs   = df["robot_x"].values

    prev_zone = None
    t_start   = times[0]

    for i in range(len(times)):
        rx = rxs[i]
        cur_zone = None
        for x0, x1, label, col in ZONES:
            if x0 <= rx < x1:
                cur_zone = (label, col)
                break
        if cur_zone != prev_zone:
            if prev_zone is not None:
                ax.axvspan(t_start, times[i], color=prev_zone[1],
                           alpha=PHASE_ALPHA, linewidth=0)
            t_start   = times[i]
            prev_zone = cur_zone

    if prev_zone is not None:
        ax.axvspan(t_start, times[-1], color=prev_zone[1],
                   alpha=PHASE_ALPHA, linewidth=0)


def draw_phase_lines(ax, df, x_col="t"):
    """Vertical dotted lines at phase transitions."""
    phases = df["phase"].values
    times  = df[x_col].values
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(times[i], color=PHASE_COLS.get(phases[i], BORDER),
                       linewidth=0.8, alpha=0.6, linestyle=":")


def zone_legend(fig):
    """Return legend handles for stair zone bands."""
    patches = [
        mpatches.Patch(color=col, alpha=0.4, label=lbl.replace("\n", " "))
        for _, _, lbl, col in ZONES
        if lbl not in ("Flat\napproach",)  # deduplicate approach/exit
    ]
    patches.insert(0, mpatches.Patch(color=ZONES[0][3], alpha=0.4, label="Flat (approach/exit)"))
    return patches


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Terrain-Adaptive Height & Body Orientation
# ═════════════════════════════════════════════════════════════════════════════

def make_figure_1(df_all, df_c):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2, 0.8]})
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Stair-Climbing MPC  –  Terrain-Adaptive Height & Body Orientation",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.995
    )

    df = df_c  # only Phase C for clean plots

    # ── Panel A: CoM z height ──────────────────────────────────────────────
    ax = axes[0]
    pz_des = df["pz"] + df["pz_error"]

    # Terrain height derived from robot x
    if "robot_x" in df.columns and not df["robot_x"].isna().all():
        terrain_z_est = df["robot_x"].apply(stair_height)
        ax.fill_between(df["t"], 0, terrain_z_est + 0.005,
                        color=TERRAIN_COL, alpha=0.5, label="Terrain surface", step="mid")

    ax.plot(df["t"], pz_des,  color=ACCENT,       lw=1.1, ls="--",
            alpha=0.9, label="pz desired (terrain-adaptive)")
    ax.plot(df["t"], df["pz"], color=LEG_COLORS["FL"], lw=1.6,
            label="CoM z (actual)")

    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("Height [m]", color=DIM_TEXT, fontsize=8)
    ax.set_title("A  |  CoM Height Tracking with Terrain Profile",
                 color=TEXT_COL, fontsize=10, fontweight="semibold", loc="left", pad=4)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=GRID_COL, lw=0.5, alpha=1.0)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, labelcolor=TEXT_COL, fontsize=8,
              loc="upper left", framealpha=0.8, edgecolor=BORDER)
    draw_zone_bands(ax, df)

    # Annotate stair zones at top
    if "robot_x" in df.columns and not df["robot_x"].isna().all():
        zone_labels = [
            (STAIR_START_X,    "Ascent"),
            (PLATFORM_START_X, "Platform"),
            (PLATFORM_END_X,   "Descent"),
        ]
        for target_x, label in zone_labels:
            # Find the time when robot_x first crosses target_x
            idx = (df["robot_x"] - target_x).abs().idxmin()
            t_cross = df.loc[idx, "t"]
            ax.axvline(t_cross, color=TEXT_COL, lw=0.7, alpha=0.4, ls=":")
            ax.text(t_cross + 0.1, 0.68, label, color=DIM_TEXT,
                    fontsize=7.5, va="top", style="italic")

    # ── Panel B: Pitch ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(df["t"], df["pitch_deg"], color=LEG_COLORS["FR"], lw=1.4,
            label="Body pitch (actual)")

    # Overlay expected pitch reference from terrain geometry
    if "robot_x" in df.columns and not df["robot_x"].isna().all():
        pitch_ref = df["robot_x"].apply(stair_pitch_ref)
        ax.plot(df["t"], pitch_ref, color=ACCENT, lw=1.0, ls="--",
                alpha=0.85, label="Pitch reference (terrain slope)")

    ax.axhline(0, color=BORDER, lw=0.8, alpha=0.7)
    ax.set_ylim(-18, 18)
    ax.set_yticks([-12, -6, 0, 6, 12])
    ax.set_title("B  |  Body Pitch vs Stair Slope Reference",
                 color=TEXT_COL, fontsize=10, fontweight="semibold", loc="left", pad=4)
    ax.set_ylabel("Pitch [°]", color=DIM_TEXT, fontsize=8)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=GRID_COL, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, labelcolor=TEXT_COL, fontsize=8,
              loc="upper right", framealpha=0.8, edgecolor=BORDER)
    draw_zone_bands(ax, df)

    # ── Panel C: Roll ──────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(df["t"], df["roll_deg"], color=LEG_COLORS["RL"], lw=1.4,
            label="Body roll")
    ax.axhline(0, color=BORDER, lw=0.8, alpha=0.7)
    ax.fill_between(df["t"], -5, 5, color=ACCENT, alpha=0.06, label="±5° band")
    ax.set_ylim(-20, 20)
    ax.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    ax.set_title("C  |  Lateral Roll Stability",
                 color=TEXT_COL, fontsize=10, fontweight="semibold", loc="left", pad=4)
    ax.set_ylabel("Roll [°]", color=DIM_TEXT, fontsize=8)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=GRID_COL, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, labelcolor=TEXT_COL, fontsize=8,
              loc="upper right", framealpha=0.8, edgecolor=BORDER)
    draw_zone_bands(ax, df)

    # ── Panel D: x-velocity ────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(df["t"], df["vx"], color=LEG_COLORS["RR"], lw=1.3, label="vx")
    ax.axhline(0.20, color=ACCENT, lw=0.8, ls=":", alpha=0.6, label="cmd 0.20 m/s")
    ax.axhline(0.10, color=ACCENT, lw=0.8, ls="--", alpha=0.6, label="cmd 0.10 m/s")
    ax.set_ylim(-0.05, 0.35)
    ax.set_title("D  |  Forward Velocity",
                 color=TEXT_COL, fontsize=10, fontweight="semibold", loc="left", pad=4)
    ax.set_ylabel("vx [m/s]", color=DIM_TEXT, fontsize=8)
    ax.set_xlabel("Time [s]", color=DIM_TEXT, fontsize=8)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=GRID_COL, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, labelcolor=TEXT_COL, fontsize=8,
              loc="upper right", framealpha=0.8, edgecolor=BORDER)
    draw_zone_bands(ax, df)

    plt.tight_layout(rect=[0, 0, 1, 0.993], h_pad=0.4)
    _save(fig, "plot_stairs_height_orientation.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Foot Placement
# ═════════════════════════════════════════════════════════════════════════════

def make_figure_2(df_all, df_c):
    has_foot = all(
        f"foot_z_{leg}" in df_c.columns and f"foot_x_{leg}" in df_c.columns
        for leg in ["FL", "FR", "RL", "RR"]
    )
    if not has_foot:
        print("[SKIP] Figure 2: foot position columns not found in CSV.")
        return

    df = df_c

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Stair-Climbing MPC  –  Per-Leg Foot Placement vs Terrain",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.998
    )

    # Terrain profile in x-space for overlay
    tx, tz = terrain_profile(1.4, 6.5)

    legs_ordered = [("FL", 0, 0), ("FR", 0, 1), ("RL", 1, 0), ("RR", 1, 1)]

    for leg, ri, ci in legs_ordered:
        ax = axes[ri][ci]
        col = LEG_COLORS[leg]

        foot_x = df[f"foot_x_{leg}"].values
        foot_z = df[f"foot_z_{leg}"].values
        t_vals = df["t"].values

        # Draw terrain cross-section
        ax.fill_between(tx, 0, tz, color=TERRAIN_COL, alpha=0.55, label="Terrain")
        ax.plot(tx, tz, color=BORDER, lw=0.8)

        # Scatter foot positions coloured by time (progress indicator)
        sc = ax.scatter(foot_x, foot_z, c=t_vals, cmap="plasma",
                        s=1.5, alpha=0.6, linewidths=0, zorder=3)

        # Add riser lines for clarity
        for step_i in range(STAIR_COUNT):
            riser_x = STAIR_START_X + (step_i + 1) * STAIR_TREAD
            riser_z = (step_i + 1) * STAIR_RISER
            ax.plot([riser_x, riser_x], [0, riser_z + 0.01],
                    color=BORDER, lw=0.5, alpha=0.4, ls=":")
        for step_i in range(STAIR_COUNT_DOWN):
            riser_x = PLATFORM_END_X + (step_i + 1) * STAIR_TREAD
            riser_z = LANDING_Z - (step_i + 1) * STAIR_RISER
            ax.plot([riser_x, riser_x], [0, riser_z + 0.01],
                    color=BORDER, lw=0.5, alpha=0.4, ls=":")

        ax.set_xlim(1.5, 6.3)
        ax.set_ylim(-0.02, 0.42)
        ax.set_title(f"{leg} Foot Placement",
                     color=col, fontsize=10, fontweight="semibold", loc="left", pad=4)
        ax.set_facecolor(PANEL_BG)
        ax.set_xlabel("World x [m]", color=DIM_TEXT, fontsize=8)
        ax.set_ylabel("Foot z [m]", color=DIM_TEXT, fontsize=8)
        ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(color=GRID_COL, lw=0.5)
        ax.set_axisbelow(True)

        # Colourbar (time)
        cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label("Time [s]", color=DIM_TEXT, fontsize=7)
        cb.ax.tick_params(colors=DIM_TEXT, labelsize=7)

        # Annotate stair zones on x-axis
        for x_mark, label in [(2.0, "↑"), (3.4, "—"), (4.4, "↓")]:
            ax.axvline(x_mark, color=TEXT_COL, lw=0.6, alpha=0.3, ls=":")
            ax.text(x_mark + 0.04, 0.38, label, color=DIM_TEXT,
                    fontsize=9, va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.993], h_pad=0.6, w_pad=0.6)
    _save(fig, "plot_stairs_foot_placement.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Ground Reaction Forces
# ═════════════════════════════════════════════════════════════════════════════

def make_figure_3(df_all, df_c):
    df = df_c

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Stair-Climbing MPC  –  Vertical Ground Reaction Forces",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.998
    )

    weight = MASS * G
    legs = ["FL", "FR", "RL", "RR"]

    for i, leg in enumerate(legs):
        ax = axes[i]
        col = LEG_COLORS[leg]

        fz = df[f"fz_{leg}"].values
        t  = df["t"].values

        # Shade stance periods derived from GRF (contact_* columns are all-1 due
        # to a logging bug; Fz threshold is the ground-truth stance indicator).
        CONTACT_THRESHOLD = 10.0  # N
        in_stance = df[f"fz_{leg}"].values > CONTACT_THRESHOLD
        # Draw stance background
        ax.fill_between(t, 0, 220, where=in_stance,
                        color=col, alpha=0.06, step="mid", label="Stance")

        ax.plot(t, fz, color=col, lw=1.1, label=f"{leg} Fz")
        ax.axhline(weight / 4, color=ACCENT, lw=0.8, ls="--",
                   alpha=0.6, label=f"mg/4 = {weight/4:.0f} N")
        ax.set_ylim(-5, 230)
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_title(f"{leg}  |  Vertical GRF",
                     color=col, fontsize=10, fontweight="semibold", loc="left", pad=4)
        ax.set_ylabel("Force [N]", color=DIM_TEXT, fontsize=8)
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(color=GRID_COL, lw=0.5)
        ax.set_axisbelow(True)
        ax.legend(facecolor=BG, labelcolor=TEXT_COL, fontsize=7.5,
                  loc="upper right", framealpha=0.8, edgecolor=BORDER,
                  ncol=2)
        draw_zone_bands(ax, df)

    axes[-1].set_xlabel("Time [s]", color=DIM_TEXT, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.993], h_pad=0.4)
    _save(fig, "plot_stairs_grf.png")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Contact Schedule & Weight Balance
# ═════════════════════════════════════════════════════════════════════════════

def make_figure_4(df_all, df_c):
    df = df_c

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2, 1.2]})
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Stair-Climbing MPC  –  Contact Schedule & Force Balance",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.998
    )

    # ── Panel A: Contact schedule (swimlane) ───────────────────────────────
    ax = axes[0]
    legs = ["FL", "FR", "RL", "RR"]
    t = df["t"].values

    CONTACT_THRESHOLD = 10.0  # N — derive stance from GRF (contact_* columns are all-1)
    for i, leg in enumerate(legs):
        col = LEG_COLORS[leg]
        c   = (df[f"fz_{leg}"].values > CONTACT_THRESHOLD).astype(float)
        ax.fill_between(t, i + 0.08, i + c * 0.84,
                        step="post", color=col, alpha=0.80, linewidth=0)
        ax.text(t[0] - 0.1, i + 0.46, leg, color=col,
                fontsize=9, fontweight="bold", ha="right", va="center")

    ax.set_ylim(-0.1, 4.1)
    ax.set_yticks([])
    ax.set_title("A  |  Trot Contact Schedule",
                 color=TEXT_COL, fontsize=10, fontweight="semibold", loc="left", pad=4)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=GRID_COL, lw=0.5, axis="x")
    ax.set_axisbelow(True)
    draw_zone_bands(ax, df)

    # Legend for zones
    handles = zone_legend(fig)
    ax.legend(handles=handles, facecolor=BG, labelcolor=TEXT_COL,
              fontsize=7.5, loc="upper right", framealpha=0.8,
              edgecolor=BORDER, ncol=2)

    # ── Panel B: Total Fz / mg ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, df["weight_ratio"], color=LEG_COLORS["FL"], lw=1.2,
            label="ΣFz / mg")
    ax.axhline(1.0, color=ACCENT, lw=0.9, ls="--", alpha=0.8, label="ΣFz = mg")
    ax.fill_between(t, 0.85, 1.15, color=ACCENT, alpha=0.06, label="±15% band")
    ax.set_ylim(0, 2.5)
    ax.set_title("B  |  Total Vertical Force / Body Weight",
                 color=TEXT_COL, fontsize=10, fontweight="semibold", loc="left", pad=4)
    ax.set_ylabel("Ratio", color=DIM_TEXT, fontsize=8)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=GRID_COL, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, labelcolor=TEXT_COL, fontsize=8,
              loc="upper right", framealpha=0.8, edgecolor=BORDER)
    draw_zone_bands(ax, df)

    # ── Panel C: Front vs Rear Fz imbalance ───────────────────────────────
    # This is a stair-specific metric: front legs carry more on ascent,
    # rear legs carry more on descent — the MPC load-sharing signature.
    ax = axes[2]
    fz_front = df["fz_FL"] + df["fz_FR"]
    fz_rear  = df["fz_RL"] + df["fz_RR"]
    weight   = MASS * G
    ax.plot(t, fz_front / weight, color=LEG_COLORS["FL"], lw=1.1,
            label="(FL+FR) / mg  [front]")
    ax.plot(t, fz_rear  / weight, color=LEG_COLORS["RL"], lw=1.1,
            label="(RL+RR) / mg  [rear]",  ls="--")
    ax.axhline(0.5, color=ACCENT, lw=0.7, ls=":", alpha=0.5, label="Equal share")
    ax.set_ylim(0, 1.5)
    ax.set_title("C  |  Front vs Rear Force Distribution (Stair Load-Sharing)",
                 color=TEXT_COL, fontsize=10, fontweight="semibold", loc="left", pad=4)
    ax.set_ylabel("Force / mg", color=DIM_TEXT, fontsize=8)
    ax.set_xlabel("Time [s]", color=DIM_TEXT, fontsize=8)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=7.5, length=3)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color=GRID_COL, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, labelcolor=TEXT_COL, fontsize=8,
              loc="upper right", framealpha=0.8, edgecolor=BORDER)
    draw_zone_bands(ax, df)

    plt.tight_layout(rect=[0, 0, 1, 0.993], h_pad=0.4)
    _save(fig, "plot_stairs_contact_balance.png")


# ═════════════════════════════════════════════════════════════════════════════
# Save helper
# ═════════════════════════════════════════════════════════════════════════════

def _save(fig, name):
    path = f"{SAVE_DIR}/{name}"
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] {name}")


# ═════════════════════════════════════════════════════════════════════════════
# Summary stats
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(df_c):
    print("\n" + "=" * 68)
    print("  Phase C summary statistics")
    print("=" * 68)

    # Split by stair zone
    if "robot_x" in df_c.columns and not df_c["robot_x"].isna().all():
        zones_data = {
            "Flat approach":  df_c[df_c["robot_x"] < STAIR_START_X],
            "Ascent":         df_c[(df_c["robot_x"] >= STAIR_START_X) &
                                   (df_c["robot_x"] <  PLATFORM_START_X)],
            "Platform":       df_c[(df_c["robot_x"] >= PLATFORM_START_X) &
                                   (df_c["robot_x"] <  PLATFORM_END_X)],
            "Descent":        df_c[(df_c["robot_x"] >= PLATFORM_END_X) &
                                   (df_c["robot_x"] <  DESCENT_END_X)],
            "Flat exit":      df_c[df_c["robot_x"] >= DESCENT_END_X],
        }
        for zone_name, zdf in zones_data.items():
            if len(zdf) < 5:
                continue
            print(f"\n  [{zone_name}]  ({len(zdf)} samples)")
            print(f"    pz        : mean={zdf['pz'].mean():.3f}  "
                  f"std={zdf['pz'].std():.3f} m")
            print(f"    pitch     : mean={zdf['pitch_deg'].mean():.1f}  "
                  f"std={zdf['pitch_deg'].std():.1f} °")
            print(f"    roll      : mean={zdf['roll_deg'].mean():.1f}  "
                  f"std={zdf['roll_deg'].std():.1f} °")
            print(f"    vx        : mean={zdf['vx'].mean():.3f}  "
                  f"std={zdf['vx'].std():.3f} m/s")
            print(f"    wt_ratio  : mean={zdf['weight_ratio'].mean():.3f}")
    else:
        print(f"  pz     : mean={df_c['pz'].mean():.4f}  std={df_c['pz'].std():.4f} m")
        print(f"  pitch  : mean={df_c['pitch_deg'].mean():.2f}  "
              f"std={df_c['pitch_deg'].std():.2f} °")
        print(f"  roll   : mean={df_c['roll_deg'].mean():.2f}  "
              f"std={df_c['roll_deg'].std():.2f} °")
        print(f"  wt_ratio mean={df_c['weight_ratio'].mean():.3f}")

    print("=" * 68)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Reading {LOG_FILE} ...")
    df_all, df_c = load_and_filter()

    if len(df_c) == 0:
        print("[ERROR] No Phase C data found after filtering. "
              "Check that mpc_debug_log.csv contains stair-climbing data.")
        return

    print_summary(df_c)

    print("\nGenerating figures ...")
    make_figure_1(df_all, df_c)   # Height + orientation
    make_figure_2(df_all, df_c)   # Foot placement
    make_figure_3(df_all, df_c)   # GRFs
    make_figure_4(df_all, df_c)   # Contact + weight balance

    print(f"\nDone.  4 PNGs saved to '{SAVE_DIR}/'")
    print("  plot_stairs_height_orientation.png")
    print("  plot_stairs_foot_placement.png")
    print("  plot_stairs_grf.png")
    print("  plot_stairs_contact_balance.png")


if __name__ == "__main__":
    main()