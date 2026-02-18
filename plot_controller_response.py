"""
plot_controller_response.py  –  FIXED VERSION

Post-process and visualize controller response for
Phase 2–6 Convex MPC Trot (Go2).

Reads:  mpc_debug_log.csv
Plots:
    • CoM height tracking
    • Base orientation
    • Linear velocity
    • Vertical GRFs per leg
    • Contact schedule
    • Weight ratio
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless – remove if you want interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_FILE  = "mpc_debug_log.csv"
SAVE_DIR  = "."                  # where to write PNGs
MASS      = 15.0
G         = 9.81

# ── colour / style ──────────────────────────────────────────────────────────
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
GRID_COL  = "#2a2a4a"
TEXT_COL  = "#e0e0e0"
ACCENT    = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77",
             "#c77dff", "#ff9a3c"]

PHASE_COLS = {"A": "#4a90d9", "B": "#f5a623", "C": "#7ed321", "RECOVERY": "#d0021b"}

def style_ax(ax, title, xlabel="Time [s]", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=11, pad=6)
    ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=9)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.6, linestyle="--", alpha=0.7)

def draw_phase_bands(ax, df):
    """Shade background by phase."""
    phases  = df["phase"].values
    times   = df["t"].values
    starts  = [0]
    labels  = [phases[0]]
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            starts.append(i)
            labels.append(phases[i])
    starts.append(len(phases)-1)
    ymin, ymax = ax.get_ylim()
    for k in range(len(labels)):
        t0 = times[starts[k]]
        t1 = times[starts[k+1]] if k+1 < len(starts) else times[-1]
        col = PHASE_COLS.get(labels[k], "#888888")
        ax.axvspan(t0, t1, color=col, alpha=0.06)
    # phase-change vertical lines
    for i in range(1, len(phases)):
        if phases[i] != phases[i-1]:
            ax.axvline(times[i], color=TEXT_COL, linewidth=0.6, alpha=0.4, linestyle=":")

def draw_cmd_lines(ax, cmd_times):
    """Mark velocity command changes."""
    for t in cmd_times:
        ax.axvline(t, color="#ff9a3c", linewidth=1.0, alpha=0.7, linestyle="--")

# ── Data loading & filtering ─────────────────────────────────────────────────

def load_and_filter():
    df = pd.read_csv(LOG_FILE)
    df = df.sort_values("t").reset_index(drop=True)

    # ----------------------------------------------------------------
    # CRITICAL: strip physically-impossible rows produced by the
    # MuJoCo interactive viewer continuing after collapse.
    # Real bounds for Go2:
    #   pz   : 0.10 – 0.50 m
    #   roll  : ±45°  (conservative)
    #   vx,vy : ±2 m/s  (trot regime)
    # ----------------------------------------------------------------
    mask = (
        (df["pz"] > 0.10) & (df["pz"] < 0.50) &
        (df["roll_deg"].abs() < 45) &
        (df["vx"].abs() < 2.0)
    )
    df_valid = df[mask].copy()

    # Also clip to the scripted simulation window (before the
    # interactive viewer produces a second long tumble log).
    # The scripted run ends at the first RECOVERY→end transition;
    # find the last valid trot time before the big gap.
    t_collapse = df_valid[df_valid["phase"] == "RECOVERY"]["t"].min()
    if pd.notna(t_collapse):
        # keep a bit of recovery for context
        df_valid = df_valid[df_valid["t"] <= t_collapse + 2.5]

    print(f"[Filter] kept {len(df_valid)}/{len(df)} rows  "
          f"(t = {df_valid['t'].iloc[0]:.2f} → {df_valid['t'].iloc[-1]:.2f} s)")
    return df_valid


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_com_height(df, ax):
    pz_des = df["pz"] + df["pz_error"]
    ax.plot(df["t"], df["pz"],    color=ACCENT[0], lw=1.4, label="CoM z (actual)")
    ax.plot(df["t"], pz_des,  color=ACCENT[1], lw=1.0,
            linestyle="--", label="Desired z")
    ax.set_ylim(0.26, 0.40)
    style_ax(ax, "CoM Height Tracking", ylabel="Height [m]")
    draw_phase_bands(ax, df)
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8, loc="lower right")


def plot_orientation(df, ax):
    ax.plot(df["t"], df["roll_deg"],  color=ACCENT[0], lw=1.2, label="Roll")
    ax.plot(df["t"], df["pitch_deg"], color=ACCENT[1], lw=1.2, label="Pitch")
    ax.plot(df["t"], df["yaw_deg"],   color=ACCENT[2], lw=1.2, label="Yaw")
    ax.axhline(0, color=TEXT_COL, lw=0.5, alpha=0.4)
    ax.set_ylim(-25, 25)
    style_ax(ax, "Base Orientation", ylabel="Degrees")
    draw_phase_bands(ax, df)
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8)


def plot_velocity(df, ax, cmd_times=(5.5, 8.0, 11.0)):
    ax.plot(df["t"], df["vx"], color=ACCENT[0], lw=1.3, label="vx")
    ax.plot(df["t"], df["vy"], color=ACCENT[1], lw=1.0, label="vy")
    ax.plot(df["t"], df["vz"], color=ACCENT[2], lw=0.8, label="vz", alpha=0.7)
    # draw command-change markers
    cmd_vx = [0.0, 0.2, 0.4, 0.6]
    for i, t in enumerate(cmd_times):
        ax.axvline(t, color="#ff9a3c", lw=1.0, ls="--", alpha=0.8)
        ax.text(t+0.05, 0.55, f"cmd={cmd_vx[i+1]}", color="#ff9a3c",
                fontsize=7, va="top")
    ax.axhline(0, color=TEXT_COL, lw=0.5, alpha=0.4)
    ax.set_ylim(-0.7, 0.7)
    style_ax(ax, "Linear Velocity", ylabel="m/s")
    draw_phase_bands(ax, df)
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8)


def plot_grf(df, axes):
    legs   = ["FL", "FR", "RL", "RR"]
    colors = ACCENT[:4]
    for i, (leg, col) in enumerate(zip(legs, colors)):
        axes[i].plot(df["t"], df[f"fz_{leg}"], color=col, lw=0.9, label=f"{leg} Fz")
        axes[i].set_ylim(-5, 220)
        style_ax(axes[i], f"GRF – {leg}", ylabel="N")
        draw_phase_bands(axes[i], df)
        axes[i].legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8, loc="upper right")


def plot_contacts(df, ax):
    leg_names = ["FL", "FR", "RL", "RR"]
    colors    = ACCENT[:4]
    for i, (leg, col) in enumerate(zip(leg_names, colors)):
        c = df[f"contact_{leg}"].values.astype(float)
        # draw as event bars
        ax.fill_between(df["t"], i + 0.05, i + c * 0.9,
                        step="post", color=col, alpha=0.75, linewidth=0)
    ax.set_ylim(-0.1, 4.1)
    ax.set_yticks([0.4, 1.4, 2.4, 3.4])
    ax.set_yticklabels(leg_names, color=TEXT_COL, fontsize=9)
    style_ax(ax, "Contact Schedule", ylabel="")
    draw_phase_bands(ax, df)


def plot_weight_ratio(df, ax):
    ax.plot(df["t"], df["weight_ratio"], color=ACCENT[0], lw=1.2,
            label="ΣFz / mg")
    ax.axhline(1.0, color=ACCENT[1], lw=1.0, ls="--", alpha=0.8, label="mg")
    ax.set_ylim(0, 3.0)
    style_ax(ax, "Total Vertical Force / Body Weight", ylabel="Ratio")
    draw_phase_bands(ax, df)
    ax.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8)


# ── Assemble figures ──────────────────────────────────────────────────────────

def make_figure_1(df):
    """CoM height + orientation + velocity  (3-panel)"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("MPC Trot  –  CoM & Velocity Response", color=TEXT_COL,
                 fontsize=13, y=0.98)
    plot_com_height(df, axes[0])
    plot_orientation(df, axes[1])
    plot_velocity(df, axes[2])
    for ax in axes[:-1]:
        ax.set_xlabel("")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{SAVE_DIR}/plot_com_velocity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] plot_com_velocity.png")


def make_figure_2(df):
    """GRF per leg  (4-panel)"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("MPC Trot  –  Vertical Ground Reaction Forces", color=TEXT_COL,
                 fontsize=13, y=0.98)
    plot_grf(df, axes)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{SAVE_DIR}/plot_grf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] plot_grf.png")


def make_figure_3(df):
    """Contact schedule + weight ratio  (2-panel)"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("MPC Trot  –  Contact & GRF Ratio", color=TEXT_COL,
                 fontsize=13, y=0.99)
    plot_contacts(df, axes[0])
    plot_weight_ratio(df, axes[1])
    axes[0].set_xlabel("")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{SAVE_DIR}/plot_contacts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] plot_contacts.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_and_filter()

    # Print summary stats on valid data
    phase_c = df[df["phase"] == "C"]
    print(f"\n=== Phase C summary (t={phase_c['t'].min():.1f}–{phase_c['t'].max():.1f}s) ===")
    print(f"  pz     : mean={phase_c['pz'].mean():.4f}  std={phase_c['pz'].std():.4f} m")
    print(f"  roll   : mean={phase_c['roll_deg'].mean():.2f}  std={phase_c['roll_deg'].std():.2f} °")
    print(f"  |vy|   : mean={phase_c['vy'].abs().mean():.4f}  max={phase_c['vy'].abs().max():.4f} m/s")
    print(f"  weight_ratio mean={phase_c['weight_ratio'].mean():.3f}\n")

    make_figure_1(df)
    make_figure_2(df)
    make_figure_3(df)

    print("\nDone. Saved 3 PNG files.")


if __name__ == "__main__":
    main()