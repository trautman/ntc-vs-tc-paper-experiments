import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

HUMAN_COLOR = "orange"
ROBOT_COLOR = "blue"


def setup_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.80, 1.00)

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_traj(ax, traj, color, lw, alpha):
    ax.plot(
        traj[:, 0],
        traj[:, 1],
        color=color,
        linewidth=lw,
        alpha=alpha,
    )

def plot_marginal(ax, trajs, weights, color):
    for traj, w in zip(trajs, weights):
        plot_traj(
            ax,
            traj,
            color,
            lw=0.95,
            alpha=0.25,
        )

def plot_pair_group(ax, H, R, pairs, h_color, r_color, alpha_scale=1.0):
    for i, j, mass in pairs:
        lw = 0.8 + 22.0 * mass
        plot_traj(ax, H[i], h_color, lw=lw, alpha=0.68 * alpha_scale)
        plot_traj(ax, R[j], r_color, lw=lw, alpha=0.62 * alpha_scale)

def label_start_goal(ax, start, goal, color, label, label_side):
    label_offset = 0.20

    x_start, y_start = start
    x_goal, y_goal = goal

    if color == HUMAN_COLOR:
        start_label_y = y_start + label_offset
        goal_label_y = y_goal + label_offset
    else:
        start_label_y = y_start - label_offset
        goal_label_y = y_goal - label_offset

    direction = np.sign(x_goal - x_start)

    dx = x_goal - x_start
    dy = y_goal - y_start
    theta = np.degrees(np.arctan2(dy, dx))

    triangle_marker = MarkerStyle(">")
    triangle_marker._transform = Affine2D().rotate_deg(theta - 90.0)

    triangle_offset = 0.095
    triangle_x = x_start + triangle_offset * np.cos(np.radians(theta))
    triangle_y = y_start + triangle_offset * np.sin(np.radians(theta))

    ax.scatter(
        [triangle_x],
        [triangle_y],
        s=95,
        marker=triangle_marker,
        facecolors=color,
        edgecolors="black",
        linewidths=1.5,
        zorder=21,
    )

    # Goal marker: thick X at exact sample goal
    ax.scatter(
        [x_goal],
        [y_goal],
        s=120,
        marker="x",
        color=color,
        linewidths=3.4,
        zorder=22,
    )

def add_agent_labels(ax, start_h, goal_h, start_r, goal_r):
    label_start_goal(ax, start_h, goal_h, HUMAN_COLOR, "Human", "left")
    label_start_goal(ax, start_r, goal_r, ROBOT_COLOR, "Robot", "right")


def add_mode_title(ax, mode_name, mass):
    ax.text(
        0.0,
        0.94,
        f"{mode_name} Mode ({100*mass:.1f}% of $\\gamma^*$ mass)",
        ha="center",
        va="center",
        fontsize=10,
        zorder=100,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.90,
            pad=1.5,
        ),
    )


def save_combined_mode_page(
    mode_name,
    model_results,
    model_order,
    H,
    R,
    start_h,
    goal_h,
    start_r,
    goal_r,
    combined_dir,
):
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.8))

    for ax, (model_name, label) in zip(axes, model_order):

        setup_axis(ax)

        pairs = model_results[model_name]["pair_groups"][mode_name]
        mass = model_results[model_name]["masses"][mode_name]

        plot_pair_group(
            ax,
            H,
            R,
            pairs,
            HUMAN_COLOR,
            ROBOT_COLOR,
        )

        add_agent_labels(
            ax,
            start_h,
            goal_h,
            start_r,
            goal_r,
        )

        ax.text(
            0.0,
            0.94,
            f"{label}\n({100*mass:.1f}% mass)",
            ha="center",
            va="center",
            fontsize=8,
            zorder=100,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.90,
                pad=1.5,
            ),
        )

    out_path = os.path.join(
        combined_dir,
        f"{mode_name}_comparison.png",
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    print(f"Wrote {out_path}")




