import os
import numpy as np


import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ot_models import (
    logsumexp,
    solve_joint_kl,
    solve_sinkhorn_ot,
    solve_marginal_kl,
    solve_balanced_ot,
)


import matplotlib.pyplot as plt
from two_agent_plotting import (
    HUMAN_COLOR,
    ROBOT_COLOR,
    setup_axis,
    plot_marginal,
    plot_pair_group,
    add_agent_labels,
    add_mode_title,
    save_combined_mode_page,
)


OUTDIR = REPO_ROOT / "paper-figures-2-agent"
SEPARATED_DIR = OUTDIR / "separated_plots"
COMBINED_DIR = OUTDIR / "combined_plots"

SEPARATED_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(7)

N = 300
N_JOINT_PAIRS_TO_DISPLAY = 300
T = 100



def softmax_from_logweights(logw):
    return np.exp(logw - logsumexp(logw))


def build_gp_like_library(start, goal, T=100, n_samples=90, seed=0):
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, T)
    x = np.linspace(start[0], goal[0], T)

    trajs = []

    for _ in range(n_samples):
        side = rng.choice([-1.0, 1.0])

        # Smooth density: many samples near centerline, fewer far away.
        amp = abs(rng.normal(loc=0.0, scale=0.23))
        amp = min(amp, 0.85)

        base = np.sin(np.pi * tau)

        wiggle = np.zeros_like(tau)
        for k in range(1, 4):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            coeff = rng.normal(0.0, 1.0 / (k ** 2.3))
            wiggle += coeff * np.sin(2.0 * np.pi * k * tau + phase)

        wiggle = wiggle / max(np.max(np.abs(wiggle)), 1e-9)
        wiggle_amp = rng.uniform(0.00, 0.025)

        # y = side * amp * base + wiggle_amp * wiggle * base
        baseline_y = np.linspace(start[1], goal[1], T)
        y = baseline_y + side * amp * base + wiggle_amp * wiggle * base


        traj = np.column_stack([x, y])
        traj[0] = start
        traj[-1] = goal
        trajs.append(traj)

    return np.array(trajs)


def preference_cost(traj):
    y = traj[:, 1]
    dy = np.diff(y)
    ddy = np.diff(y, n=2)
    max_dev = np.max(np.abs(y))

    return (
        0.35 * max_dev ** 2
        + 2.0 * max(0.0, max_dev - 0.65) ** 2
        + 0.35 * np.sum(dy ** 2)
        + 0.60 * np.sum(ddy ** 2)
    )


def compute_marginal_weights(trajs):
    costs = np.array([preference_cost(tr) for tr in trajs])
    return softmax_from_logweights(-0.9 * costs)


def pairwise_cost(tr_h, tr_r):
    d = np.linalg.norm(tr_h - tr_r, axis=1)
    d_min = np.min(d)

    collision_wall = 60.0 * np.exp(-(d_min / 0.24) ** 2)
    comfort_wall = 12.0 / (1.0 + np.exp(22.0 * (d_min - 0.70)))


    effort = 0.20 * (np.mean(tr_h[:, 1] ** 2) + np.mean(tr_r[:, 1] ** 2))

    return collision_wall + comfort_wall + effort

def top_joint_pairs(gamma, k=1000):
    idx = np.argsort(gamma.ravel())[::-1]
    n_r = gamma.shape[1]

    pairs = []
    for flat_idx in idx:
        i = flat_idx // n_r
        j = flat_idx % n_r
        mass = gamma[i, j]
        if mass <= 1e-10:
            continue
        pairs.append((i, j, mass))
        if len(pairs) >= k:
            break

    return pairs


def local_lateral_side(traj, eps=1e-6):
    mid = len(traj) // 2

    start = traj[0]
    goal = traj[-1]
    point = traj[mid]

    heading = goal - start
    heading_norm = np.linalg.norm(heading)

    if heading_norm < 1e-12:
        return 0

    # left normal of this agent's own start-goal direction
    left_normal = np.array([-heading[1], heading[0]]) / heading_norm

    baseline_mid = 0.5 * (start + goal)
    signed_lateral = np.dot(point - baseline_mid, left_normal)

    if signed_lateral > eps:
        return 1
    if signed_lateral < -eps:
        return -1
    return 0


def split_pairs(H, R, gamma, k_scan=1000, k_each=18):
    all_pairs = top_joint_pairs(gamma, k=k_scan)

    ll = []
    rr = []
    lr = []
    rl = []
    center = []

    for i, j, mass in all_pairs:
        h_side = local_lateral_side(H[i])
        r_side = local_lateral_side(R[j])

        if h_side == 0 or r_side == 0:
            center.append((i, j, mass))
        elif h_side > 0 and r_side > 0:
            ll.append((i, j, mass))
        elif h_side < 0 and r_side < 0:
            rr.append((i, j, mass))
        elif h_side > 0 and r_side < 0:
            lr.append((i, j, mass))
        else:
            rl.append((i, j, mass))
 
    return {
        "LL": ll[:k_each],
        "RR": rr[:k_each],
        "LR": lr[:k_each],
        "RL": rl[:k_each],
        "center": center[:k_each],
    }




def sample_pairs_from_gamma(gamma, n_samples=1000, seed=0):
    rng = np.random.default_rng(seed)

    flat_probs = gamma.ravel()
    flat_probs = flat_probs / flat_probs.sum()

    flat_indices = rng.choice(
        len(flat_probs),
        size=n_samples,
        replace=True,
        p=flat_probs,
    )

    n_r = gamma.shape[1]

    pairs = []
    for flat_idx in flat_indices:
        i = flat_idx // n_r
        j = flat_idx % n_r
        pairs.append((i, j, gamma[i, j]))

    return pairs


def classify_pairs(H, R, pairs):
    groups = {
        "LL": [],
        "RR": [],
        "LR": [],
        "RL": [],
        "center": [],
    }

    for i, j, mass in pairs:
        h_side = local_lateral_side(H[i])
        r_side = local_lateral_side(R[j])

        if h_side == 0 or r_side == 0:
            groups["center"].append((i, j, mass))
        elif h_side > 0 and r_side > 0:
            groups["LL"].append((i, j, mass))
        elif h_side < 0 and r_side < 0:
            groups["RR"].append((i, j, mass))
        elif h_side > 0 and r_side < 0:
            groups["LR"].append((i, j, mass))
        else:
            groups["RL"].append((i, j, mass))

    return groups


def classify_pair_mode(H, R, i, j):
    h_side = local_lateral_side(H[i])
    r_side = local_lateral_side(R[j])

    if h_side == 0 or r_side == 0:
        return "center"
    if h_side > 0 and r_side > 0:
        return "LL"
    if h_side < 0 and r_side < 0:
        return "RR"
    if h_side > 0 and r_side < 0:
        return "LR"
    return "RL"








def main():
 
    start_h = np.array([-1.25,  0.0])
    goal_h  = np.array([ 1.25,  0.18])

    start_r = np.array([ 1.25, -0.3])
    goal_r  = np.array([-1.25, 0.28])

    H = build_gp_like_library(start_h, goal_h, T=T, n_samples=N, seed=3)
    R = build_gp_like_library(start_r, goal_r, T=T, n_samples=N, seed=11)

    p_h = compute_marginal_weights(H)
    p_r = compute_marginal_weights(R)

    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = pairwise_cost(H[i], R[j])



    # ----------------------------
    # Figure 1: overlaid marginals
    # ----------------------------

    fig_marg = plt.figure(figsize=(7.2, 3.8))
    ax_marg = fig_marg.add_subplot(1, 1, 1)

    setup_axis(ax_marg)

    plot_marginal(ax_marg, H, p_h, HUMAN_COLOR)
    plot_marginal(ax_marg, R, p_r, ROBOT_COLOR)

    add_agent_labels(ax_marg, start_h, goal_h, start_r, goal_r)


    # Direction labels

    ax_marg.text(
        start_h[0],
        start_h[1] - 0.25,
        r"$p_h$",
        color=HUMAN_COLOR,
        fontsize=15,
        ha="center",
    )

    ax_marg.text(
        start_r[0],
        start_r[1] - 0.25,
        r"$p_r$",
        color=ROBOT_COLOR,
        fontsize=15,
        ha="center",
    )

    ax_marg.text(
        0.0,
        0.94,
        "Marginals",
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
 

    marg_png_path = OUTDIR / "ot_marginals_only.png"
 
    fig_marg.savefig(
        marg_png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )


    models = {
        "ot_kl_joint": solve_joint_kl(p_h, p_r, C, lam=0.9),
        "ot_sinkhorn": solve_sinkhorn_ot(p_h, p_r, C, reg=0.9),
        "ot_kl_marg": solve_marginal_kl(p_h, p_r, C, lam_h=0.9, lam_r=0.9, reg=1e-2),
        "ot_balanced": solve_balanced_ot(p_h, p_r, C),
    }

    model_results = {}
    for model_name, gamma in models.items():

        sampled_pairs = sample_pairs_from_gamma(
            gamma,
            n_samples=N_JOINT_PAIRS_TO_DISPLAY,
            seed=42,
        )

        pair_groups = classify_pairs(H, R, sampled_pairs)

        print(f"\nModel: {model_name}")
        print("Sampled pair counts:")
        for name, pairs in pair_groups.items():
            print(name, len(pairs))

        total_ll = 0.0
        total_rr = 0.0
        total_lr = 0.0
        total_rl = 0.0
        total_center = 0.0

        for i in range(len(H)):
            for j in range(len(R)):
                h_side = local_lateral_side(H[i])
                r_side = local_lateral_side(R[j])

                if h_side == 0 or r_side == 0:
                    total_center += gamma[i, j]
                elif h_side > 0 and r_side > 0:
                    total_ll += gamma[i, j]
                elif h_side < 0 and r_side < 0:
                    total_rr += gamma[i, j]
                elif h_side > 0 and r_side < 0:
                    total_lr += gamma[i, j]
                else:
                    total_rl += gamma[i, j]

        print("Full gamma mass:")
        print(f"LL      {total_ll:.6f}  ({100*total_ll:.2f}%)")
        print(f"RR      {total_rr:.6f}  ({100*total_rr:.2f}%)")
        print(f"LR      {total_lr:.6f}  ({100*total_lr:.2f}%)")
        print(f"RL      {total_rl:.6f}  ({100*total_rl:.2f}%)")
        print(f"center  {total_center:.6f}  ({100*total_center:.2f}%)")
        print(f"total   {total_ll + total_rr + total_lr + total_rl + total_center:.6f}")

        model_results[model_name] = {
            "pair_groups": pair_groups,
            "masses": {
                "LL": total_ll,
                "RR": total_rr,
                "LR": total_lr,
                "RL": total_rl,
            },
        }




        # ----------------------------
        # LL mode
        # ----------------------------

        fig_ll = plt.figure(figsize=(7.2, 3.8))
        ax_ll = fig_ll.add_subplot(1, 1, 1)

        setup_axis(ax_ll)

        plot_pair_group(
            ax_ll,
            H,
            R,
            pair_groups["LL"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )


        add_agent_labels(ax_ll, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_ll, "LL", total_ll)



        ll_png_path = SEPARATED_DIR / f"{model_name}_LL.png"

        fig_ll.savefig(
            ll_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )


        # ----------------------------
        # RR mode
        # ----------------------------

        fig_rr = plt.figure(figsize=(7.2, 3.8))
        ax_rr = fig_rr.add_subplot(1, 1, 1)

        setup_axis(ax_rr)

        plot_pair_group(
            ax_rr,
            H,
            R,
            pair_groups["RR"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )


        add_agent_labels(ax_rr, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_rr, "RR", total_rr)

        rr_png_path = SEPARATED_DIR / f"{model_name}_RR.png"

        fig_rr.savefig(
            rr_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )

        # ----------------------------
        # LR mode
        # ----------------------------

        fig_lr = plt.figure(figsize=(7.2, 3.8))
        ax_lr = fig_lr.add_subplot(1, 1, 1)

        setup_axis(ax_lr)

        plot_pair_group(
            ax_lr,
            H,
            R,
            pair_groups["LR"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )




        add_agent_labels(ax_lr, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_lr, "LR", total_lr)

        lr_png_path = SEPARATED_DIR / f"{model_name}_LR.png"

        fig_lr.savefig(
            lr_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )


        # ----------------------------
        # RL mode
        # ----------------------------

        fig_rl = plt.figure(figsize=(7.2, 3.8))
        ax_rl = fig_rl.add_subplot(1, 1, 1)

        setup_axis(ax_rl)

        plot_pair_group(
            ax_rl,
            H,
            R,
            pair_groups["RL"],
            HUMAN_COLOR,
            ROBOT_COLOR,
        )

        add_agent_labels(ax_rl, start_h, goal_h, start_r, goal_r)
        add_mode_title(ax_rl, "RL", total_rl)

        rl_png_path = SEPARATED_DIR / f"{model_name}_RL.png"

        fig_rl.savefig(
            rl_png_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )


    model_order = [
        ("ot_balanced", "Balanced"),
        ("ot_sinkhorn", "Sinkhorn"),
        ("ot_kl_marg", "Marginal KL"),
        ("ot_kl_joint", "Joint KL"),
    ]

    for mode_name in ["LL", "RR", "LR", "RL"]:
        save_combined_mode_page(
            mode_name,
            model_results,
            model_order,
            H,
            R,
            start_h,
            goal_h,
            start_r,
            goal_r,
            COMBINED_DIR,
        )

    print(f"Wrote {marg_png_path}")
    print(f"Wrote {ll_png_path}")
    print(f"Wrote {rr_png_path}")
    print(f"Wrote {lr_png_path}")
    print(f"Wrote {rl_png_path}")





if __name__ == "__main__":
    main()