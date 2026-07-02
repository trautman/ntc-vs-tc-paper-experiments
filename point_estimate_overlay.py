import numpy as np


def compute_collaborative_point_estimate(p_h, p_r, C, lam=0.9, eps=1e-300):
    score = (
        np.log(p_h[:, None] + eps)
        + np.log(p_r[None, :] + eps)
        - C / lam
    )

    flat_idx = int(np.argmax(score))
    i, j = np.unravel_index(flat_idx, C.shape)

    return i, j, score[i, j]


def compute_joint_argmax(gamma):
    flat_idx = int(np.argmax(gamma))
    i, j = np.unravel_index(flat_idx, gamma.shape)

    return i, j, gamma[i, j]


def plot_estimate_pair(
    ax,
    H,
    R,
    pair,
    h_color="black",
    r_color="black",
    lw=3.0,
    linestyle="-",
    alpha=1.0,
    zorder=80,
):
    i, j = pair

    ax.plot(
        H[i][:, 0],
        H[i][:, 1],
        color=h_color,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )

    ax.plot(
        R[j][:, 0],
        R[j][:, 1],
        color=r_color,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )


def add_point_estimate_solution_to_figure(
    ax,
    H,
    R,
    p_h,
    p_r,
    C,
    gamma,
    mode_name,
    classify_pair_mode_fn,
    lam=0.9,
):
    point_i, point_j, point_score = compute_collaborative_point_estimate(
        p_h,
        p_r,
        C,
        lam=lam,
    )

    joint_i, joint_j, joint_mass = compute_joint_argmax(gamma)

    point_mode = classify_pair_mode_fn(H, R, point_i, point_j)

    if point_mode != mode_name:
        return False

    plot_estimate_pair(
        ax,
        H,
        R,
        (point_i, point_j),
        h_color="black",
        r_color="black",
        lw=4.0,
        linestyle="-",
        alpha=1.0,
        zorder=120,
    )

    plot_estimate_pair(
        ax,
        H,
        R,
        (joint_i, joint_j),
        h_color="red",
        r_color="red",
        lw=2.8,
        linestyle="--",
        alpha=1.0,
        zorder=125,
    )

    ax.plot(
        [],
        [],
        color="black",
        linewidth=4.0,
        linestyle="-",
        label=r"$(h^*_{\mathrm{opt}}, r^*_{\mathrm{opt}})$",
    )

    ax.plot(
        [],
        [],
        color="red",
        linestyle="--",
        linewidth=2.8,
        label=r"$(h^*_{\gamma}, r^*_{\gamma})$",
    )

    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.90,
        facecolor="white",
    )

    return True