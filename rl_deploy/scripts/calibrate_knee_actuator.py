"""CMA-ES calibration of Spot knee torque-speed limits (ReLIC-style) with
observability for sim/real mismatch diagnosis.

Replays recorded joint position commands OPEN-LOOP in Isaac Lab under
candidate torque-speed breakpoints, minimizing 1D Wasserstein distance
between simulated and real knee torque distributions. Supports multiple
HDF5 logs (averaged cost) for better coverage of operating conditions.

Outputs a full set of diagnostic plots so a flat/non-improving calibration
can be traced to either (a) the torque-speed envelope, (b) PID gain
mismatch, or (c) tracking-error mismatch. Prints a numeric verdict at the
end of every run.

Writes optimized breakpoint arrays ready to paste into
rl_deploy/isaaclab_spot/spot_knee_actuator.py.
"""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hdf5_file", type=Path, nargs="+",
                       help="One or more HDF5 log files from real robot.")
    group.add_argument("--hdf5_dir", type=Path,
                       help="Directory containing spot_isaac_real_*.hdf5 files.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=-1,
                        help="Max steps per file. -1 for all.")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Initial sim steps to discard before scoring.")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="CMA-ES initial sigma as fraction of |theta0|.")
    parser.add_argument("--out_dir", type=Path, default=Path("logs"))

    # Gain-override knobs: let the user A/B sim PID gains against real-robot
    # K_Q_P / K_QD_P without editing isaac_model.py. If the knee torque match
    # stays bad even with matched gains, envelope calibration really is the
    # bottleneck; otherwise PID gains are the dominant problem.
    parser.add_argument("--override_knee_stiffness", type=float, default=None)
    parser.add_argument("--override_knee_damping", type=float, default=None)
    parser.add_argument("--override_hip_stiffness", type=float, default=None)
    parser.add_argument("--override_hip_damping", type=float, default=None)

    parser.add_argument("--skip_landscape", action="store_true",
                        help="Skip 1D Wasserstein landscape sweeps (slow).")
    parser.add_argument("--landscape_points", type=int, default=11,
                        help="Sweep points per theta dim for landscape plot.")

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Parameter model: 6-dim theta maps to the two 3-point breakpoint arrays used
# by SpotKneeActuatorCfg. Outer endpoints (-30, 30 rad/s) are held fixed.
#
# Why only the knees? Per ReLIC §E.2, "calibrating only the knee joints using
# torque-velocity constraints is sufficient." Hips use plain DelayedPDActuator
# with no torque-speed envelope, so extending CMA-ES to hips would require a
# new actuator class. Instead, we record hip data in the diagnostic plots so
# a hip sim/real mismatch surfaces as a sign that the problem is PID gains,
# not the knee torque-speed envelope.
# ----------------------------------------------------------------------------

I_POS_TAU = 0
I_POS_DQ_CUT = 1
I_NEG_TAU = 2
I_NEG_DQ_CUT = 3
I_POS_DQ_FLAT = 4
I_NEG_DQ_FLAT = 5

THETA0 = [96.9972, 14.0, 96.9972, -15.0, 0.0, 0.0]

BOUNDS = [
    (50.0, 150.0),   # pos_tau_stall
    (8.0, 25.0),     # pos_dq_cutoff
    (50.0, 150.0),   # neg_tau_stall
    (-25.0, -8.0),   # neg_dq_cutoff
    (-5.0, 10.0),    # pos_dq_flat_start
    (-10.0, 5.0),    # neg_dq_flat_end
]

# Real-robot PID gains from rl_deploy/spot/constants.py::set_default_gains.
# Kept here so the diagnostic table is a pure read even if constants.py moves.
REAL_GAINS = {
    "hx": {"k_q_p": 624.0, "k_qd_p": 5.20},
    "hy": {"k_q_p": 936.0, "k_qd_p": 5.20},
    "kn": {"k_q_p": 286.0, "k_qd_p": 2.04},
}


def theta_to_limit_arrays(theta):
    pos_limits = [
        [-30.0, theta[I_POS_TAU]],
        [float(theta[I_POS_DQ_FLAT]), float(theta[I_POS_TAU])],
        [float(theta[I_POS_DQ_CUT]), 0.0],
    ]
    neg_limits = [
        [float(theta[I_NEG_DQ_CUT]), 0.0],
        [float(theta[I_NEG_DQ_FLAT]), -float(theta[I_NEG_TAU])],
        [30.0, -float(theta[I_NEG_TAU])],
    ]
    return pos_limits, neg_limits


def _sanitize_theta(theta):
    out = list(theta)
    for i, (lo, hi) in enumerate(BOUNDS):
        out[i] = float(min(max(out[i], lo), hi))
    if out[I_POS_DQ_FLAT] >= out[I_POS_DQ_CUT] - 0.5:
        out[I_POS_DQ_FLAT] = out[I_POS_DQ_CUT] - 0.5
    if out[I_NEG_DQ_FLAT] <= out[I_NEG_DQ_CUT] + 0.5:
        out[I_NEG_DQ_FLAT] = out[I_NEG_DQ_CUT] + 0.5
    return out


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _resolve_hdf5_paths(args_cli):
    if args_cli.hdf5_dir is not None:
        paths = sorted(args_cli.hdf5_dir.glob("spot_isaac_real_*.hdf5"))
        if not paths:
            raise FileNotFoundError(
                f"No spot_isaac_real_*.hdf5 files in {args_cli.hdf5_dir}")
        return paths
    return list(args_cli.hdf5_file)


def _load_real_data(hdf5_path, leg_indices, knee_local_idx, num_steps,
                    warmup, n_full_joints, arm_defaults, h5py, np):
    """Load one HDF5 log. Returns dict with padded actions + per-joint
    real-robot torques, velocities, and positions for all 12 leg joints.

    `leg_indices` maps leg-joint-index (0..11) -> index in ORDERED_JOINT_NAMES.
    `knee_local_idx` is the subset of 0..11 that are knees (for cost).
    """
    with h5py.File(hdf5_path, "r") as f:
        actions = f["commanded_action"][:]
        real_loads = f["raw_joint_loads"][:]
        real_vels = f["raw_joint_velocities"][:]
        # raw_joint_positions is what the HDF5 recorder actually writes
        # (see rl_deploy/orbit/onnx_command_generator.py). If older logs
        # don't have it, fall back to zeros — plots degrade gracefully.
        if "raw_joint_positions" in f:
            real_pos = f["raw_joint_positions"][:]
        else:
            real_pos = np.zeros_like(real_loads)

    total = actions.shape[0]
    n = min(num_steps, total) if num_steps > 0 else total
    warmup_n = min(warmup, n - 1)

    actions = actions[:n]
    # Pad 12-dim (legs only) actions to 19-dim if needed.
    if actions.shape[1] < n_full_joints:
        pad = np.tile(arm_defaults, (n, 1))
        actions = np.concatenate([actions, pad], axis=1).astype(np.float32)

    return {
        "actions": actions,
        "commanded_leg_pos": actions[warmup_n:n, leg_indices],
        "real_torques_leg": real_loads[:n, leg_indices][warmup_n:],
        "real_velocities_leg": real_vels[:n, leg_indices][warmup_n:],
        "real_positions_leg": real_pos[:n, leg_indices][warmup_n:],
        "knee_local_idx": knee_local_idx,
        "num_steps": n,
        "warmup": warmup_n,
        "path": hdf5_path,
    }


# ----------------------------------------------------------------------------
# Gain introspection / override
# ----------------------------------------------------------------------------

def _get_leg_gains_tensor(actuator, joint_order, actuator_joint_names):
    """Read stiffness/damping for an actuator group; returns dicts by name."""
    stiffness = actuator.stiffness[0].detach().cpu().numpy()
    damping = actuator.damping[0].detach().cpu().numpy()
    out = {}
    for local_i, jname in enumerate(actuator_joint_names):
        out[jname] = (float(stiffness[local_i]), float(damping[local_i]))
    return out


def _print_gain_table(env):
    """Print a sim-vs-real PID table. Helps the user see the scale of the
    mismatch before trusting any torque-distribution plot.
    """
    robot = env.scene["robot"]
    hip_act = robot.actuators["spot_hip"]
    knee_act = robot.actuators["spot_knee"]

    def _one_row(name, sim_k, sim_d, real_k, real_d):
        k_ratio = real_k / sim_k if sim_k > 1e-6 else float("nan")
        d_ratio = real_d / sim_d if sim_d > 1e-6 else float("nan")
        return (f"  {name:6s}  sim K={sim_k:7.2f}  D={sim_d:6.2f}   "
                f"real K={real_k:7.2f}  D={real_d:6.2f}   "
                f"ratio K={k_ratio:5.2f}  D={d_ratio:5.2f}")

    print("\n=== Sim vs Real PID gains (per-joint-class) ===")
    sim_hip_k = float(hip_act.stiffness[0, 0].detach().cpu())
    sim_hip_d = float(hip_act.damping[0, 0].detach().cpu())
    sim_knee_k = float(knee_act.stiffness[0, 0].detach().cpu())
    sim_knee_d = float(knee_act.damping[0, 0].detach().cpu())

    # Note: sim uses a single stiffness/damping for all .*_h[xy] (one hip
    # actuator group), while real distinguishes hx from hy.  We print both
    # real rows against the same sim row so the under-tuning is obvious.
    print(_one_row("hip_hx", sim_hip_k, sim_hip_d,
                   REAL_GAINS["hx"]["k_q_p"], REAL_GAINS["hx"]["k_qd_p"]))
    print(_one_row("hip_hy", sim_hip_k, sim_hip_d,
                   REAL_GAINS["hy"]["k_q_p"], REAL_GAINS["hy"]["k_qd_p"]))
    print(_one_row("knee",   sim_knee_k, sim_knee_d,
                   REAL_GAINS["kn"]["k_q_p"], REAL_GAINS["kn"]["k_qd_p"]))
    print("  (sim \"hip\" is a single actuator group for both hx and hy)\n")


def _apply_gain_overrides(env, args_cli):
    """Optionally overwrite actuator stiffness/damping in-place."""
    robot = env.scene["robot"]
    hip_act = robot.actuators["spot_hip"]
    knee_act = robot.actuators["spot_knee"]

    def _set(actuator, tag, k_override, d_override):
        if k_override is not None:
            actuator.stiffness[:] = float(k_override)
            print(f"  override {tag} stiffness -> {k_override}")
        if d_override is not None:
            actuator.damping[:] = float(d_override)
            print(f"  override {tag} damping   -> {d_override}")

    touched = any(v is not None for v in [
        args_cli.override_knee_stiffness, args_cli.override_knee_damping,
        args_cli.override_hip_stiffness, args_cli.override_hip_damping,
    ])
    if not touched:
        return
    print("=== Applying PID gain overrides ===")
    _set(hip_act, "hip",
         args_cli.override_hip_stiffness, args_cli.override_hip_damping)
    _set(knee_act, "knee",
         args_cli.override_knee_stiffness, args_cli.override_knee_damping)
    print()


# ----------------------------------------------------------------------------
# Rollout + cost
# ----------------------------------------------------------------------------

def _apply_theta(knee_actuator, theta, device, LinearInterpolation, torch):
    pos_limits, neg_limits = theta_to_limit_arrays(theta)
    pos_data = torch.tensor(pos_limits, device=device)
    neg_data = torch.tensor(neg_limits, device=device)
    knee_actuator._pos_torque_speed_data = pos_data
    knee_actuator._neg_torque_speed_data = neg_data
    knee_actuator._pos_torque_speed_limit = LinearInterpolation(
        pos_data[:, 0], pos_data[:, 1], device=device
    )
    knee_actuator._neg_torque_speed_limit = LinearInterpolation(
        neg_data[:, 0], neg_data[:, 1], device=device
    )


def _build_rollout_fn(env, device, leg_indices):
    """Returns a closure: rollout(actions_np, num_steps, warmup) -> dict.

    Open-loop replay: directly steps the env with recorded joint position
    targets, bypassing the ONNX policy. Records torque, velocity, AND
    position for all 12 leg joints so diagnostic plots can compare sim
    vs real for hips too (CMA-ES still only costs on knees).
    """
    import numpy as np
    import torch

    n_leg = len(leg_indices)

    def rollout(actions_np, num_steps, warmup):
        actions_t = torch.tensor(actions_np, dtype=torch.float32, device=device)

        env.reset()

        sim_tau = np.zeros((num_steps, n_leg), dtype=np.float32)
        sim_vel = np.zeros((num_steps, n_leg), dtype=np.float32)
        sim_pos = np.zeros((num_steps, n_leg), dtype=np.float32)
        for i in range(num_steps):
            obs_dict, _ = env.step(actions_t[i : i + 1])
            effort = obs_dict["spot"]["joint_effort"][0].cpu().numpy()
            vel = obs_dict["spot"]["joint_vel"][0].cpu().numpy()
            pos = obs_dict["spot"]["joint_pos"][0].cpu().numpy()
            sim_tau[i] = effort[leg_indices]
            sim_vel[i] = vel[leg_indices]
            sim_pos[i] = pos[leg_indices]

        return {
            "torques": sim_tau[warmup:],
            "velocities": sim_vel[warmup:],
            "positions": sim_pos[warmup:],
        }

    return rollout


def _wasserstein_cost(real, sim, indices, wasserstein_distance):
    """Sum of 1D Wasserstein distances over a subset of joint columns."""
    cost = 0.0
    for j in indices:
        cost += wasserstein_distance(real[:, j], sim[:, j])
    return float(cost)


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------

def _dump_best_yaml(path, theta):
    pos_limits, neg_limits = theta_to_limit_arrays(theta)
    lines = [
        "# Optimized knee torque-speed breakpoints. Paste into",
        "# rl_deploy/isaaclab_spot/spot_knee_actuator.py.",
        "POS_TORQUE_SPEED_LIMIT:",
    ]
    for p in pos_limits:
        lines.append(f"  - [{p[0]:.4f}, {p[1]:.4f}]")
    lines.append("NEG_TORQUE_SPEED_LIMIT:")
    for p in neg_limits:
        lines.append(f"  - [{p[0]:.4f}, {p[1]:.4f}]")
    path.write_text("\n".join(lines) + "\n")


def _stats_line(a):
    import numpy as np
    return (f"mean={np.mean(a):+6.2f}  std={np.std(a):5.2f}  "
            f"p50={np.percentile(a, 50):+6.2f}  p95={np.percentile(a, 95):+6.2f}")


def _plot_compare(path, real_knee_torques, sim_before, sim_after, knee_names):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(len(knee_names), 1, figsize=(10, 2.8 * len(knee_names)))
    if len(knee_names) == 1:
        axes = [axes]
    bins = np.linspace(
        min(real_knee_torques.min(), sim_before.min(), sim_after.min()) - 5,
        max(real_knee_torques.max(), sim_before.max(), sim_after.max()) + 5,
        80,
    )
    for j, name in enumerate(knee_names):
        ax = axes[j]
        ax.hist(real_knee_torques[:, j], bins=bins, alpha=0.5,
                color="blue", label="Real", density=True)
        ax.hist(sim_before[:, j], bins=bins, alpha=0.4,
                color="gray", label="Sim (initial)", density=True,
                histtype="step", linewidth=1.5)
        ax.hist(sim_after[:, j], bins=bins, alpha=0.5,
                color="red", label="Sim (calibrated)", density=True,
                histtype="step", linewidth=1.5)
        ax.set_title(f"Torque distribution: {name}")
        ax.set_xlabel("Effort (Nm)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        stats_text = (
            "real:  " + _stats_line(real_knee_torques[:, j]) + "\n"
            "init:  " + _stats_line(sim_before[:, j]) + "\n"
            "calib: " + _stats_line(sim_after[:, j])
        )
        ax.text(0.01, 0.98, stats_text, transform=ax.transAxes,
                fontsize=7, family="monospace", va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_torque_velocity_envelope(path, real_vels, real_torques,
                                   sim_vels, sim_torques,
                                   theta_initial, theta_calibrated, knee_names):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    t_color = np.linspace(0.0, 1.0, real_vels.shape[0])

    for j, (ax, name) in enumerate(zip(axes, knee_names)):
        # Real samples colored by time so static clouds vs sweeps are visible.
        ax.scatter(real_vels[:, j], real_torques[:, j], s=2, c=t_color,
                   cmap="Blues", alpha=0.5, label="Real (time→)",
                   rasterized=True)
        # Sim samples at calibrated theta (orange), sparser marker.
        ax.scatter(sim_vels[:, j], sim_torques[:, j], s=2, alpha=0.25,
                   color="darkorange", label="Sim (calibrated)",
                   rasterized=True)

        for theta, color, lbl, lw in [
            (theta_initial, "red", "Initial limits", 2.0),
            (theta_calibrated, "limegreen", "Calibrated limits", 2.0),
        ]:
            pos_limits, neg_limits = theta_to_limit_arrays(theta)
            pos_x = [p[0] for p in pos_limits]
            pos_y = [p[1] for p in pos_limits]
            neg_x = [p[0] for p in neg_limits]
            neg_y = [p[1] for p in neg_limits]
            ax.plot(pos_x, pos_y, color=color, linewidth=lw, label=f"{lbl} (+)")
            ax.plot(neg_x, neg_y, color=color, linewidth=lw, linestyle="--",
                    label=f"{lbl} (-)")

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Joint velocity (rad/s)")
        ax.set_ylabel("Joint torque (Nm)")
        ax.set_xlim(-32, 32)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Torque-Velocity: Real (blue, time-colored) vs Sim Envelope",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_timeseries(path, real_torques, sim_initial, sim_calibrated, knee_names):
    import matplotlib.pyplot as plt
    import numpy as np

    n_steps = min(real_torques.shape[0], sim_initial.shape[0], sim_calibrated.shape[0])
    t = np.arange(n_steps)
    fig, axes = plt.subplots(len(knee_names), 1,
                             figsize=(14, 2.8 * len(knee_names)), sharex=True)
    if len(knee_names) == 1:
        axes = [axes]

    for j, (ax, name) in enumerate(zip(axes, knee_names)):
        ax.plot(t, real_torques[:n_steps, j], color="steelblue", alpha=0.8,
                linewidth=0.8, label="Real")
        ax.plot(t, sim_initial[:n_steps, j], color="gray", alpha=0.6,
                linewidth=0.8, linestyle="--", label="Sim (initial)")
        ax.plot(t, sim_calibrated[:n_steps, j], color="crimson", alpha=0.7,
                linewidth=0.8, label="Sim (calibrated)")
        ax.set_ylabel("Torque (Nm)")
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Timestep (after warmup)")
    fig.suptitle("Knee Torque Time Series: Real vs Sim", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_convergence(path, baseline_cost, gen_best_costs, gen_all_costs):
    """Best-so-far line AND scatter of every candidate cost per generation.
    Works even if only one generation ran.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(baseline_cost, color="gray", linestyle=":", linewidth=1,
               label=f"Baseline ({baseline_cost:.3f})")

    for gen_i, costs in enumerate(gen_all_costs):
        xs = np.full(len(costs), gen_i) + np.random.uniform(-0.1, 0.1, len(costs))
        ax.scatter(xs, costs, s=18, color="steelblue", alpha=0.5,
                   label="Candidate cost" if gen_i == 0 else None)

    if gen_best_costs:
        gens = np.arange(len(gen_best_costs))
        ax.plot(gens, gen_best_costs, "o-", color="crimson", markersize=5,
                label="Best-so-far")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Wasserstein cost (sum over knees, mean over files)")
    ax.set_title("CMA-ES Convergence (all candidates + best-so-far)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_per_knee_wasserstein(path, real_torques, sim_initial, sim_calibrated,
                               knee_names, wasserstein_distance):
    import matplotlib.pyplot as plt
    import numpy as np

    n_knees = len(knee_names)
    costs_before = [wasserstein_distance(real_torques[:, j], sim_initial[:, j])
                    for j in range(n_knees)]
    costs_after = [wasserstein_distance(real_torques[:, j], sim_calibrated[:, j])
                   for j in range(n_knees)]

    x = np.arange(n_knees)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, costs_before, width, color="gray",
           label="Initial", alpha=0.8)
    ax.bar(x + width / 2, costs_after, width, color="crimson",
           label="Calibrated", alpha=0.8)

    for i in range(n_knees):
        if costs_before[i] > 0:
            pct = (costs_before[i] - costs_after[i]) / costs_before[i] * 100
            ax.text(x[i] + width / 2, costs_after[i] + 0.3,
                    f"{pct:+.0f}%", ha="center", fontsize=8, color="crimson")

    ax.set_xticks(x)
    ax.set_xticklabels(knee_names)
    ax.set_ylabel("Wasserstein distance (Nm)")
    ax.set_title("Per-Knee Sim2Real Gap: Before vs After Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------- New diagnostic plots ----------

def _plot_tracking_error(path, commanded_pos, real_pos, sim_pos_before,
                         sim_pos_after, leg_joint_names):
    """Tracking error (commanded - actual) timeseries, all 12 leg joints.
    If sim tracks tightly but real lags at comparable torque, PID gains
    (especially stiffness) are under-tuned in sim.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = min(commanded_pos.shape[0], real_pos.shape[0],
            sim_pos_before.shape[0], sim_pos_after.shape[0])
    t = np.arange(n)

    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
    axes = axes.flatten()

    real_err = commanded_pos[:n] - real_pos[:n]
    sim_err_before = commanded_pos[:n] - sim_pos_before[:n]
    sim_err_after = commanded_pos[:n] - sim_pos_after[:n]

    for j, (ax, name) in enumerate(zip(axes, leg_joint_names)):
        ax.plot(t, real_err[:, j], color="steelblue", alpha=0.8,
                linewidth=0.7, label="Real")
        ax.plot(t, sim_err_before[:, j], color="gray", alpha=0.7,
                linewidth=0.7, linestyle="--", label="Sim (initial)")
        ax.plot(t, sim_err_after[:, j], color="crimson", alpha=0.7,
                linewidth=0.7, label="Sim (calibrated)")
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)

        r_rms = float(np.sqrt(np.mean(real_err[:, j] ** 2)))
        s_rms = float(np.sqrt(np.mean(sim_err_after[:, j] ** 2)))
        ratio = s_rms / r_rms if r_rms > 1e-6 else float("nan")
        ax.set_title(f"{name}  rms real={r_rms:.3f} sim={s_rms:.3f} "
                     f"(sim/real={ratio:.2f})", fontsize=9)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7)
        if j >= 8:
            ax.set_xlabel("Timestep")
        if j % 4 == 0:
            ax.set_ylabel("cmd − q (rad)")

    fig.suptitle("Tracking error (commanded − actual position), all leg joints",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_all_joints_torque_overlay(path, real_tau, sim_tau_before,
                                    sim_tau_after, leg_joint_names):
    """Per-joint torque timeseries across all 12 leg joints. If hip rows
    show real ≫ sim (or vice-versa), the problem is almost certainly PID
    gains, not the knee torque-speed envelope.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = min(real_tau.shape[0], sim_tau_before.shape[0], sim_tau_after.shape[0])
    t = np.arange(n)

    fig, axes = plt.subplots(3, 4, figsize=(18, 9), sharex=True)
    axes = axes.flatten()

    for j, (ax, name) in enumerate(zip(axes, leg_joint_names)):
        ax.plot(t, real_tau[:n, j], color="steelblue", alpha=0.8,
                linewidth=0.7, label="Real")
        ax.plot(t, sim_tau_before[:n, j], color="gray", alpha=0.7,
                linewidth=0.7, linestyle="--", label="Sim (initial)")
        ax.plot(t, sim_tau_after[:n, j], color="crimson", alpha=0.7,
                linewidth=0.7, label="Sim (calibrated)")

        r_mean = float(np.mean(np.abs(real_tau[:n, j])))
        s_mean = float(np.mean(np.abs(sim_tau_after[:n, j])))
        ratio = s_mean / r_mean if r_mean > 1e-6 else float("nan")
        ax.set_title(f"{name}  |tau| real={r_mean:.1f} sim={s_mean:.1f} "
                     f"(sim/real={ratio:.2f})", fontsize=9)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7)
        if j >= 8:
            ax.set_xlabel("Timestep")
        if j % 4 == 0:
            ax.set_ylabel("Torque (Nm)")

    fig.suptitle("Torque timeseries, all leg joints "
                 "(hips are not optimized — diagnostic only)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_wasserstein_landscape(path, theta0, bounds, evaluate, n_points):
    """1D sweep of cost along each theta dim around baseline. Flat curves
    explain why CMA-ES makes no progress; sharply-curved curves say the
    parameter matters and more generations would help.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["pos_tau_stall", "pos_dq_cutoff", "neg_tau_stall",
              "neg_dq_cutoff", "pos_dq_flat", "neg_dq_flat"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        lo, hi = bounds[i]
        values = np.linspace(lo, hi, n_points)
        costs = []
        for v in values:
            th = list(theta0)
            th[i] = float(v)
            c, _ = evaluate(th)
            costs.append(c)
        ax.plot(values, costs, "o-", color="crimson")
        ax.axvline(theta0[i], color="gray", linestyle=":", linewidth=1,
                   label=f"baseline ({theta0[i]:.2f})")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("parameter value")
        ax.set_ylabel("Wasserstein cost")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("1D Wasserstein cost sweeps (baseline theta, one dim at a time)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Verdict print
# ----------------------------------------------------------------------------

def _print_diagnostic_verdict(leg_joint_names, real_tau, sim_tau_after,
                              commanded_pos, real_pos, sim_pos_after,
                              knee_local_idx, wasserstein_distance):
    """Print a compact per-joint table and a heuristic recommendation so the
    run ends with a verdict, not 7 PNGs for the user to interpret by hand.
    """
    import numpy as np

    print("\n=== Sim/Real Diagnostic ===")
    header = (f"  {'joint':8s}  {'|tau|_real':>11s}  {'|tau|_sim':>11s}  "
              f"{'t_ratio':>8s}  {'rms_err_real':>13s}  {'rms_err_sim':>12s}  "
              f"{'W(tau)':>8s}  {'role':>7s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    t_ratios = []
    err_ratios = []
    w_by_name = {}

    for j, name in enumerate(leg_joint_names):
        r_tau = float(np.mean(np.abs(real_tau[:, j])))
        s_tau = float(np.mean(np.abs(sim_tau_after[:, j])))
        t_ratio = s_tau / r_tau if r_tau > 1e-6 else float("nan")

        r_err = commanded_pos[:, j] - real_pos[:, j]
        s_err = commanded_pos[:, j] - sim_pos_after[:, j]
        r_rms = float(np.sqrt(np.mean(r_err ** 2)))
        s_rms = float(np.sqrt(np.mean(s_err ** 2)))
        err_ratio = s_rms / r_rms if r_rms > 1e-6 else float("nan")

        w = wasserstein_distance(real_tau[:, j], sim_tau_after[:, j])
        w_by_name[name] = w

        role = "KNEE*" if j in knee_local_idx else "hip"
        print(f"  {name:8s}  {r_tau:11.2f}  {s_tau:11.2f}  "
              f"{t_ratio:8.2f}  {r_rms:13.4f}  {s_rms:12.4f}  "
              f"{w:8.2f}  {role:>7s}")

        t_ratios.append(t_ratio)
        err_ratios.append(err_ratio)

    # Split hip vs knee for the heuristic.
    knee_t = [t_ratios[j] for j in knee_local_idx]
    hip_t = [t_ratios[j] for j in range(len(leg_joint_names))
             if j not in knee_local_idx]
    knee_err = [err_ratios[j] for j in knee_local_idx]
    hip_err = [err_ratios[j] for j in range(len(leg_joint_names))
               if j not in knee_local_idx]

    def _nan_median(xs):
        import numpy as np
        vals = [x for x in xs if not (x != x)]  # drop NaN
        return float(np.median(vals)) if vals else float("nan")

    knee_t_m = _nan_median(knee_t)
    hip_t_m = _nan_median(hip_t)
    knee_err_m = _nan_median(knee_err)
    hip_err_m = _nan_median(hip_err)

    print(f"\n  medians: knee |tau| ratio={knee_t_m:.2f}  "
          f"hip |tau| ratio={hip_t_m:.2f}  "
          f"knee err ratio={knee_err_m:.2f}  hip err ratio={hip_err_m:.2f}")

    # Heuristic verdict. Intentionally cautious: prints multiple lines when
    # multiple signals match rather than choosing one.
    notes = []
    if hip_t_m < 0.7 and 0.7 <= knee_t_m <= 1.3:
        notes.append(
            "  -> Hip torques are much lower in sim than real while knees match: "
            "this looks like a PID gain mismatch (real K_Q_P hip ≈ 624/936, "
            "sim ≈ 60). The knee torque-speed envelope is probably NOT the "
            "bottleneck.")
    if knee_t_m > 1.5:
        notes.append(
            "  -> Sim knee torques overshoot real. Envelope calibration is "
            "appropriate; consider increasing --generations and --popsize.")
    if hip_err_m < 0.5 and knee_err_m < 0.5:
        notes.append(
            "  -> Sim tracks commanded positions much tighter than real. "
            "Raise sim stiffness (try --override_knee_stiffness 286 "
            "--override_hip_stiffness 624) and re-run.")
    if 0.8 <= knee_t_m <= 1.2 and 0.8 <= hip_t_m <= 1.2:
        notes.append(
            "  -> Torques match within 20% everywhere; remaining Wasserstein "
            "distance is likely noise / sensor delay, not a model error.")
    if not notes:
        notes.append("  -> No single hypothesis dominates. Inspect plots.")

    print("\n  Suggested next step:")
    for n in notes:
        print(n)
    print()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args_cli = _parse_args()
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import h5py
    import numpy as np
    from cmaes import CMA
    from scipy.stats import wasserstein_distance
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils import LinearInterpolation
    import torch

    from rl_deploy.isaaclab_spot.spot_env import SpotFlatEnvCfg
    from rl_deploy.spot.constants import (
        ORDERED_JOINT_NAMES_SPOT,
        ORDERED_JOINT_NAMES_SPOT_ARM,
        ORDERED_JOINT_NAMES_SPOT_BASE,
    )
    from rl_deploy.isaaclab_spot.isaac_model import SPOT_DEFAULT_JOINT_POS

    # --- Resolve HDF5 files ---
    hdf5_paths = _resolve_hdf5_paths(args_cli)
    print(f"Using {len(hdf5_paths)} HDF5 file(s):")
    for p in hdf5_paths:
        print(f"  {p}")

    # --- Joint setup ---
    leg_joint_names = list(ORDERED_JOINT_NAMES_SPOT_BASE)  # 12 joints
    leg_indices = [ORDERED_JOINT_NAMES_SPOT.index(n) for n in leg_joint_names]
    knee_local_idx = [i for i, n in enumerate(leg_joint_names)
                      if n.endswith("_kn")]
    knee_names = [leg_joint_names[i] for i in knee_local_idx]

    n_full = len(ORDERED_JOINT_NAMES_SPOT)
    arm_defaults = np.array(
        [SPOT_DEFAULT_JOINT_POS[n] for n in ORDERED_JOINT_NAMES_SPOT_ARM],
        dtype=np.float32,
    )[np.newaxis, :]  # shape (1, 7)

    # --- Load all real datasets ---
    datasets = []
    for p in hdf5_paths:
        d = _load_real_data(p, leg_indices, knee_local_idx,
                            args_cli.num_steps, args_cli.warmup_steps,
                            n_full, arm_defaults, h5py, np)
        real_knee = d["real_torques_leg"][:, knee_local_idx]
        print(f"  {p.name}: {d['num_steps']} steps, "
              f"knee torque range [{real_knee.min():.1f}, {real_knee.max():.1f}] Nm")
        datasets.append(d)

    # Aggregate real data for cross-file plots.
    all_real_tau_leg = np.concatenate([d["real_torques_leg"] for d in datasets], axis=0)
    all_real_vel_leg = np.concatenate([d["real_velocities_leg"] for d in datasets], axis=0)
    all_real_pos_leg = np.concatenate([d["real_positions_leg"] for d in datasets], axis=0)
    all_cmd_pos_leg = np.concatenate([d["commanded_leg_pos"] for d in datasets], axis=0)
    print(f"Total real samples: {all_real_tau_leg.shape[0]} (per leg joint)")

    # --- Build environment (once) ---
    env_cfg = SpotFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedEnv(env_cfg)

    knee_actuator = env.scene["robot"].actuators["spot_knee"]
    device = knee_actuator._pos_torque_speed_data.device

    # Diagnostic: show how far sim gains are from the real K_Q_P/K_QD_P values
    # BEFORE any override. The user's 2026 question was exactly this.
    _print_gain_table(env)

    # Optional gain overrides (apply once, before any rollout).
    _apply_gain_overrides(env, args_cli)
    if any(v is not None for v in [
        args_cli.override_knee_stiffness, args_cli.override_knee_damping,
        args_cli.override_hip_stiffness, args_cli.override_hip_damping,
    ]):
        print("=== Sim PID gains after override ===")
        _print_gain_table(env)

    rollout = _build_rollout_fn(env, device, leg_indices)

    def evaluate(theta):
        """Open-loop replay over all datasets. Cost is knee-only (ReLIC);
        aggregated hip+knee data is carried back for diagnostic plots.
        """
        theta = _sanitize_theta(theta)
        _apply_theta(knee_actuator, theta, device, LinearInterpolation, torch)

        total_cost = 0.0
        sim_tau_list, sim_vel_list, sim_pos_list = [], [], []
        for d in datasets:
            result = rollout(d["actions"], d["num_steps"], d["warmup"])
            cost_i = _wasserstein_cost(
                d["real_torques_leg"], result["torques"],
                knee_local_idx, wasserstein_distance,
            )
            total_cost += cost_i
            sim_tau_list.append(result["torques"])
            sim_vel_list.append(result["velocities"])
            sim_pos_list.append(result["positions"])

        mean_cost = total_cost / len(datasets)
        return mean_cost, {
            "sim_tau_leg": np.concatenate(sim_tau_list, axis=0),
            "sim_vel_leg": np.concatenate(sim_vel_list, axis=0),
            "sim_pos_leg": np.concatenate(sim_pos_list, axis=0),
        }

    args_cli.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Baseline ---
    print("Evaluating baseline theta...")
    baseline_cost, baseline_result = evaluate(THETA0)
    print(f"Baseline Wasserstein cost: {baseline_cost:.4f}")

    # --- CMA-ES ---
    theta0 = np.array(THETA0, dtype=np.float64)
    sigma0 = float(args_cli.sigma) * float(np.mean(np.abs(theta0)))
    bounds_arr = np.array(BOUNDS, dtype=np.float64)
    popsize = max(args_cli.popsize, 4 + int(3 * np.log(len(THETA0))))
    if popsize != args_cli.popsize:
        print(f"Adjusted popsize {args_cli.popsize} -> {popsize} "
              f"(minimum for {len(THETA0)}-dim)")
    optimizer = CMA(
        mean=theta0.copy(),
        sigma=sigma0,
        bounds=bounds_arr,
        population_size=popsize,
        seed=42,
    )

    history_path = args_cli.out_dir / "knee_calibration_history.csv"
    best_cost = baseline_cost
    best_theta = list(THETA0)
    best_result = baseline_result
    gen_best_costs = []
    gen_all_costs = []  # list of lists, for scatter plot

    with history_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation", "candidate", "cost",
                         *[f"theta_{i}" for i in range(6)]])
        writer.writerow([-1, 0, baseline_cost, *THETA0])

        for gen in range(args_cli.generations):
            solutions = []
            costs_this_gen = []
            for cand in range(optimizer.population_size):
                theta = optimizer.ask()
                cost, result = evaluate(theta)
                solutions.append((theta, cost))
                costs_this_gen.append(cost)
                writer.writerow([gen, cand, cost, *theta.tolist()])
                fh.flush()
                if cost < best_cost:
                    best_cost = cost
                    best_theta = _sanitize_theta(theta)
                    best_result = result
                    print(f"  gen {gen} cand {cand}: new best {cost:.4f}")
            optimizer.tell(solutions)
            gen_best_costs.append(best_cost)
            gen_all_costs.append(costs_this_gen)
            print(f"gen {gen} done. best so far: {best_cost:.4f}")
            if optimizer.should_stop():
                print("CMA-ES converged / stopped early.")
                break

    # --- Outputs ---
    best_yaml = args_cli.out_dir / "knee_calibration_best.yaml"
    _dump_best_yaml(best_yaml, best_theta)
    print(f"Wrote {best_yaml}")

    real_knee_tau = all_real_tau_leg[:, knee_local_idx]
    sim_before_knee = baseline_result["sim_tau_leg"][:, knee_local_idx]
    sim_after_knee = best_result["sim_tau_leg"][:, knee_local_idx]

    real_knee_vel = all_real_vel_leg[:, knee_local_idx]
    sim_after_knee_vel = best_result["sim_vel_leg"][:, knee_local_idx]

    # 1. Torque histogram comparison (knees only).
    p = args_cli.out_dir / "knee_calibration_compare.png"
    _plot_compare(p, real_knee_tau, sim_before_knee, sim_after_knee, knee_names)
    print(f"Wrote {p}")

    # 2. Torque-Velocity scatter + envelope (ReLIC Fig 11), with sim overlay.
    p = args_cli.out_dir / "knee_calibration_torque_velocity.png"
    _plot_torque_velocity_envelope(
        p, real_knee_vel, real_knee_tau,
        sim_after_knee_vel, sim_after_knee,
        THETA0, best_theta, knee_names,
    )
    print(f"Wrote {p}")

    # 3. Time-series overlay (first dataset only for readability).
    first_d = datasets[0]
    first_n = first_d["real_torques_leg"].shape[0]
    first_real_knee = first_d["real_torques_leg"][:, knee_local_idx]
    p = args_cli.out_dir / "knee_calibration_timeseries.png"
    _plot_timeseries(
        p, first_real_knee,
        sim_before_knee[:first_n],
        sim_after_knee[:first_n],
        knee_names,
    )
    print(f"Wrote {p}")

    # 4. CMA-ES convergence (scatter of all candidates + best line).
    p = args_cli.out_dir / "knee_calibration_convergence.png"
    _plot_convergence(p, baseline_cost, gen_best_costs, gen_all_costs)
    print(f"Wrote {p}")

    # 5. Per-knee Wasserstein breakdown.
    p = args_cli.out_dir / "knee_calibration_per_knee.png"
    _plot_per_knee_wasserstein(
        p, real_knee_tau, sim_before_knee, sim_after_knee,
        knee_names, wasserstein_distance,
    )
    print(f"Wrote {p}")

    # 6. NEW: tracking error across all 12 leg joints (PID-gain diagnostic).
    p = args_cli.out_dir / "tracking_error.png"
    _plot_tracking_error(
        p, all_cmd_pos_leg, all_real_pos_leg,
        baseline_result["sim_pos_leg"], best_result["sim_pos_leg"],
        leg_joint_names,
    )
    print(f"Wrote {p}")

    # 7. NEW: torque overlay across all 12 leg joints (hip/knee side-by-side).
    p = args_cli.out_dir / "all_joints_torque_overlay.png"
    _plot_all_joints_torque_overlay(
        p, all_real_tau_leg,
        baseline_result["sim_tau_leg"], best_result["sim_tau_leg"],
        leg_joint_names,
    )
    print(f"Wrote {p}")

    # 8. NEW: 1D Wasserstein landscape (explains flat CMA-ES runs).
    if not args_cli.skip_landscape:
        p = args_cli.out_dir / "wasserstein_landscape.png"
        _plot_wasserstein_landscape(
            p, THETA0, BOUNDS, evaluate, args_cli.landscape_points,
        )
        print(f"Wrote {p}")

    # --- Final summary ---
    print(f"\nBaseline cost: {baseline_cost:.4f}")
    print(f"Best cost:     {best_cost:.4f}  (improvement: {baseline_cost - best_cost:.4f})")
    print(f"Best theta:    {best_theta}")

    _print_diagnostic_verdict(
        leg_joint_names, all_real_tau_leg, best_result["sim_tau_leg"],
        all_cmd_pos_leg, all_real_pos_leg, best_result["sim_pos_leg"],
        knee_local_idx, wasserstein_distance,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
