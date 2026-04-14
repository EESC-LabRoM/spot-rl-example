"""
The expected effort:

computed_effort = stiffness * error_pos + damping * error_vel + control_action.joint_efforts

The real robot has 80% of the gain of the simulation.

"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(
        description="Compare actual joint loads with IsaacLab predicted loads."
    )
    parser.add_argument(
        "--hdf5_file",
        type=Path,
        default=Path("spot_isaac_real_20260414_194845.hdf5"),
        help="Path to the HDF5 log file.",
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from isaaclab.utils.types import ArticulationActions

    from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT
    from rl_deploy.isaaclab_spot.isaac_model import (
        JOINT_PARAMETER_LOOKUP_TABLE,
        KNEE_DAMPING,
        KNEE_STIFFNESS,
    )
    from rl_deploy.isaaclab_spot.spot_knee_actuator import (
        SpotKneeActuator,
        SpotKneeActuatorCfg,
    )

    out_dir = Path("logs")
    hdf5_path = args_cli.hdf5_file

    if not hdf5_path.exists():
        print(f"File {hdf5_path} does not exist.")
        simulation_app.close()
        return

    print(f"Reading data from {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        joint_positions = f["spot_current_positions"][:]
        joint_velocities = f["spot_current_velocities"][:]
        joint_loads = f["raw_joint_loads"][:]
        commanded_action = f["commanded_action"][:]

    knee_joint_names = [
        name for name in ORDERED_JOINT_NAMES_SPOT if name.endswith("_kn")
    ]
    knee_indices = [ORDERED_JOINT_NAMES_SPOT.index(name) for name in knee_joint_names]

    print(f"Found knee joints: {knee_joint_names} at indices {knee_indices}")

    num_timesteps = joint_positions.shape[0]
    device = "cpu"

    cfg = SpotKneeActuatorCfg(
        joint_names_expr=knee_joint_names,
        effort_limit=None,
        stiffness=KNEE_STIFFNESS / 0.7,
        damping=KNEE_DAMPING,
        enable_torque_speed_limit=True,
        joint_parameter_lookup=JOINT_PARAMETER_LOOKUP_TABLE,
        min_delay=0.0,
        max_delay=0.0,
    )

    actuator = SpotKneeActuator(
        cfg=cfg,
        joint_names=knee_joint_names,
        joint_ids=list(range(len(knee_joint_names))),
        num_envs=num_timesteps,
        device=device,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
    )

    knee_pos = torch.tensor(
        joint_positions[:, knee_indices], dtype=torch.float32, device=device
    ).contiguous()
    knee_vel = torch.tensor(
        joint_velocities[:, knee_indices], dtype=torch.float32, device=device
    ).contiguous()
    knee_loads_actual = joint_loads[:, knee_indices]

    cmd_pos = torch.tensor(
        commanded_action[:, knee_indices], dtype=torch.float32, device=device
    ).contiguous()

    control_action = ArticulationActions(
        joint_positions=cmd_pos,
        joint_velocities=torch.zeros_like(cmd_pos),
        joint_efforts=torch.zeros_like(cmd_pos),
    )

    output_action = actuator.compute(
        control_action=control_action, joint_pos=knee_pos, joint_vel=knee_vel
    )

    knee_loads_predicted = output_action.joint_efforts.detach().cpu().numpy()
    error_pos = (cmd_pos - knee_pos).cpu().numpy()
    knee_vel = knee_vel.cpu().numpy()

    out_dir.mkdir(exist_ok=True, parents=True)

    limit = min(20000, num_timesteps)

    for i, joint_name in enumerate(knee_joint_names):
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        skip_steps = 10_000
        max_limit = num_timesteps  # - skip_steps
        if max_limit <= skip_steps:
            max_limit = num_timesteps
            skip_steps = 0

        valid_range = slice(skip_steps, min(limit + skip_steps, max_limit))
        time_plot = np.arange(skip_steps, min(limit + skip_steps, max_limit))

        # Plot Load
        ax_load = axes[0]
        ax_load.plot(
            time_plot,
            knee_loads_actual[valid_range, i],
            label="Actual Load",
            color="blue",
            alpha=0.7,
        )
        ax_load.plot(
            time_plot,
            knee_loads_predicted[valid_range, i],
            label="Predicted Load",
            color="red",
            alpha=0.7,
            linestyle="--",
        )
        ax_load.set_title(f"Load Comparison: {joint_name}")
        ax_load.set_ylabel("Effort (N.m)")
        ax_load.legend()
        ax_load.grid(True)

        # Plot Position Error
        ax_pos = axes[1]
        ax_pos.plot(
            time_plot,
            knee_pos[valid_range, i],
            label="Position",
            color="green",
            alpha=0.7,
        )
        ax_pos.set_title(f"Position: {joint_name}")
        ax_pos.set_ylabel("Pose (rad)")
        ax_pos.legend()
        ax_pos.grid(True)

        # Plot Velocity Error
        ax_vel = axes[2]
        ax_vel.plot(
            time_plot,
            knee_vel[valid_range, i],
            label="Velocity Error",
            color="orange",
            alpha=0.7,
        )
        ax_vel.set_title(f"Velocity Error: {joint_name}")
        ax_vel.set_xlabel("Timesteps (ignoring first 1s)")
        ax_vel.set_ylabel("Error (rad/s)")
        ax_vel.legend()
        ax_vel.grid(True)

        plt.tight_layout()
        out_img = out_dir / f"actuator_load_comparison_{joint_name}.png"
        plt.savefig(out_img)
        plt.close(fig)
        print(f"Plot saved to {out_img}")

    simulation_app.close()


if __name__ == "__main__":
    main()
