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
        default=Path("spot_isaac_real_20260414_171227.hdf5"),
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
        HIP_DAMPING,
        HIP_STIFFNESS,
    )
    from isaaclab.actuators import DelayedPDActuatorCfg, DelayedPDActuator

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

    hip_joint_names = [name for name in ORDERED_JOINT_NAMES_SPOT if name.endswith("hy")]
    hip_indices = [ORDERED_JOINT_NAMES_SPOT.index(name) for name in hip_joint_names]

    print(f"Found hip joints: {hip_joint_names} at indices {hip_indices}")

    num_timesteps = joint_positions.shape[0]
    device = "cpu"

    cfg = DelayedPDActuatorCfg(
        joint_names_expr=hip_joint_names,
        effort_limit=None,
        stiffness=HIP_STIFFNESS,
        damping=HIP_DAMPING,
        min_delay=0.0,
        max_delay=0.0,
    )

    actuator = DelayedPDActuator(
        cfg=cfg,
        joint_names=hip_joint_names,
        joint_ids=list(range(len(hip_joint_names))),
        num_envs=num_timesteps,
        device=device,
        stiffness=cfg.stiffness,
        damping=cfg.damping,
    )

    hip_pos = torch.tensor(
        joint_positions[:, hip_indices], dtype=torch.float32, device=device
    ).contiguous()
    hip_vel = torch.tensor(
        joint_velocities[:, hip_indices], dtype=torch.float32, device=device
    ).contiguous()
    hip_loads_actual = joint_loads[:, hip_indices]

    cmd_pos = torch.tensor(
        commanded_action[:, hip_indices], dtype=torch.float32, device=device
    ).contiguous()

    control_action = ArticulationActions(
        joint_positions=cmd_pos,
        joint_velocities=torch.zeros_like(cmd_pos),
        joint_efforts=torch.zeros_like(cmd_pos),
    )

    output_action = actuator.compute(
        control_action=control_action, joint_pos=hip_pos, joint_vel=hip_vel
    )

    hip_loads_predicted = output_action.joint_efforts.detach().cpu().numpy()
    error_pos = (cmd_pos - hip_pos).cpu().numpy()
    error_vel = hip_vel.cpu().numpy()

    out_dir.mkdir(exist_ok=True, parents=True)

    limit = min(20000, num_timesteps)

    for i, joint_name in enumerate(hip_joint_names):
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        skip_steps = 2_000
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
            hip_loads_actual[valid_range, i],
            label="Actual Load",
            color="blue",
            alpha=0.7,
        )
        ax_load.plot(
            time_plot,
            hip_loads_predicted[valid_range, i],
            label="Simulated Load",
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
            hip_pos[valid_range, i],
            label="Position ",
            color="green",
            alpha=0.7,
        )
        ax_pos.set_title(f"Position : {joint_name}")
        ax_pos.set_ylabel("Pose (rad)")
        ax_pos.legend()
        ax_pos.grid(True)

        # Plot Velocity Error
        ax_vel = axes[2]
        ax_vel.plot(
            time_plot,
            error_vel[valid_range, i],
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
        out_img = out_dir / f"actuator_load_comparison_{joint_name[::-1]}.png"
        plt.savefig(out_img)
        plt.close(fig)
        print(f"Plot saved to {out_img}")

    simulation_app.close()


if __name__ == "__main__":
    main()
