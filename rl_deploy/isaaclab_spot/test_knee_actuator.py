import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test Knee Actuator limits.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.utils.types import ArticulationActions

from rl_deploy.isaaclab_spot.spot_knee_actuator import SpotKneeActuator, SpotKneeActuatorCfg, POS_TORQUE_SPEED_LIMIT, NEG_TORQUE_SPEED_LIMIT
from rl_deploy.isaaclab_spot.isaac_model import JOINT_PARAMETER_LOOKUP_TABLE

def validate_computed_values():
    device = "cpu"
    num_envs = 500

    cfg = SpotKneeActuatorCfg(
        joint_names_expr=[".*_kn"],
        effort_limit=None,
        stiffness=0.0,
        damping=0.0,
        enable_torque_speed_limit=True,
        joint_parameter_lookup=JOINT_PARAMETER_LOOKUP_TABLE,
    )

    actuator = SpotKneeActuator(
        cfg=cfg,
        joint_names=["fl_kn"],
        joint_ids=[0],
        num_envs=num_envs,
        device=device,
        stiffness=cfg.stiffness,
        damping=cfg.damping
    )

    velocities = torch.linspace(-40.0, 40.0, num_envs, device=device).unsqueeze(1)
    
    # Very high effort request (positive)
    pos_effort_request = torch.full_like(velocities, 500.0)
    jd_pos = torch.zeros_like(velocities)
    jd_vel = torch.zeros_like(velocities)
    
    control_action_pos = ArticulationActions(
        joint_positions=jd_pos,
        joint_velocities=jd_vel,
        joint_efforts=pos_effort_request
    )
    
    output_action_pos = actuator.compute(
        control_action=control_action_pos,
        joint_pos=torch.zeros_like(velocities),
        joint_vel=velocities
    )
    actual_pos_efforts = output_action_pos.joint_efforts.squeeze().detach().numpy()

    # Very small effort request (negative)
    neg_effort_request = torch.full_like(velocities, -500.0)
    control_action_neg = ArticulationActions(
        joint_positions=jd_pos,
        joint_velocities=jd_vel,
        joint_efforts=neg_effort_request
    )
    
    output_action_neg = actuator.compute(
        control_action=control_action_neg,
        joint_pos=torch.zeros_like(velocities),
        joint_vel=velocities
    )
    actual_neg_efforts = output_action_neg.joint_efforts.squeeze().detach().numpy()

    vel_np = velocities.squeeze().numpy()

    pos_limit_x = [p[0] for p in POS_TORQUE_SPEED_LIMIT]
    pos_limit_y = [p[1] for p in POS_TORQUE_SPEED_LIMIT]

    neg_limit_x = [p[0] for p in NEG_TORQUE_SPEED_LIMIT]
    neg_limit_y = [p[1] for p in NEG_TORQUE_SPEED_LIMIT]

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(vel_np, actual_pos_efforts, label="Computed +Effort (Requested: 500)", color='blue', linestyle='-')
    plt.plot(vel_np, actual_neg_efforts, label="Computed -Effort (Requested: -500)", color='red', linestyle='-')
    
    # Highlight raw data points
    plt.scatter(pos_limit_x, pos_limit_y, color='cyan', zorder=5, label='Raw POS Limit points')
    plt.scatter(neg_limit_x, neg_limit_y, color='magenta', zorder=5, label='Raw NEG Limit points')

    # Also plot the limit line extending
    plt.plot(pos_limit_x, pos_limit_y, color='cyan', linestyle='--', alpha=0.5)
    plt.plot(neg_limit_x, neg_limit_y, color='magenta', linestyle='--', alpha=0.5)

    plt.xlabel('Joint Velocity (rad/s)')
    plt.ylabel('Joint Effort (N.m)')
    plt.title('Spot Knee Actuator Computed Torque Limits')
    plt.grid(True)
    plt.legend()
    
    output_img = "knee_actuator_limits_validation.png"
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")


if __name__ == "__main__":
    validate_computed_values()
    simulation_app.close()
