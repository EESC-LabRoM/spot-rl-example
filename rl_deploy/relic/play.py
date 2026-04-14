# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--center", action="store_true", default=False, help="Look at the robot."
)
# append RSL-RL cli arguments

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym

# Import extensions to set up environment tasks
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils import parse_env_cfg
from tqdm import tqdm
import onnxruntime as ort


from rl_deploy.relic.interlimb_env_cfg import SpotInterlimbEnvCfg_Phase_1
##
# Register Gym environments.
##

gym.register(
    id="Isaac-Spot-Interlimb-Phase-1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SpotInterlimbEnvCfg_Phase_1,
    },
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        "Isaac-Spot-Interlimb-Phase-1-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.center:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, 3.0, 3.0)
        env_cfg.viewer.resolution = (1920, 1080)  # (4096, 2160)

    # create isaac environment
    env = gym.make(
        "Isaac-Spot-Interlimb-Phase-1-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    inference_session = ort.InferenceSession("rl_deploy/configs/policy.onnx")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    progress_bar = (
        tqdm(total=args_cli.video_length, desc="Recording video")
        if args_cli.video
        else None
    )
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = inference_session.run(None, {"obs": obs["policy"].cpu().numpy()})[
                0
            ].tolist()[0]
            # env stepping
            obs, _, _, _ = env.step(
                torch.tensor(actions, device=args_cli.device).unsqueeze(0)
            )
        if args_cli.video:
            timestep += 1
            progress_bar.update(1)
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    if progress_bar is not None:
        progress_bar.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
