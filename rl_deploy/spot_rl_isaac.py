import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Setting up a Spot Gripper environment.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

parser.add_argument("--hdf5_log", type=str, default="spot_isaac_sim.hdf5", help="Path to save HDF5 log of observations.")

# parse the arguments
args_cli = parser.parse_args()
# args_cli.experience = "isaacsim.exp.full.kit"  # Set the experience here

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import os
import sys

# This allows for absolute imports from 'spot_mgrasping'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from isaaclab.envs import ManagerBasedEnv
from utils.hdf5_logger import HDF5Logger

from rl_deploy.hid.keyboard import Keyboard
from rl_deploy.orbit import orbit_configuration
from rl_deploy.orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from rl_deploy.spot.isaac_spot import IsaacMockSpot
from rl_deploy.spot.spot_env import SpotFlatEnvCfg


def main():
    """Main function."""
    export_model_dir = "rl_deploy/configs"
    env_config = orbit_configuration.detect_config_file(export_model_dir)
    policy_file = orbit_configuration.detect_policy_file(export_model_dir)
    config = orbit_configuration.load_configuration(env_config)

    env_cfg = SpotFlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device


    # wrap for video recording
    env = ManagerBasedEnv(env_cfg)
    
    obs, _ = env.reset()
    logger = HDF5Logger(args_cli.hdf5_log)
    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    command_generator = OnnxCommandGenerator(context, config, policy_file, False, logger=logger)
    gamepad = Keyboard(context, x_vel=0.0, y_vel=0.0, yaw=0.0)

    spot = IsaacMockSpot()


    # Start streams
    spot.start_state_stream(state_handler)

    obs_dict, _ = env.reset()
    obs = obs_dict["spot"]
    spot.set_state(obs)
    spot.start_command_stream(command_generator)

    for i in range(100):
        # run everything in inference mode
        with torch.inference_mode():
            actions = spot.command_update().to(env_cfg.sim.device)
            obs_dict,  _ = env.step(actions)
            obs = obs_dict["spot"]
            spot.set_state(obs)
            # The logger object might not have logger.log so let's log safe
            if logger and hasattr(logger, "log"):
                logger.log(obs_dict)
            gamepad.listen_loop()

    # close the simulator
    env.close()
    logger.save()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()