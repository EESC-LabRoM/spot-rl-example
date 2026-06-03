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
    default=500,
    help="Length of the recorded video (in steps).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

parser.add_argument(
    "--hdf5_log",
    type=str,
    default="spot_isaac_sim.hdf5",
    help="Path to save HDF5 log of observations.",
)

# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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

import gymnasium as gym
import torch
import isaacsim
from pxr import Usd, Sdf
from isaaclab.envs import ManagerBasedEnv
from utils.hdf5_logger import HDF5Logger

from rl_deploy.hid.terminal_keyboard import TerminalKeyboard
from rl_deploy.orbit import orbit_configuration
from rl_deploy.orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from rl_deploy.isaaclab_spot.isaac_spot import IsaacMockSpot
from rl_deploy.isaaclab_spot.spot_env import SpotFlatEnvCfg


class GymnasiumVideoWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env, render_mode=None):
        self.env = env
        self.render_mode = render_mode
        self.cfg = env.cfg
        self.sim = env.sim
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,))
        self.action_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(1,))

    @property
    def unwrapped(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        return self.env

    def step(self, action):
        obs, extra = self.env.step(action)
        return obs, 0.0, False, False, extra

    def reset(self, seed=None, options=None):
        obs, extra = self.env.reset()
        return obs, extra

    def render(self, recompute=False):
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        if self.render_mode == "rgb_array":
            import numpy as np
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            rgb_data = self._rgb_annotator.get_data()
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        return None

    def close(self):
        self.env.close()


def disable_default_lights(stage):
    from pxr import UsdLux
    # Traverse the stage and find all lights
    for prim in stage.Traverse():
        if 'Light' in prim.GetTypeName():
            path = str(prim.GetPath())
            # Disable default lights (those not part of the LightRig we reference)
            if not path.startswith("/World/LightRig/"):
                print(f"[INFO] Disabling default/non-colored light: {path}")
                intensity_attr = prim.GetAttribute("inputs:intensity")
                if intensity_attr:
                    intensity_attr.Set(0.0)
                exposure_attr = prim.GetAttribute("inputs:exposure")
                if exposure_attr:
                    exposure_attr.Set(0.0)


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
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("logs", "videos", "spot_rl_isaac"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during execution.")
        env = GymnasiumVideoWrapper(env, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()
    logger = HDF5Logger(args_cli.hdf5_log)
    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    command_generator = OnnxCommandGenerator(
        context, config, policy_file, False, logger=logger
    )
    gamepad = TerminalKeyboard(context, x_vel=0.0, y_vel=0.0, yaw=0.0)

    spot = IsaacMockSpot()

    # Start streams
    spot.start_state_stream(state_handler)

    obs_dict, _ = env.reset()
    unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
    disable_default_lights(unwrapped_env.sim.stage)

    # Dynamically locate and load the official Colored Lights USDA rig as a reference
    isaacsim_path = os.path.dirname(isaacsim.__file__)
    usda_path = None
    for root, dirs, files in os.walk(isaacsim_path):
        if "Colored_Lights.usda" in files:
            usda_path = os.path.join(root, "Colored_Lights.usda")
            break

    if usda_path:
        print(f"[INFO] Referencing official Colored Lights rig from: {usda_path}")
        light_rig_prim = unwrapped_env.sim.stage.DefinePrim('/World/LightRig', 'Xform')
        light_rig_prim.GetReferences().AddReference(usda_path)
        # Force all lights in the rig to full white
        from pxr import UsdLux, Gf
        for prim in unwrapped_env.sim.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith('/World/LightRig') and 'Light' in prim.GetTypeName():
                color_attr = prim.GetAttribute('inputs:color')
                if color_attr:
                    color_attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))
                    print(f"[INFO] Set light to white: {path}")
                intensity_attr = prim.GetAttribute('inputs:intensity')
                if intensity_attr:
                    intensity_attr.Set(3000.0)
    else:
        print("[WARNING] Could not locate Colored_Lights.usda in isaacsim package!")
    spot.set_state(obs_dict["spot"])
    spot.start_command_stream(command_generator)

    for i in range(20000):
        # run everything in inference mode
        with torch.inference_mode():
            actions = spot.command_update().to(env_cfg.sim.device)
            obs_dict = env.step(actions)[0]
            spot.set_state(obs_dict["spot"])
            # The logger object might not have logger.log so let's log safe
            if logger and hasattr(logger, "log"):
                logger.log(obs_dict)

        # Time-based velocity command: forward for first 10s, then backward
        sim_time = obs_dict["spot"]["sim_time"].item()
        if sim_time < 10.0:
            context.velocity_cmd = [0.5, 0.0, 0.0]   # forward
        else:
            context.velocity_cmd = [-0.5, 0.0, 0.0]  # backward

        if args_cli.video and i + 1 == args_cli.video_length:
            break

    # gamepad.stop_listening()

    # close the simulator
    env.close()
    logger.save()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
