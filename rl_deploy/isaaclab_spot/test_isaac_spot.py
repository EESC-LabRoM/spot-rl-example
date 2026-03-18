import argparse
import numpy as np
import h5py
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Setting up a Spot Gripper environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # Force headless mode for testing

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch

from rl_deploy.orbit import orbit_configuration
from rl_deploy.orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from rl_deploy.isaaclab_spot.isaac_spot import IsaacMockSpot


def main():
    """Main function."""
    export_model_dir = "rl_deploy/configs"
    env_config = orbit_configuration.detect_config_file(export_model_dir)
    policy_file = orbit_configuration.detect_policy_file(export_model_dir)
    config = orbit_configuration.load_configuration(env_config)

    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    command_generator = OnnxCommandGenerator(context, config, policy_file, False)

    spot = IsaacMockSpot()
    spot.start_state_stream(state_handler)
    # Read hdf5 file
    dataset = h5py.File("datasets/isaac_spot_dataset_2026-03-18_10-32-46.hdf5", "r")

    demo_0_obs = dataset["data"]["demo_0"]["obs"]["spot"]
    demo_0_policy = dataset["data"]["demo_0"]["obs"]["policy"]
    demo_0_debug = dataset["data"]["demo_0"]["obs"]["debug"]
    demo_0_actions = dataset["data"]["demo_0"]["actions"]
    member_names = list(demo_0_obs.keys())
    itx = 0
    num_iters = demo_0_policy.shape[0]
    for itx in range(min(num_iters, 100)):
        print(itx)
        obs_itx = {
            member_name: torch.tensor([demo_0_obs[member_name][itx]])
            for member_name in member_names
        }
        spot.set_state(obs_itx)
        inputs_dict = command_generator.collect_inputs(
            command_generator._context.latest_state, command_generator._config
        )

        # 1. Compare each observation
        assert np.isclose(
            demo_0_debug["base_lin_vel"][itx], inputs_dict["base_linear_velocity"]
        ).all(), "Base linear velocity does not match"
        assert np.isclose(
            demo_0_debug["base_ang_vel"][itx], inputs_dict["base_angular_velocity"]
        ).all(), "Base angular velocity does not match"
        assert np.isclose(
            demo_0_debug["projected_gravity"][itx], inputs_dict["projected_gravity"]
        ).all(), "Projected gravity does not match"
        assert np.isclose(
            demo_0_debug["velocity_commands"][itx],
            inputs_dict["velocity_cmd"],
            atol=1e-6,
        ).all(), "Last action does not match"
        assert np.isclose(
            demo_0_debug["commands"][itx], inputs_dict["joint_commands"]
        ).all(), "Joint command does not match"
        assert np.isclose(
            demo_0_debug["joint_pos"][itx], inputs_dict["joint_positions"], atol=1e-6
        ).all(), "Joint positions do not match"
        assert np.isclose(
            demo_0_debug["joint_vel"][itx], inputs_dict["joint_velocities"], atol=1e-6
        ).all(), "Joint velocities do not match"
        assert np.isclose(
            demo_0_debug["actions"][itx], inputs_dict["last_action"], atol=1e-5
        ).all(), "Last action does not match"

        # 2. Compare the input to the network
        input_list = []
        for key in [
            "base_linear_velocity",
            "base_angular_velocity",
            "projected_gravity",
            "velocity_cmd",
            "joint_commands",
            "joint_positions",
            "joint_velocities",
            "last_action",
        ]:
            input_list += inputs_dict[key]
        assert np.isclose(demo_0_policy[itx], input_list, atol=1e-5).all(), (
            "Concatenated input does not match"
        )

        # 3. Compare the output action
        output = command_generator._compute_action(input_list)
        assert np.isclose(demo_0_actions["actions"][itx], output, atol=1e-5).all(), (
            "Output action does not match"
        )
        action = command_generator._post_process_action_to_spot(output)
        command_generator._last_action = output

        command_generator.joints_offsets_ordered_base
        _offset = torch.tensor([ 0.1200, -0.1200,  0.1200, -0.1200,  0.5000,  0.5000,  0.5000,  0.5000, -1.0000, -1.0000, -1.0000, -1.0000])
        np.isclose(demo_0_actions["leg_processed_actions"][itx], (torch.tensor(output) * 0.2 + _offset).numpy(), atol=1e-5).all()

        # TODO the leg_processe actions is not on the same order as the action itself


    print("All matched!")


if __name__ == "__main__":
    main()
