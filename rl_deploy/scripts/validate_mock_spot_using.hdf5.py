import h5py
from matplotlib.style import context
import numpy as np
from rl_deploy.isaaclab_spot.isaac_spot import IsaacMockSpot
from rl_deploy.orbit import orbit_configuration
from rl_deploy.orbit.onnx_command_generator import (
    JOINTS_ORDER_RELIC,
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT_BASE
from rl_deploy.utils.dict_tools import find_ordering, reorder
import torch


class PipelineValidator:
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path

        # 1. Setup Config
        export_model_dir = "rl_deploy/configs"
        env_config = orbit_configuration.detect_config_file(export_model_dir)
        policy_file = orbit_configuration.detect_policy_file(export_model_dir)

        assert env_config is not None, (
            f"No environment config file found in {export_model_dir}"
        )
        assert policy_file is not None, (
            f"No policy ONNX file found in {export_model_dir}"
        )

        config = orbit_configuration.load_configuration(env_config)

        # 2. Setup Context and Generator

        self.context = OnnxControllerContext()
        self.context.velocity_cmd = [0.0, 0.0, 0.0]  # Default, will update per step
        state_handler = StateHandler(self.context)

        self.generator = OnnxCommandGenerator(
            self.context, config, policy_file, verbose=False, mock=False
        )

        self.mock_spot = IsaacMockSpot()
        self.latest_msg = None
        self.mock_spot.start_state_stream(state_handler)

    def validate(self):
        with h5py.File(self.hdf5_path, "r") as file:
            f = file["data/demo_0"]
            num_steps = f["obs/spot/sim_time"].shape[0]
            print(f"Validating {num_steps} steps from {self.hdf5_path}...")

            for i in range(num_steps):
                # -------------------------------------------------------------
                # 1. Reconstruct IsaacLab State Dictionary (Simulated SpotObs)
                # -------------------------------------------------------------
                state_dict = {
                    "joint_pos": torch.tensor(f["obs/spot/joint_pos"][i]).unsqueeze(0),
                    "joint_vel": torch.tensor(f["obs/spot/joint_vel"][i]).unsqueeze(0),
                    "root_lin_vel_w": torch.tensor(
                        f["obs/spot/root_lin_vel_w"][i]
                    ).unsqueeze(0),
                    "root_ang_vel_w": torch.tensor(
                        f["obs/spot/root_ang_vel_w"][i]
                    ).unsqueeze(0),
                    "root_quat_w": torch.tensor(f["obs/spot/root_quat_w"][i]).unsqueeze(
                        0
                    ),
                    "joint_effort": torch.tensor(
                        f["obs/spot/joint_effort"][i]
                    ).unsqueeze(0),
                    "sim_time": torch.tensor(f["obs/spot/sim_time"][i]).unsqueeze(0),
                }

                action_dict = {
                    "actions": torch.tensor(f["processed_actions"][i]).unsqueeze(0),
                }
                # -------------------------------------------------------------
                # 2. Extract Ground Truth Policy Observations (Isaac PolicyCfg)
                # -------------------------------------------------------------
                # Assuming the recorder saved the flat policy buffer in 'obs'
                # If your recorder saves a dictionary of groups, adjust to f["obs/policy"][i]
                # isaac_obs = f["obs/policy"][i]

                # Slice the flat Isaac observation array based on PolicyCfg
                # Update these indices if your sizes differ (e.g., N_DOF)
                isaac_base_lin_vel = torch.tensor(
                    f["obs/spot/base_lin_vel"][i]
                ).unsqueeze(0)
                isaac_base_ang_vel = torch.tensor(
                    f["obs/spot/base_ang_vel"][i]
                ).unsqueeze(0)
                isaac_projected_gravity = torch.tensor(
                    f["obs/spot/projected_gravity"][i]
                ).unsqueeze(0)
                isaac_velocity_cmd = torch.tensor(
                    f["obs/spot/velocity_commands"][i]
                ).unsqueeze(0)
                isaac_joint_pos = torch.tensor(f["obs/spot/joint_pos"][i]).unsqueeze(0)
                isaac_joint_vel = torch.tensor(f["obs/spot/joint_vel"][i]).unsqueeze(0)
                isaac_last_action = torch.tensor(f["obs/spot/actions"][i]).unsqueeze(0)
                print("Step ", i)
                print("base_lin_vel:", isaac_base_lin_vel)
                print("base_ang_vel:", isaac_base_ang_vel)
                print("projected_gravity:", isaac_projected_gravity)
                print("velocity_commands:", isaac_velocity_cmd)
                print("joint_positions:", isaac_joint_pos)
                print("joint_velocities:", isaac_joint_vel)
                print("last_actions:", isaac_last_action)
                # Update context with the actual velocity command & last action from this step
                self.context.velocity_cmd = isaac_velocity_cmd.tolist()
                self.generator._last_action = isaac_last_action.tolist()

                # -------------------------------------------------------------
                # 3. Run Pipeline (IsaacMockSpot -> OnnxCommandGenerator)
                # -------------------------------------------------------------
                self.mock_spot.set_state(state_dict)

                # Collect inputs using your deployed logic
                pipeline_inputs = self.generator.collect_inputs(
                    self.context.latest_state, self.generator._config
                )  # , joint_commands=[i for i in isaac_commands] )

                # -------------------------------------------------------------
                # 4. Compare Pipeline Output vs IsaacLab Ground Truth
                # -------------------------------------------------------------
                #         input_names = [
                #     "base_linear_velocity",
                #     "base_angular_velocity",
                #     "projected_gravity",
                #     "velocity_commands",
                #     "joint_positions",
                #     "joint_velocities",
                #     "last_actions",
                # ]
                self._compare_vectors(
                    "Base Linear Velocity",
                    isaac_base_lin_vel,
                    pipeline_inputs["base_linear_velocity"],
                    step=i,
                )
                self._compare_vectors(
                    "Base Angular Velocity",
                    isaac_base_ang_vel,
                    pipeline_inputs["base_angular_velocity"],
                    step=i,
                )
                self._compare_vectors(
                    "Projected Gravity",
                    isaac_projected_gravity,
                    pipeline_inputs["projected_gravity"],
                    step=i,
                )
                self._compare_vectors(
                    "Joint Positions",
                    isaac_joint_pos,
                    pipeline_inputs["joint_positions"],
                    step=i,
                )
                self._compare_vectors(
                    "Joint Velocities",
                    isaac_joint_vel,
                    pipeline_inputs["joint_velocities"],
                    step=i,
                )

                # -------------------------------------------------------------
                # 5. Compare Pipeline Action vs IsaacLab Ground Truth
                # -------------------------------------------------------------
                pipeline_action = self.generator()

                self._compare_vectors(
                    "Actions",
                    expected=action_dict["actions"][0],
                    actual=pipeline_action.joint_command.position[:12],
                    step=i,
                    tolerance=1e-2,
                )

            print("Validation complete! All tested steps match.")

    def _compare_vectors(
        self,
        name: str,
        expected: np.ndarray,
        actual: list,
        step: int,
        tolerance: float = 1e-4,
    ):
        actual_np = np.array(actual)
        expected_np = np.array(expected)

        diff = np.max(np.abs(expected_np - actual_np))
        if diff > tolerance:
            print(f"--- MISMATCH AT STEP {step} IN {name} ---")
            print(f"Max Diff: {diff}")
            print(f"Expected (IsaacLab) : {expected_np}")
            print(f"Actual   (Pipeline) : {actual_np}")
            raise AssertionError(
                f"Pipeline output for {name} diverged from IsaacLab ground truth."
            )


if __name__ == "__main__":
    # Replace with your actual HDF5 dataset filename
    validator = PipelineValidator("datasets/flat_box_dataset_2026-04-14_15-51-45.hdf5")
    validator.validate()
