# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import os
import time
from dataclasses import dataclass
from operator import add, mul
from threading import Event
from typing import List

import numpy as np
import onnxruntime as ort
from bosdyn.api import robot_command_pb2
from bosdyn.api.robot_command_pb2 import JointControlStreamRequest
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse
from bosdyn.util import seconds_to_timestamp, set_timestamp_from_now, timestamp_to_sec

import rl_deploy.orbit.observations as ob
from rl_deploy.orbit.orbit_configuration import OrbitConfig
from rl_deploy.orbit.orbit_constants import (
    ORDERED_JOINT_NAMES_ARM_ISAAC,
    ORDERED_JOINT_NAMES_BASE_ISAAC,
    ORDERED_JOINT_NAMES_ISAAC,
)
from rl_deploy.spot.constants import (
    DEFAULT_K_Q_P,
    DEFAULT_K_QD_P,
    JOINT_LIMITS,
    JOINT_SOFT_LIMITS,
    ORDERED_JOINT_NAMES_SPOT,
    ORDERED_JOINT_NAMES_SPOT_ARM,
    ORDERED_JOINT_NAMES_SPOT_BASE,
)
from rl_deploy.utils.dict_tools import dict_to_list, find_ordering, reorder
from rl_deploy.utils.hdf5_logger import HDF5Logger
import onnx
from onnx import numpy_helper


@dataclass
class OnnxControllerContext:
    """data class to hold runtime data needed by the controller"""

    event = Event()
    latest_state = None
    velocity_cmd = [0.0, 0.0, 0.0]
    count = 0

    def __post_init__(self):
        self.timing_dict = {}


class StateHandler:
    """Class to be used as callback for state stream to put state date
    into the controllers context
    """

    def __init__(self, context: OnnxControllerContext) -> None:
        self._context = context

    def __call__(self, state: RobotStateStreamResponse):
        """make class a callable and handle incoming state stream when called

        arguments
        state -- proto msg from spot containing most recent data on the robots state"""
        self._context.latest_state = state
        if hasattr(self._context, "timing_dict"):
            self._context.timing_dict["state_arrival"] = time.perf_counter()
        self._context.event.set()


def print_observations(observations: dict[str, list]):
    """debug function to print out the observation data used as model input

    arguments
    observations -- list of float values ready to be passed into the model
    """
    for key, value in observations.items():
        print(f"{key}: {value}\n")


JOINTS_ORDER_RELIC = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]

# Relic Plus samples this command uniformly in [0.5, 0.7] during training.
# A zero here is out-of-distribution and changes the policy's leg targets.
DEFAULT_HEIGHT_COMMAND = 0.6


def extract_shift_from_onnx(onnx_file_path):
    # 1. Load the ONNX model graph
    print(f"Loading {onnx_file_path}...")
    model = onnx.load(onnx_file_path)

    # 2. Iterate through the graph's initializers (weights, biases, buffers)
    for init in model.graph.initializer:
        # PyTorch usually names the buffer exactly "shift" or something similar like "model.shift"
        if "shift" in init.name.lower():
            # 3. Convert the ONNX tensor to a standard Numpy array
            shift_array = numpy_helper.to_array(init)

            print(f"\n[SUCCESS] Found buffer named: '{init.name}'")
            print(f"Shape: {shift_array.shape}")
            print(f"Values:\n{shift_array}")

            # Return just the 12 leg joints, or the whole thing
            return shift_array

    print("\n[FAILED] Could not find any initializer containing the name 'shift'.")
    return None


class OnnxCommandGenerator:
    """class to be used as generator for spots command stream that executes
    an onnx model and converts the output to a spot command"""

    def __init__(
        self,
        context: OnnxControllerContext,
        config: OrbitConfig,
        policy_file_name: os.PathLike | str,
        verbose: bool,
        logger: HDF5Logger | None = None,
        mock: bool = False,
        arm_motion=None,
    ):
        self._context = context
        self._config = config
        self.logger = logger
        self.mock = mock
        self._inference_session = ort.InferenceSession(policy_file_name)
        self._session_input_names = {
            item.name for item in self._inference_session.get_inputs()
        }
        self._session_output_names_in_order = [
            item.name for item in self._inference_session.get_outputs()
        ]
        self._session_output_names = set(self._session_output_names_in_order)
        expected_inputs = set(getattr(config, "policy_inputs", []))
        if expected_inputs and expected_inputs != self._session_input_names:
            raise ValueError(
                "ONNX inputs do not match policy.yaml: "
                f"model={sorted(self._session_input_names)}, "
                f"manifest={sorted(expected_inputs)}"
            )
        self._flat_obs_size = self._detect_flat_obs_size()
        self._hidden_state_shape = self._detect_hidden_state_shape()
        self._hidden_state = None
        self._gait_phase = 0.0
        self._count = 1
        self._init_pos = None
        self._init_load = None
        self.verbose = verbose

        self.joints_offsets_ordered_spot = dict_to_list(
            self._config.default_joints, ORDERED_JOINT_NAMES_SPOT
        )
        self.base_offsets_ordered_spot = self.joints_offsets_ordered_spot[:12]
        self.action_scale = (
            1.0 if self._config.action_scale is None else self._config.action_scale
        )

        self.arm_offsets_ordered = [0.0, -3.1415, 3.1415, 1.5655, 0.00, -1.5655, 0.0]
        # dict_to_list(
        #     [0.0, -3.1415, 3.1415, 1.5655, 0.00, 0.0, 0.0], ORDERED_JOINT_NAMES_ARM_ISAAC
        # )

        # None keeps the arm stowed. Otherwise the bundle's stow must agree with ours, or the
        # first commanded arm target would step away from where the arm actually is.
        self.arm_motion = arm_motion
        if arm_motion is not None and not np.allclose(
            arm_motion.stow, self.arm_offsets_ordered, atol=1e-3
        ):
            raise ValueError(
                f"policy.yaml arm stow {arm_motion.stow} does not match deployment stow "
                f"{self.arm_offsets_ordered}"
            )

        self._triggered_safety = False
        self._safety_pos = None
        self._last_contact_binary = None

        self._safe_limits = self._generate_safe_limits()
        self.reset_policy_state()

    def _detect_flat_obs_size(self):
        if "obs" not in self._session_input_names:
            return None
        if self._session_input_names != {"obs"}:
            raise ValueError("Flat ONNX policies must expose only the 'obs' input")
        obs_input = self._inference_session.get_inputs()[0]
        size = obs_input.shape[-1]
        if size not in (65, 84):
            raise ValueError(f"Unsupported flat ONNX observation width: {size}")
        return size

    def _detect_hidden_state_shape(self):
        if "hidden_state" not in self._session_input_names:
            return None
        required_outputs = {"actions_output", "next_hidden_state"}
        missing = required_outputs - self._session_output_names
        if missing:
            raise ValueError(
                f"Recurrent ONNX is missing required outputs: {sorted(missing)}"
            )
        hidden_input = next(
            item
            for item in self._inference_session.get_inputs()
            if item.name == "hidden_state"
        )
        shape = hidden_input.shape
        if (
            len(shape) != 3
            or not isinstance(shape[0], int)
            or not isinstance(shape[2], int)
            or (isinstance(shape[1], int) and shape[1] != 1)
        ):
            raise ValueError(
                "hidden_state must have shape [num_layers, batch, hidden_size] "
                f"with deployment batch 1; got {shape}"
            )
        return shape[0], 1, shape[2]

    def reset_policy_state(self):
        self._gait_phase = 0.0
        self._hidden_state = (
            None
            if self._hidden_state_shape is None
            else np.zeros(self._hidden_state_shape, dtype=np.float32)
        )
        if self._flat_obs_size is not None:
            self._last_action = [0.0] * 12
            return
        self._last_action = list(self.base_offsets_ordered_spot)

    def _foot_height_commands(self, config):
        foot_radius = float(config.foot_radius)
        if np.linalg.norm(self._context.velocity_cmd) < float(
            config.standing_velocity_threshold
        ):
            return np.full((1, 4), foot_radius, dtype=np.float32)
        swing_fraction = float(config.gait_swing_fraction)
        phase_offsets = np.asarray(config.gait_phase_offsets, dtype=np.float32)
        theta = (self._gait_phase - phase_offsets) % 1.0
        swing = theta < swing_fraction
        height = np.sin(np.pi * theta / swing_fraction)
        command = foot_radius + float(config.foot_height_max) * np.where(
            swing, height, 0.0
        )
        return command.astype(np.float32).reshape(1, 4)

    def _advance_gait_phase(self, config):
        self._gait_phase = (
            self._gait_phase
            + float(config.gait_frequency) * float(config.control_period_s)
        ) % 1.0

    def _timestamp_to_seconds(self, timestamp) -> float:
        return float(timestamp.seconds) + float(timestamp.nanos) * 1e-9

    def _extract_contact_telemetry(self, state: RobotStateStreamResponse):
        contact_enum = list(state.contact_states[:4])
        if len(contact_enum) < 4:
            contact_enum.extend([0] * (4 - len(contact_enum)))

        contact_binary = [1 if value == 1 else 0 for value in contact_enum]
        if self._last_contact_binary is None:
            contact_transition = [0, 0, 0, 0]
        else:
            contact_transition = [
                current - previous
                for current, previous in zip(contact_binary, self._last_contact_binary)
            ]

        self._last_contact_binary = contact_binary
        return contact_enum, contact_binary, contact_transition

    def _extract_latest_imu(self, state: RobotStateStreamResponse):
        if len(state.inertial_state.packets) == 0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], None

        packet = state.inertial_state.packets[-1]
        acceleration = packet.acceleration_rt_odom_in_link_frame
        angular_velocity = packet.angular_velocity_rt_odom_in_link_frame
        return (
            [acceleration.x, acceleration.y, acceleration.z],
            [angular_velocity.x, angular_velocity.y, angular_velocity.z],
            self._timestamp_to_seconds(packet.timestamp),
        )

    def _generate_safe_limits(self):
        """
        Generate safe limits for each joint based on the joint limits and soft limits.

        The soft limits were generated from simulated data, using the formula:

        max_val, min_val = max and min needed during simulation
        max, min = max and min of the joint limit range

        middle = (max + min)/2
        full_range = max - min

        min_margin = (middle - min_val)/full_range * 2
        max_margin = (max_val - middle)/full_range * 2

        """
        safe_limits = {}
        for joint_name in JOINT_SOFT_LIMITS:
            lower = JOINT_LIMITS[joint_name]["lower"]
            upper = JOINT_LIMITS[joint_name]["upper"]
            middle = (lower + upper) / 2
            full_range = upper - lower

            min_margin, max_margin = JOINT_SOFT_LIMITS[joint_name]
            min_val = middle - (min_margin * full_range / 2)
            max_val = middle + (max_margin * full_range / 2)

            safe_limits[joint_name] = (min_val, max_val)

        msg = "\nSafety Limits:\n"
        msg += "\n".join(
            [
                f"  {joint_name}: [{min_val:.3f}, {max_val:.3f}]\n"
                for joint_name, (min_val, max_val) in safe_limits.items()
            ]
        )
        print(msg)

        return safe_limits

    def __call__(self):
        """makes class a callable and computes model output for latest controller context

        return proto message to be used in spots command stream
        """
        t_start_call = time.perf_counter()
        if hasattr(self._context, "timing_dict"):
            last_call = self._context.timing_dict.get("last_call_time", t_start_call)
            dt_total_step = t_start_call - last_call
            self._context.timing_dict["last_call_time"] = t_start_call
            dt_divider_to_onnx = t_start_call - self._context.timing_dict.get(
                "divider_end", t_start_call
            )
            dt_state_arrival_to_compute = t_start_call - self._context.timing_dict.get(
                "state_arrival", t_start_call
            )
        else:
            dt_total_step, dt_divider_to_onnx, dt_state_arrival_to_compute = (
                0.0,
                0.0,
                0.0,
            )

        # cache initial joint position when command stream starts
        if self._init_pos is None:
            self._init_pos = self._context.latest_state.joint_states.position
            self._init_load = self._context.latest_state.joint_states.load

        if self._safety_pos is not None:
            return self.create_proto(self._safety_pos)

        # extract observation data from latest spot state data
        inputs_dict = self.collect_inputs(self._context.latest_state, self._config)

        current_positions_map = dict(
            zip(
                ORDERED_JOINT_NAMES_SPOT,
                self._context.latest_state.joint_states.position,
            )
        )

        # Safety Check
        self._triggered_safety = False  # self._check_safety(current_positions_map)

        if self._triggered_safety:
            print("Triggered safety")
            # Create hold command from current positions
            hold_pos = [
                current_positions_map[name] for name in ORDERED_JOINT_NAMES_SPOT
            ]
            self._safety_pos = hold_pos
            return self.create_proto(hold_pos)

        if self.mock:
            # Action of zeros results in default joint values after post-processing
            mocked_action = (
                [0.0] * 12
                if self._flat_obs_size is not None
                else list(self.base_offsets_ordered_spot)
            )
            output = mocked_action
            t_onx_start = t_onx_end = time.perf_counter()
        else:
            t_onx_start = time.perf_counter()
            output = self._compute_action(inputs_dict)
            t_onx_end = time.perf_counter()

        t_post_start = time.perf_counter()
        action = self._deployment_action(output)
        t_post_end = time.perf_counter()

        # generate proto message from target joint positions
        proto = self.create_proto(action)

        if self.logger is not None:
            dt_onnx = t_onx_end - t_onx_start
            dt_post = t_post_end - t_post_start
            dt_divider_wait = (
                self._context.timing_dict.get("dt_divider_wait", 0.0)
                if hasattr(self._context, "timing_dict")
                else 0.0
            )

            raw_state = self._context.latest_state
            (
                foot_contact_enum,
                foot_contact_binary,
                foot_contact_transition,
            ) = self._extract_contact_telemetry(raw_state)
            (
                imu_linear_acceleration,
                imu_angular_velocity,
                imu_packet_timestamp,
            ) = self._extract_latest_imu(raw_state)
            self.logger.log_state(
                raw_base_linear_velocity=ob.get_base_linear_velocity(raw_state),
                raw_base_angular_velocity=ob.get_base_angular_velocity(raw_state),
                raw_projected_gravity=ob.get_projected_gravity(raw_state),
                raw_joint_positions=ob.get_joint_positions(
                    raw_state, self.joints_offsets_ordered_spot
                ),
                raw_joint_velocities=ob.get_joint_velocity(raw_state),
                raw_joint_loads=ob.get_join_load(raw_state),
                response_timestamp=ob.get_response_timestamp(raw_state),
                spot_current_positions=list(raw_state.joint_states.position),
                spot_current_velocities=list(raw_state.joint_states.velocity),
                preprocessed_base_linear_velocity=self._observation_terms["base_linear_velocity"],
                preprocessed_base_angular_velocity=self._observation_terms["base_angular_velocity"],
                preprocessed_projected_gravity=self._observation_terms["projected_gravity"],
                preprocessed_velocity_cmd=self._observation_terms["velocity_commands"],
                preprocessed_joint_positions=self._observation_terms["joint_positions"],
                preprocessed_joint_velocities=self._observation_terms["joint_velocities"],
                preprocessed_last_action=self._observation_terms["last_actions"],
                commanded_action=action,
                dt_divider_wait=dt_divider_wait,
                dt_divider_to_onnx=dt_divider_to_onnx,
                dt_onnx_compute=dt_onnx,
                dt_post_process=dt_post,
                dt_total_step=dt_total_step,
                dt_state_arrival_to_compute=dt_state_arrival_to_compute,
                raw_state_proto_bytes=raw_state.SerializeToString(),
                proto_bytes=proto.SerializeToString(),
                foot_contact_enum=foot_contact_enum,
                foot_contact_binary=foot_contact_binary,
                foot_contact_transition=foot_contact_transition,
                imu_linear_acceleration=imu_linear_acceleration,
                imu_angular_velocity=imu_angular_velocity,
                imu_packet_timestamp=imu_packet_timestamp,
                command_user_key=proto.joint_command.user_command_key,
                command_request_timestamp=self._timestamp_to_seconds(
                    proto.header.request_timestamp
                ),
                command_end_time=self._timestamp_to_seconds(
                    proto.joint_command.end_time
                ),
            )

        # cache data for history and logging
        self._last_action = output
        if "foot_height_commands" in self._session_input_names:
            self._advance_gait_phase(self._config)
        self._count += 1
        self._context.count += 1

        if self.mock:
            return proto

        return proto

    def _check_safety(self, current_positions_map):
        for joint_name, (safe_min, safe_max) in self._safe_limits.items():
            current_val = current_positions_map.get(joint_name)

            if current_val is None:
                print(f"[SAFETY STOP] Joint {joint_name} value is None")
                return True

            if current_val < safe_min or current_val > safe_max:
                print(
                    f"[SAFETY STOP] Joint {joint_name} value {current_val:.4f} outside safe range [{safe_min:.4f}, {safe_max:.4f}]"
                )
                return True
        return False

    def _compute_action(self, input_dict: dict[str, float]):
        # execute model from onnx file
        if self._hidden_state_shape is None:
            return self._inference_session.run(None, input_dict)[0].tolist()[0]
        recurrent_inputs = dict(input_dict)
        recurrent_inputs["hidden_state"] = self._hidden_state
        values = self._inference_session.run(None, recurrent_inputs)
        outputs = dict(zip(self._session_output_names_in_order, values))
        self._hidden_state = outputs["next_hidden_state"]
        return outputs["actions_output"].tolist()[0]

    def _deployment_action(self, output):
        output = np.asarray(output, dtype=np.float32)
        if output.shape != (12,):
            raise ValueError(f"Policy output must have shape (12,), got {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("Policy output contains non-finite values")
        if self._flat_obs_size is None:
            legs = output
        else:
            legs = np.asarray(self.base_offsets_ordered_spot) + self.action_scale * output
        legs = [
            float(np.clip(value, JOINT_LIMITS[name]["lower"], JOINT_LIMITS[name]["upper"]))
            for name, value in zip(ORDERED_JOINT_NAMES_SPOT_BASE, legs)
        ]
        if self.arm_motion is None:
            return legs + self.arm_offsets_ordered
        arm = [
            float(np.clip(value, JOINT_LIMITS[name]["lower"], JOINT_LIMITS[name]["upper"]))
            for name, value in zip(ORDERED_JOINT_NAMES_SPOT_ARM, self.arm_motion.step())
        ]
        return legs + arm

    def collect_inputs(
        self,
        state: JointControlStreamRequest,
        config: OrbitConfig,
        joint_commands: List[float] | None = None,
    ) -> dict:
        """extract observation data from spots current state and format for onnx

        arguments
        state -- proto msg with spots latest state
        config -- model configuration data from orbit

        return dict of isolated preprocessed observations
        """
        if self.verbose:
            print("[INFO] cmd", self._context.velocity_cmd)

        absolute_joint_positions = np.array(state.joint_states.position, dtype=np.float32).reshape(1, -1)
        joint_velocities = np.array(state.joint_states.velocity, dtype=np.float32).reshape(1, -1)
        inputs = {
            "base_linear_velocity": ob.get_base_linear_velocity(state)
            .astype(np.float32)
            .reshape(1, 3),
            "base_angular_velocity": ob.get_base_angular_velocity(state)
            .astype(np.float32)
            .reshape(1, 3),
            "projected_gravity": ob.get_projected_gravity(state)
            .astype(np.float32)
            .reshape(1, 3),
            "velocity_commands": np.array(self._context.velocity_cmd)
            .astype(np.float32)
            .reshape(1, 3),
            "joint_positions": absolute_joint_positions,
            "joint_velocities": joint_velocities,
            "last_actions": np.array(self._last_action)
            .astype(np.float32)
            .reshape(1, -1),
        }

        # The deployment config may opt into a different fixed policy command;
        # old OrbitConfig instances remain compatible through the fallback.
        height_command = float(
            getattr(config, "height_command", DEFAULT_HEIGHT_COMMAND)
        )

        if self._flat_obs_size is not None:
            inputs["joint_positions"] = absolute_joint_positions - np.array(
                self.joints_offsets_ordered_spot, dtype=np.float32
            )
            parts = [
                inputs["base_linear_velocity"],
                inputs["base_angular_velocity"],
                inputs["projected_gravity"],
                inputs["velocity_commands"],
            ]
            if self._flat_obs_size == 84:
                commands = (
                    ob.generate_joint_commands(state)
                    if joint_commands is None
                    else joint_commands
                )
                parts.append(np.array(commands, dtype=np.float32).reshape(1, 22))
            parts.extend(
                [inputs["joint_positions"], inputs["joint_velocities"], inputs["last_actions"]]
            )
            if self._flat_obs_size == 65:
                parts.extend(
                    [
                        np.full((1, 1), height_command, dtype=np.float32),
                        np.zeros((1, 2), dtype=np.float32),
                    ]
                )
            self._observation_terms = inputs
            return {"obs": np.concatenate(parts, axis=1)}

        # Check if the loaded ONNX model requires height_commands or base_orientation_commands
        if "height_commands" in self._session_input_names:
            inputs["height_commands"] = np.array([[height_command]], dtype=np.float32)
        if "base_orientation_commands" in self._session_input_names:
            inputs["base_orientation_commands"] = np.array(
                [config.orientation_command], dtype=np.float32
            )
        if "foot_height_commands" in self._session_input_names:
            inputs["foot_height_commands"] = self._foot_height_commands(config)

        self._observation_terms = inputs
        return {
            name: inputs[name]
            for name in self._session_input_names
            if name != "hidden_state"
        }

    def create_proto(self, pos_command: List[float]):
        """generate a proto msg for spot with a given pos_command

        arguments
        pos_command -- list of joint positions see spot.constants for order

        return proto message to send in spots command stream
        """

        update_proto = robot_command_pb2.JointControlStreamRequest()
        update_proto.Clear()

        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = dict_to_list(self._config.kp, ORDERED_JOINT_NAMES_SPOT)
        k_qd_p = dict_to_list(self._config.kd, ORDERED_JOINT_NAMES_SPOT)

        N_DOF = len(pos_command)
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = pos_command[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = 0

        # Fill in gains the first dt
        if self._count <= 3:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(pos_cmd)
        update_proto.joint_command.velocity.extend(vel_cmd)
        update_proto.joint_command.load.extend(load_cmd)

        observation_time = self._context.latest_state.joint_states.acquisition_timestamp
        end_time = seconds_to_timestamp(timestamp_to_sec(observation_time) + 0.1)
        update_proto.joint_command.end_time.CopyFrom(end_time)

        # Let it extrapolate the command a little
        update_proto.joint_command.extrapolation_duration.nanos = int(5 * 1e6)

        # Set user key for latency tracking
        update_proto.joint_command.user_command_key = self._count
        return update_proto

    def create_proto_hold(self):
        """generate a proto msg that holds spots current pose useful for debugging

        return proto message to send in spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        update_proto.Clear()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = DEFAULT_K_Q_P[0:19]
        k_qd_p = DEFAULT_K_QD_P[0:19]

        N_DOF = 19
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = self._init_pos[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = self._init_load[joint_ind]

        # Fill in gains the first dt
        if self._count == 1:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(pos_cmd)
        update_proto.joint_command.velocity.extend(vel_cmd)
        update_proto.joint_command.load.extend(load_cmd)

        observation_time = self._context.latest_state.joint_states.acquisition_timestamp
        end_time = seconds_to_timestamp(timestamp_to_sec(observation_time) + 0.1)
        update_proto.joint_command.end_time.CopyFrom(end_time)

        # Let it extrapolate the command a little
        update_proto.joint_command.extrapolation_duration.nanos = int(5 * 1e6)

        # Set user key for latency tracking
        update_proto.joint_command.user_command_key = self._count
        return update_proto
