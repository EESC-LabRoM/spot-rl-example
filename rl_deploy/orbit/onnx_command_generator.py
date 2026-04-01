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
    ORDERED_JOINT_NAMES_SPOT_BASE
)
from rl_deploy.utils.dict_tools import dict_to_list, find_ordering, reorder
from rl_deploy.utils.hdf5_logger import HDF5Logger


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


def print_observations(observations: List[float]):
    """debug function to print out the observation data used as model input

    arguments
    observations -- list of float values ready to be passed into the model
    """
    print("base_linear_velocity:", observations[0:3])
    print("base_angular_velocity:", observations[3:6])
    print("projected_gravity:", observations[6:9])
    print("joint_positions", observations[12:24])
    print("joint_velocity", observations[24:36])
    print("last_action", observations[36:48])


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
    ):
        self._context = context
        self._config = config
        self.logger = logger
        self.mock = mock
        self._inference_session = ort.InferenceSession(policy_file_name)
        self._last_action = [0] * 12
        self._count = 1
        self._init_pos = None
        self._init_load = None
        self.verbose = verbose

        self.joints_offsets_ordered_base = dict_to_list(
            self._config.default_joints, ORDERED_JOINT_NAMES_SPOT_BASE
        )
        self.joints_offsets_ordered_spot = dict_to_list(
            self._config.default_joints, ORDERED_JOINT_NAMES_SPOT
        )

        self.arm_offsets_ordered = [0.0, -3.1415, 3.1415, 1.5655, 0.00, 0.0, 0.0]
        # dict_to_list(
        #     [0.0, -3.1415, 3.1415, 1.5655, 0.00, 0.0, 0.0], ORDERED_JOINT_NAMES_ARM_ISAAC
        # )

        self._triggered_safety = False
        self._safety_pos = None

        self._safe_limits = self._generate_safe_limits()

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

        # Flatten for the onnx model
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
            mocked_action = [0.0] * 12
            output = mocked_action
            t_onx_start = t_onx_end = time.perf_counter()
        else:
            t_onx_start = time.perf_counter()
            output = self._compute_action(input_list)
            t_onx_end = time.perf_counter()

        t_post_start = time.perf_counter()
        action = self._post_process_action_to_spot(output)
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
                preprocessed_base_linear_velocity=inputs_dict["base_linear_velocity"],
                preprocessed_base_angular_velocity=inputs_dict["base_angular_velocity"],
                preprocessed_projected_gravity=inputs_dict["projected_gravity"],
                preprocessed_velocity_cmd=inputs_dict["velocity_cmd"],
                preprocessed_joint_positions=inputs_dict["joint_positions"],
                preprocessed_joint_velocities=inputs_dict["joint_velocities"],
                preprocessed_last_action=inputs_dict["last_action"],
                commanded_action=action,
                dt_divider_wait=dt_divider_wait,
                dt_divider_to_onnx=dt_divider_to_onnx,
                dt_onnx_compute=dt_onnx,
                dt_post_process=dt_post,
                dt_total_step=dt_total_step,
                dt_state_arrival_to_compute=dt_state_arrival_to_compute,
                raw_state_proto_bytes=raw_state.SerializeToString(),
                proto_bytes=proto.SerializeToString(),
            )

        # cache data for history and logging
        self._last_action = output
        self._count += 1
        self._context.count += 1

        if self.mock:
            mocked_action = [0.0] * 12
            return self.create_proto(mocked_action)

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

    def _compute_action(self, input_list: List[float]):
        # execute model from onnx file
        input = [np.array(input_list).astype("float32")]
        output = self._inference_session.run(None, {"obs": input})[0].tolist()[0]
        return output

    def _post_process_action_to_spot(self, output: List[float]):
        # post process model output apply action scaling and return to spots
        # TODO: test with rampup scaling commands
        scaled_output = list(map(mul, [self._config.action_scale] * 12, output))

        # reorder for spot
        isaac_to_spot = find_ordering(
            JOINTS_ORDER_RELIC,
            ORDERED_JOINT_NAMES_SPOT_BASE,
        )
        reordered_output = reorder(scaled_output, isaac_to_spot)

        # set joint offsets for base
        shifted_output_base = list(
            map(add, reordered_output, self.joints_offsets_ordered_base)
        )
        shifted_output = shifted_output_base + self.arm_offsets_ordered

        return shifted_output

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

        return {
            "base_linear_velocity": ob.get_base_linear_velocity(state),
            "base_angular_velocity": ob.get_base_angular_velocity(state),
            "projected_gravity": ob.get_projected_gravity(state),
            "velocity_cmd": self._context.velocity_cmd,
            "joint_commands": joint_commands
            if joint_commands is not None
            else ob.generate_joint_commands(state),
            # TODO
            "joint_positions": ob.get_joint_positions(
                state, self.joints_offsets_ordered_spot
            ),
            "joint_velocities": ob.get_joint_velocity(state),
            "last_action": self._last_action,
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
