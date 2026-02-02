# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from contextlib import nullcontext
from typing import Callable

import torch
from bosdyn.api.robot_command_pb2 import JointControlStreamRequest
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse

from rl_deploy.orbit.orbit_constants import (
    ORDERED_JOINT_NAMES_BASE_ISAAC,
)
from rl_deploy.spot.constants import (
    ORDERED_JOINT_NAMES_SPOT_BASE,
)
from rl_deploy.utils.dict_tools import find_ordering, reorder


class IsaacMockSpot:
    _command_thread = None
    _state_stream_stopping = False
    _command_stream_stopping = False

    def __init__(self): ...

    def start_state_stream(
        self, on_state_update: Callable[[RobotStateStreamResponse], None]
    ):
        self._on_state_update = on_state_update

    def set_state(self, state: dict[str, torch.Tensor]):
        """
        Calculate linear velocity of spots base in the base frame from data
        available in spots state update.  note spot gives velocity in odom frame
        so we need to rotate it to current estimated pose of the base
        """
        joint_pos = state["joint_pos"].cpu().tolist()[0]
        joint_vel = state["joint_vel"].cpu().tolist()[0]
        root_lin_vel_w = state["root_lin_vel_w"].cpu().tolist()[0]
        root_ang_vel_w = state["root_ang_vel_w"].cpu().tolist()[0]
        root_quat_w = state["root_quat_w"].cpu().tolist()[0]

        joint_load = [0] * 19

        self._state_msg = RobotStateStreamResponse()
        self._state_msg.kinematic_state.odom_tform_body.rotation.w = root_quat_w[0]
        self._state_msg.kinematic_state.odom_tform_body.rotation.x = root_quat_w[1]
        self._state_msg.kinematic_state.odom_tform_body.rotation.y = root_quat_w[2]
        self._state_msg.kinematic_state.odom_tform_body.rotation.z = root_quat_w[3]
        self._state_msg.kinematic_state.velocity_of_body_in_odom.linear.x = (
            root_lin_vel_w[0]
        )
        self._state_msg.kinematic_state.velocity_of_body_in_odom.linear.y = (
            root_lin_vel_w[1]
        )
        self._state_msg.kinematic_state.velocity_of_body_in_odom.linear.z = (
            root_lin_vel_w[2]
        )
        self._state_msg.kinematic_state.velocity_of_body_in_odom.angular.x = (
            root_ang_vel_w[0]
        )
        self._state_msg.kinematic_state.velocity_of_body_in_odom.angular.y = (
            root_ang_vel_w[1]
        )
        self._state_msg.kinematic_state.velocity_of_body_in_odom.angular.z = (
            root_ang_vel_w[2]
        )
        self._state_msg.joint_states.position.extend(joint_pos)
        self._state_msg.joint_states.velocity.extend(joint_vel)
        self._state_msg.joint_states.load.extend(joint_load)

        self._on_state_update(self._state_msg)

        self.last_command = None

    def start_command_stream(
        self,
        command_policy: Callable[[None], JointControlStreamRequest],
    ):
        self._command_generator = command_policy

    def lease_keep_alive(self):
        return nullcontext()

    def command_update(self):
        positions = self._command_generator().joint_command.position

        spot_to_isaac = find_ordering(
            ORDERED_JOINT_NAMES_SPOT_BASE, ORDERED_JOINT_NAMES_BASE_ISAAC
        )
        reordered_output = reorder(positions, spot_to_isaac)
        return torch.tensor(reordered_output).unsqueeze(0)

    def power_on(self):
        pass

    def stand(self, body_height: float):
        pass

    def stop_state_stream(self): ...

    def stop_command_stream(self):
        if self._command_thread is not None:
            self._command_stream_stopping = True
            self._command_thread.join()
