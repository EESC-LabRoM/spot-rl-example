# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Custom actuator definitions for Spot"""

from collections.abc import Sequence

import torch

# IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd_cfg.py
from isaaclab.actuators.actuator_cfg import RemotizedPDActuatorCfg
from isaaclab.actuators.actuator_pd import RemotizedPDActuator
from isaaclab.utils import LinearInterpolation, configclass
from isaaclab.utils.types import ArticulationActions
from torch._tensor import Tensor

POS_TORQUE_SPEED_LIMIT: list[list[float]] = [
    [-30.0, 96.9972],
    [0.0, 96.9972],
    [14.0, 0.0000],
]

NEG_TORQUE_SPEED_LIMIT: list[list[float]] = [
    [-15.0, 0.0000],
    [0.0, -96.9972],
    [30.0000, -96.9972],
]


class SpotKneeActuator(RemotizedPDActuator):
    """Spot knee actuator."""

    def __init__(
        self,
        cfg: RemotizedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: Tensor | float = 0,
        damping: Tensor | float = 0,
        armature: Tensor | float = 0,
        friction: Tensor | float = 0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,
        effort_limit: Tensor | float = torch.inf,
        velocity_limit: Tensor | float = torch.inf,
    ):

        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            stiffness,
            damping,
            armature,
            friction=friction,
            dynamic_friction=dynamic_friction,
            viscous_friction=viscous_friction,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
        )

        self._pos_torque_speed_data = torch.tensor(
            cfg.pos_torque_speed_limit, device=device
        )
        self._neg_torque_speed_data = torch.tensor(
            cfg.neg_torque_speed_limit, device=device
        )
        self._enable_torque_speed_limit = cfg.enable_torque_speed_limit

        # define remotized joint torque limit
        self._pos_torque_speed_limit = LinearInterpolation(
            self._pos_torque_speed_data[:, 0],
            self._pos_torque_speed_data[:, 1],
            device=device,
        )
        self._neg_torque_speed_limit = LinearInterpolation(
            self._neg_torque_speed_data[:, 0],
            self._neg_torque_speed_data[:, 1],
            device=device,
        )

    def compute_pd_ideal(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        # calculate the desired joint torques
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
        )
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def compute_delayed(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(
            control_action.joint_positions
        )
        control_action.joint_velocities = self.velocities_delay_buffer.compute(
            control_action.joint_velocities
        )
        control_action.joint_efforts = self.efforts_delay_buffer.compute(
            control_action.joint_efforts
        )
        # compte actuator model
        return self.compute_pd_ideal(control_action, joint_pos, joint_vel)

    def compute_remotized_pd(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # call the base method
        control_action = self.compute_delayed(control_action, joint_pos, joint_vel)
        # compute the absolute torque limits for the current joint positions
        abs_torque_limits = self._torque_limit.compute(joint_pos)
        # apply the limits
        control_action.joint_efforts = torch.clamp(
            control_action.joint_efforts, min=-abs_torque_limits, max=abs_torque_limits
        )
        self.applied_effort = control_action.joint_efforts
        return control_action

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute the control action for the Spot robot with positional torque speed limits."""
        control_action = self.compute_remotized_pd(control_action, joint_pos, joint_vel)

        # compute torque-speed limits
        if self._enable_torque_speed_limit:
            pos_torque_limits = self._pos_torque_speed_limit.compute(joint_vel)
            neg_torque_limits = self._neg_torque_speed_limit.compute(joint_vel)
            control_action.joint_efforts = torch.clamp(
                control_action.joint_efforts,
                min=neg_torque_limits,
                max=pos_torque_limits,
            )
        self.applied_effort = control_action.joint_efforts
        return control_action


@configclass
class SpotKneeActuatorCfg(RemotizedPDActuatorCfg):
    """Configuration for the Spot knee actuator."""

    class_type: type = SpotKneeActuator

    enable_torque_speed_limit: bool = False

    pos_torque_speed_limit = POS_TORQUE_SPEED_LIMIT
    neg_torque_speed_limit = NEG_TORQUE_SPEED_LIMIT
