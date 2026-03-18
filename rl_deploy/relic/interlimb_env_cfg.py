# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

import math

# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from dataclasses import MISSING
from datetime import datetime
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from collections.abc import Sequence

import torch
from isaaclab.actuators.actuator_cfg import RemotizedPDActuatorCfg
from isaaclab.actuators.actuator_pd import RemotizedPDActuator
from isaaclab.utils import LinearInterpolation
from isaaclab.utils.types import ArticulationActions
from torch._tensor import Tensor

import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as isaac_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import JointAction, JointActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg

from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT
##
# Joint actions.
##


class MixedPDArmMultiLegJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: "MixedPDArmMultiLegJointPositionActionCfg"
    """The configuration of the action term."""

    def __init__(
        self,
        cfg: "MixedPDArmMultiLegJointPositionActionCfg",
        env: ManagerBasedEnv,
    ):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        print("JointAction preserve order:", self.cfg.preserve_order)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()
            print("OFFSET: ", self._offset)
        print("Joint IDS: ", self._joint_names)
        # setup the arm command buffer
        self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
            self.cfg.arm_joint_names, #preserve_order=True
        )
        self._leg_joint_ids = {
            leg: self._asset.find_joints(names)[0]#, preserve_order=True)[0]
            for leg, names in self.cfg.leg_joint_names.items()
        }
        print("LEG JOINT IDS: ", self._leg_joint_ids)
        self.command_name = cfg.command_name
        self.command_manager = env.command_manager
        self.command = self.command_manager.get_term(self.command_name)

        self._arm_raw_actions = torch.zeros(
            self.num_envs, len(self._arm_joint_ids), device=self.device
        )
        self._arm_processed_actions = torch.zeros_like(self.arm_raw_actions)

        self._leg_raw_actions = torch.zeros(
            self.num_envs, len(self._leg_joint_ids["fl"]), device=self.device
        )
        self._leg_processed_actions = torch.zeros_like(self._leg_raw_actions)

        self.batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)
        self.action_joint_idxs = torch.tensor(self._joint_ids, device=self.device)

    def apply_actions(self):
        """Apply the actions."""
        # set position targets
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
        self._asset.set_joint_position_target(
            self.arm_processed_actions, joint_ids=self._arm_joint_ids
        )

    def process_actions(self, actions: torch.Tensor):
        """Process the actions."""
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        #print("Action Scale: ", self._scale)
        #print("Action Offset: ", self._offset)
        # store the non-command leg actions
        no_command_leg_processed_actions = self._processed_actions.clone()[
            ~self.command.command_leg
        ]

        # store the raw arm actions, which is the target joint pos
        self._arm_raw_actions[:] = self.command.arm_joint_sub_goal
        self._arm_processed_actions[:] = self._arm_raw_actions.clone()

        # store the raw leg actions
        command_joint_idxs = self.command.command_leg_joint_idxs[
            self.command.command_leg_idxs
        ]
        self._leg_raw_actions[:] = self.command.leg_joint_sub_goal[
            self.batch_indices, command_joint_idxs
        ]
        self._leg_processed_actions[:] = self._leg_raw_actions.clone()

        # overwrite command leg actions
        # --- order of command: [
        # 'fl_hx', 'fl_hy', 'fl_kn', 'fr_hx', 'fr_hy', 'fr_kn',
        # 'hl_hx', 'hl_hy', 'hl_kn', 'hr_hx', 'hr_hy', 'hr_kn'
        # ]
        # --- order of control: [
        # 'fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy',
        # 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn'
        # ]
        # leg_joint_idxs = self.command.leg_joint_idxs[
        #     self.command.command_leg_idxs
        # ]  # commanded leg joint idx in the simulation
        # action_joint_idxs = (
        #     (
        #         leg_joint_idxs.view(-1).unsqueeze(1)
        #         == self.action_joint_idxs.unsqueeze(0)
        #     )
        #     .nonzero(as_tuple=True)[1]
        #     .view(leg_joint_idxs.shape)
        # )


        # self._processed_actions[self.batch_indices, action_joint_idxs] = (
        #     self._leg_processed_actions[:].clone()
        # )
        # # restore the non-command leg actions
        # self._processed_actions[~self.command.command_leg] = (
        #     no_command_leg_processed_actions.clone()
        # )

    @property
    def arm_raw_actions(self) -> torch.Tensor:
        """Get the raw arm actions."""
        return self._arm_raw_actions

    @property
    def arm_processed_actions(self) -> torch.Tensor:
        """Get the processed arm actions."""
        return self._arm_processed_actions

    @property
    def leg_raw_actions(self) -> torch.Tensor:
        """Get the raw leg actions."""
        return self._leg_raw_actions

    @property
    def leg_processed_actions(self) -> torch.Tensor:
        """Get the processed leg actions."""
        return self._leg_processed_actions


@configclass
class MixedPDArmMultiLegJointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = MixedPDArmMultiLegJointPositionAction

    arm_joint_names: tuple[str, ...] = MISSING
    leg_joint_names: dict = MISSING

    command_name: str = MISSING

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Constant data for Boston Dynamics Spot robot."""

SPOT_DEFAULT_POS: tuple[float, float, float] = (0.0, 0.0, 0.65)

ARM_JOINT_NAMES = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]

LEG_JOINT_NAMES_ALL = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]


FL_JOINT_NAMES = ["fl_hx", "fl_hy", "fl_kn"]
FR_JOINT_NAMES = ["fr_hx", "fr_hy", "fr_kn"]
HL_JOINT_NAMES = ["hl_hx", "hl_hy", "hl_kn"]
HR_JOINT_NAMES = ["hr_hx", "hr_hy", "hr_kn"]
LEG_JOINT_NAMES = {
    "fl": FL_JOINT_NAMES,
    "fr": FR_JOINT_NAMES,
    "hl": HL_JOINT_NAMES,
    "hr": HR_JOINT_NAMES,
}
FEET_NAMES = [f"{leg_name}_foot" for leg_name in LEG_JOINT_NAMES.keys()]

SPOT_DEFAULT_JOINT_POS: dict[str, float] = {
    "arm_sh0": 0.0,
    "arm_sh1": -0.9,
    "arm_el0": 1.8,
    "arm_el1": 0.0,
    "arm_wr0": -0.9,
    "arm_wr1": 0.0,
    "arm_f1x": -1.54,
    "fl_hx": 0.12,
    "fr_hx": -0.12,
    "hl_hx": 0.12,
    "hr_hx": -0.12,
    "fl_hy": 0.5,
    "fr_hy": 0.5,
    "hl_hy": 0.5,
    "hr_hy": 0.5,
    "fl_kn": -1.0,
    "fr_kn": -1.0,
    "hl_kn": -1.0,
    "hr_kn": -1.0,
}

ARM_STOW_JOINT_POS: dict[str, float] = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.1415,
    "arm_el0": 3.1415,
    "arm_el1": 1.5655,
    "arm_wr0": 0.00,
    "arm_wr1": 0.0,
    "arm_f1x": 0,
}

HIP_EFFORT_LIMIT: float = 45.0
HIP_STIFFNESS: float = 60.0
HIP_DAMPING: float = 1.5
HIP_FRICTION: float = 0.008

KNEE_STIFFNESS: float = 60.0
KNEE_DAMPING: float = 1.5
KNEE_FRICTION: float = 0.180

ARM_EFFORT_LIMIT: tuple[float, ...] = (90.9, 181.8, 90.9, 30.3, 30.3, 30.3, 15.32)

ARM_STIFFNESS: tuple[float, ...] = (120.0, 120.0, 120.0, 100.0, 100.0, 100.0, 16.0)

ARM_DAMPING: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.32)

ARM_ARMATURE: tuple[float, ...] = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001)

"""The lookup table for the knee joint parameters of the Boston Dynamics Spot robot.

This table describes the relationship between the joint angle (rad), the transmission ratio (in/out),
and the output torque (N*m). It is used to interpolate the output torque based on the joint angle.
"""
JOINT_PARAMETER_LOOKUP_TABLE: list[list[float]] = [
    [-2.792900, -24.776718, 37.165077],
    [-2.767442, -26.290108, 39.435162],
    [-2.741984, -27.793369, 41.690054],
    [-2.716526, -29.285997, 43.928996],
    [-2.691068, -30.767536, 46.151304],
    [-2.665610, -32.237423, 48.356134],
    [-2.640152, -33.695168, 50.542751],
    [-2.614694, -35.140221, 52.710331],
    [-2.589236, -36.572052, 54.858078],
    [-2.563778, -37.990086, 56.985128],
    [-2.538320, -39.393730, 59.090595],
    [-2.512862, -40.782406, 61.173609],
    [-2.487404, -42.155487, 63.233231],
    [-2.461946, -43.512371, 65.268557],
    [-2.436488, -44.852371, 67.278557],
    [-2.411030, -46.174873, 69.262310],
    [-2.385572, -47.479156, 71.218735],
    [-2.360114, -48.764549, 73.146824],
    [-2.334656, -50.030334, 75.045502],
    [-2.309198, -51.275761, 76.913641],
    [-2.283740, -52.500103, 78.750154],
    [-2.258282, -53.702587, 80.553881],
    [-2.232824, -54.882442, 82.323664],
    [-2.207366, -56.038860, 84.058290],
    [-2.181908, -57.171028, 85.756542],
    [-2.156450, -58.278133, 87.417200],
    [-2.130992, -59.359314, 89.038971],
    [-2.105534, -60.413738, 90.620607],
    [-2.080076, -61.440529, 92.160793],
    [-2.054618, -62.438812, 93.658218],
    [-2.029160, -63.407692, 95.111538],
    [-2.003702, -64.346268, 96.519402],
    [-1.978244, -65.253670, 97.880505],
    [-1.952786, -66.128944, 99.193417],
    [-1.927328, -66.971176, 100.456764],
    [-1.901870, -67.779457, 101.669186],
    [-1.876412, -68.552864, 102.829296],
    [-1.850954, -69.290451, 103.935677],
    [-1.825496, -69.991325, 104.986988],
    [-1.800038, -70.654541, 105.981812],
    [-1.774580, -71.279190, 106.918785],
    [-1.749122, -71.864319, 107.796478],
    [-1.723664, -72.409088, 108.613632],
    [-1.698206, -72.912567, 109.368851],
    [-1.672748, -73.373871, 110.060806],
    [-1.647290, -73.792130, 110.688194],
    [-1.621832, -74.166512, 111.249767],
    [-1.596374, -74.496147, 111.744221],
    [-1.570916, -74.780251, 112.170376],
    [-1.545458, -75.017998, 112.526997],
    [-1.520000, -75.208656, 112.812984],
    [-1.494542, -75.351448, 113.027172],
    [-1.469084, -75.445686, 113.168530],
    [-1.443626, -75.490677, 113.236015],
    [-1.418168, -75.485771, 113.228657],
    [-1.392710, -75.430344, 113.145515],
    [-1.367252, -75.323830, 112.985744],
    [-1.341794, -75.165688, 112.748531],
    [-1.316336, -74.955406, 112.433109],
    [-1.290878, -74.692551, 112.038826],
    [-1.265420, -74.376694, 111.565041],
    [-1.239962, -74.007477, 111.011215],
    [-1.214504, -73.584579, 110.376869],
    [-1.189046, -73.107742, 109.661613],
    [-1.163588, -72.576752, 108.865128],
    [-1.138130, -71.991455, 107.987183],
    [-1.112672, -71.351707, 107.027561],
    [-1.087214, -70.657486, 105.986229],
    [-1.061756, -69.908813, 104.863220],
    [-1.036298, -69.105721, 103.658581],
    [-1.010840, -68.248337, 102.372505],
    [-0.985382, -67.336861, 101.005291],
    [-0.959924, -66.371513, 99.557270],
    [-0.934466, -65.352615, 98.028923],
    [-0.909008, -64.280533, 96.420799],
    [-0.883550, -63.155693, 94.733540],
    [-0.858092, -61.978588, 92.967882],
    [-0.832634, -60.749775, 91.124662],
    [-0.807176, -59.469845, 89.204767],
    [-0.781718, -58.139503, 87.209255],
    [-0.756260, -56.759487, 85.139231],
    [-0.730802, -55.330616, 82.995924],
    [-0.705344, -53.853729, 80.780594],
    [-0.679886, -52.329796, 78.494694],
    [-0.654428, -50.759762, 76.139643],
    [-0.628970, -49.144699, 73.717049],
    [-0.603512, -47.485737, 71.228605],
    [-0.578054, -45.784004, 68.676006],
    [-0.552596, -44.040764, 66.061146],
    [-0.527138, -42.257267, 63.385900],
    [-0.501680, -40.434883, 60.652325],
    [-0.476222, -38.574947, 57.862421],
    [-0.450764, -36.678982, 55.018473],
    [-0.425306, -34.748432, 52.122648],
    [-0.399848, -32.784836, 49.177254],
    [-0.374390, -30.789810, 46.184715],
    [-0.348932, -28.764952, 43.147428],
    [-0.323474, -26.711969, 40.067954],
    [-0.298016, -24.632576, 36.948864],
    [-0.272558, -22.528547, 33.792821],
    [-0.247100, -20.401667, 30.602500],
]

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

# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Configuration for the Boston Dynamics robot.

The following configuration parameters are available:

* :obj:`SPOT_ARM_CFG`: The Spot Arm robot with delay PD and remote PD actuators.
"""


##
# Configuration
##


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
        effort_limit: Tensor | float = torch.inf,
        velocity_limit: Tensor | float = torch.inf,
        **kwargs,
    ):

        super().__init__(
            cfg=cfg,
            joint_names=joint_names,
            joint_ids=joint_ids,
            num_envs=num_envs,
            device=device,
            stiffness=stiffness,
            damping=damping,
            armature=armature,
            friction=friction,
            effort_limit=effort_limit,
            velocity_limit=velocity_limit,
            **kwargs,
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

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute the control action for the Spot robot with positional torque speed limits."""
        control_action = super().compute(control_action, joint_pos, joint_vel)

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


SPOT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        link_density=1.0e-8,
        asset_path="rl_deploy/spot/spot_with_arm.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SPOT_DEFAULT_POS,
        joint_pos=SPOT_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=HIP_EFFORT_LIMIT,
            stiffness=HIP_STIFFNESS,
            damping=HIP_DAMPING,
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_knee": SpotKneeActuatorCfg(
            joint_names_expr=[".*_kn"],
            joint_parameter_lookup=JOINT_PARAMETER_LOOKUP_TABLE,
            effort_limit=None,  # torque limits are handled based experimental data
            stiffness=KNEE_STIFFNESS,
            damping=KNEE_DAMPING,
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
            enable_torque_speed_limit=True,
        ),
        "spot_arm_sh0": DelayedPDActuatorCfg(
            joint_names_expr=["arm_sh0"],
            effort_limit=ARM_EFFORT_LIMIT[0],
            stiffness=ARM_STIFFNESS[0],
            damping=ARM_DAMPING[0],
            armature=ARM_ARMATURE[0],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_sh1": DelayedPDActuatorCfg(
            joint_names_expr=["arm_sh1"],
            effort_limit=ARM_EFFORT_LIMIT[1],
            stiffness=ARM_STIFFNESS[1],
            damping=ARM_DAMPING[1],
            armature=ARM_ARMATURE[1],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_el0": DelayedPDActuatorCfg(
            joint_names_expr=["arm_el0"],
            effort_limit=ARM_EFFORT_LIMIT[2],
            stiffness=ARM_STIFFNESS[2],
            damping=ARM_DAMPING[2],
            armature=ARM_ARMATURE[2],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_el1": DelayedPDActuatorCfg(
            joint_names_expr=["arm_el1"],
            effort_limit=ARM_EFFORT_LIMIT[3],
            stiffness=ARM_STIFFNESS[3],
            damping=ARM_DAMPING[3],
            armature=ARM_ARMATURE[3],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_wr0": DelayedPDActuatorCfg(
            joint_names_expr=["arm_wr0"],
            effort_limit=ARM_EFFORT_LIMIT[4],
            stiffness=ARM_STIFFNESS[4],
            damping=ARM_DAMPING[4],
            armature=ARM_ARMATURE[4],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_wr1": DelayedPDActuatorCfg(
            joint_names_expr=["arm_wr1"],
            effort_limit=ARM_EFFORT_LIMIT[5],
            stiffness=ARM_STIFFNESS[5],
            damping=ARM_DAMPING[5],
            armature=ARM_ARMATURE[5],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_f1x": DelayedPDActuatorCfg(
            joint_names_expr=["arm_f1x"],
            effort_limit=ARM_EFFORT_LIMIT[6],
            stiffness=ARM_STIFFNESS[6],
            damping=ARM_DAMPING[6],
            armature=ARM_ARMATURE[6],
            min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
    },
)

########################################################
# Pre-defined configs
########################################################

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "uniform_terrain": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.8, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.15
        ),
    },
)


########################################################
# Scene definition
########################################################


@configclass
class InterlimbSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    robot_to_ground_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


########################################################
# MDP settings
########################################################


class ArmLegJointBasePoseCommand(CommandTerm):
    """Command term that generates arm joint trajectory for spot arm."""

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
        ----
            cfg: The configuration parameters for the command term.
            env: The environment object.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.arm_joint_idxs = self.robot.find_joints(self.cfg.arm_joint_names)[0]
        self.leg_joint_idxs = torch.tensor(  # order: fl, fr, hl, hr
            [
                self.robot.find_joints(names)[0]
                for _, names in self.cfg.leg_joint_names.items()
            ],
            device=self.device,
        )
        self.leg_joint_names = [
            joint_name
            for _, names in self.cfg.leg_joint_names.items()
            for joint_name in self.robot.find_joints(names)[1]
        ]

        # create buffers to store the command
        # -- arm trajectory command
        self.arm_joint_start = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )
        self.arm_joint_goal = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )
        self.arm_joint_sub_goal = torch.zeros(
            self.num_envs, len(self.arm_joint_idxs), device=self.device
        )

        # -- leg trajectory command
        self.leg_names = ["fl", "fr", "hl", "hr"]
        total_leg_joints = torch.numel(self.leg_joint_idxs)
        self.leg_joint_start = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        self.leg_joint_goal = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        self.leg_joint_sub_goal = torch.zeros(
            self.num_envs, total_leg_joints, device=self.device
        )
        # -- indices for which leg is PD controlled
        self.command_leg_idxs = torch.zeros(self.num_envs, device=self.device).int()
        # -- indicator for whether any leg is PD controlled
        self.command_leg = torch.ones(self.num_envs, device=self.device).bool()
        # -- indices for the leg joint in command, in the order of fl, fr, hl, hr
        self.command_leg_joint_idxs = torch.arange(
            total_leg_joints, device=self.device
        ).view(len(self.leg_joint_idxs), -1)

        # -- torso pose command
        self.torso_roll_pitch_height_goal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.torso_projected_gravity_goal = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.gravity_vec = self.robot.data.GRAVITY_VEC_W.clone()

        # timer
        self.step_dt = env.step_dt
        self.timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (self.num_envs,),
                device=self.device,
            )
            / self.step_dt
        ).int()

        # Pre-compute feasible commands
        self.num_cached_command = 500000
        self.cached_command: dict[str, torch.Tensor] = dict()
        self.cached_command["roll"] = math_utils.sample_uniform(
            self.cfg.command_range_roll[0],
            self.cfg.command_range_roll[1],
            (self.num_cached_command,),
            device=self.device,
        )
        self.cached_command["pitch"] = math_utils.sample_uniform(
            self.cfg.command_range_pitch[0],
            self.cfg.command_range_pitch[1],
            (self.num_cached_command,),
            device=self.device,
        )
        self.cached_command["height"] = math_utils.sample_uniform(
            self.cfg.command_range_height[0],
            self.cfg.command_range_height[1],
            (self.num_cached_command,),
            device=self.device,
        )
        self.cfg.command_range_arm_joint = torch.tensor(
            self.cfg.command_range_arm_joint, device=self.device
        )
        self.cached_command["arm_joint"] = math_utils.sample_uniform(
            self.cfg.command_range_arm_joint[0],
            self.cfg.command_range_arm_joint[1],
            (self.num_cached_command, len(self.cfg.command_range_arm_joint[0])),
            device=self.device,
        )
        self.cfg.command_range_leg_joint = torch.tensor(
            self.cfg.command_range_leg_joint, device=self.device
        )
        self.cached_command["sampled_leg_joint"] = math_utils.sample_uniform(
            self.cfg.command_range_leg_joint[0],
            self.cfg.command_range_leg_joint[1],
            (self.num_cached_command, len(self.cfg.command_range_leg_joint[0])),
            device=self.device,
        )
        self.cached_command["sampled_leg_name"] = torch.randint(
            -1, 4, (self.num_cached_command,), device=self.device
        )

        # filter the commands based on command_which_leg
        assert self.cfg.command_which_leg in [-1, 0, 1, 2, 3, 4]
        if self.cfg.command_which_leg != 4:  # mix all leg command
            valid_command_mask = (
                self.cached_command["sampled_leg_name"] == self.cfg.command_which_leg
            )
            self.num_cached_command = valid_command_mask.sum().item()
            for k, v in self.cached_command.items():
                self.cached_command[k] = v[valid_command_mask]

        # -- metrics
        self.metrics["arm_joint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["leg_joint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_roll_pitch_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["base_height_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        self.feet_body_ids = torch.tensor([12, 16, 20, 24], device=self.device).repeat(
            self.env.num_envs, 1
        )
        self.feet_index = torch.arange(4, device=env.device).repeat(
            self.env.num_envs, 1
        )
        self.batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)

    def __str__(self) -> str:
        msg = "ArmLegJointBasePoseCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired arm joints in the environment frame. Shape is (num_envs, 22)."""
        value = torch.cat(
            [
                self.arm_joint_sub_goal,
                self.leg_joint_sub_goal,
                self.torso_roll_pitch_height_goal,
            ],
            dim=1,
        )
        # print("Commanded values: ", value[0])
        return value

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # -- compute the arm joint tracking error
        self.metrics["arm_joint_error"] = torch.norm(
            self.robot.data.joint_pos[:, self.arm_joint_idxs] - self.arm_joint_sub_goal,
            dim=1,
        )
        # -- compute the arm joint tracking error
        batch_indices = torch.arange(self.num_envs).view(-1, 1).repeat(1, 3)
        command_joint_idxs = self.command_leg_joint_idxs[self.command_leg_idxs]
        leg_joint_idxs = self.leg_joint_idxs[self.command_leg_idxs]
        self.metrics["leg_joint_error"] = torch.norm(
            self.robot.data.joint_pos[batch_indices, leg_joint_idxs]
            - self.leg_joint_sub_goal[batch_indices, command_joint_idxs],
            dim=1,
        )
        # -- compute the base pose tracking error
        self.metrics["base_roll_pitch_error"] = torch.norm(
            self.torso_projected_gravity_goal[:, :2]
            - self.robot.data.projected_gravity_b[:, :2],
            dim=1,
        )
        self.metrics["base_height_error"] = torch.abs(
            self.torso_roll_pitch_height_goal[:, 2] - self.robot.data.root_pos_w[:, 2]
        )

    def _resample_arm_command(
        self, env_ids: Sequence[int], sampled_command_idx: Sequence[int]
    ):
        self.arm_joint_start[env_ids] = torch.clamp(
            self.robot.data.joint_pos[env_ids][:, self.arm_joint_idxs],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_idxs, 0],
            self.robot.data.soft_joint_pos_limits[env_ids][:, self.arm_joint_idxs, 1],
        )
        self.arm_joint_sub_goal[env_ids] = self.arm_joint_start[env_ids]
        self.arm_joint_goal[env_ids] = self.cached_command["arm_joint"][
            sampled_command_idx
        ]

    def _resample_leg_command(
        self, env_ids: Sequence[int], sampled_command_idx: Sequence[int]
    ):
        self.command_leg_idxs[env_ids] = self.cached_command["sampled_leg_name"][
            sampled_command_idx
        ].int()
        self.command_leg_idxs[self.command_leg_idxs == -1] = (
            0  # get a dummy placeholder for no-PD-leg envs
        )

        self.leg_joint_start[env_ids] = 0.0
        self.leg_joint_sub_goal[env_ids] = 0.0
        self.leg_joint_goal[env_ids] = 0.0

        batch_indices = env_ids.view(-1, 1).repeat(1, 3)
        command_joint_idxs = self.command_leg_joint_idxs[
            self.command_leg_idxs[env_ids]
        ]  # commanded leg joint idxs in the command
        leg_joint_idxs = self.leg_joint_idxs[
            self.command_leg_idxs[env_ids]
        ]  # commanded leg joint idxs in the simulation

        self.leg_joint_start[batch_indices, command_joint_idxs] = (
            self.robot.data.joint_pos[batch_indices, leg_joint_idxs].clone()
        )
        self.leg_joint_sub_goal[env_ids] = self.leg_joint_start[env_ids].clone()
        self.leg_joint_goal[batch_indices, command_joint_idxs] = self.cached_command[
            "sampled_leg_joint"
        ][sampled_command_idx]

        self.command_leg[env_ids] = (
            self.cached_command["sampled_leg_name"][sampled_command_idx].int() != -1
        )
        self.leg_joint_start[~self.command_leg] = 0.0
        self.leg_joint_sub_goal[~self.command_leg] = 0.0
        self.leg_joint_goal[~self.command_leg] = 0.0

    def _resample_base_command(
        self, env_ids: Sequence[int], sampled_command_idx: Sequence[int]
    ):
        self.torso_roll_pitch_height_goal[env_ids, 0] = self.cached_command["roll"][
            sampled_command_idx
        ]
        self.torso_roll_pitch_height_goal[env_ids, 1] = self.cached_command["pitch"][
            sampled_command_idx
        ]
        self.torso_roll_pitch_height_goal[env_ids, 2] = self.cached_command["height"][
            sampled_command_idx
        ]

        # pre-compute torso_projected_gravity_goal
        # https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/rewards/corl_rewards.py#L148C17-L148C36
        quat_roll = math_utils.quat_from_angle_axis(
            self.torso_roll_pitch_height_goal[env_ids, 0], self.x_axis
        )
        quat_pitch = math_utils.quat_from_angle_axis(
            self.torso_roll_pitch_height_goal[env_ids, 1], self.y_axis
        )
        desired_base_quat = math_utils.quat_mul(quat_roll, quat_pitch)
        self.torso_projected_gravity_goal[env_ids] = math_utils.quat_rotate_inverse(
            desired_base_quat, self.gravity_vec[env_ids]
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample goal from the cached commands
        sampled_command_idx = torch.randint(
            0, self.num_cached_command, (len(env_ids),), device=self.device
        )

        # -- resample buffers for arm
        self._resample_arm_command(env_ids, sampled_command_idx)
        # -- resample buffers for leg
        self._resample_leg_command(env_ids, sampled_command_idx)
        # -- reset base pose
        self._resample_base_command(env_ids, sampled_command_idx)

        # reset timer
        self.traj_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.trajectory_time[0],
                self.cfg.trajectory_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.hold_timesteps[env_ids] = (
            math_utils.sample_uniform(
                self.cfg.hold_time[0],
                self.cfg.hold_time[1],
                (len(env_ids),),
                device=self.device,
            )
            / self.step_dt
        ).int()
        self.timer[env_ids] *= 0.0

    def _update_command(self):
        # step the timer
        self.timer += 1.0

        # update the mid_goal as timer goes
        reaching = self.timer <= self.traj_timesteps
        holding = torch.logical_and(
            self.traj_timesteps < self.timer,
            self.timer <= self.traj_timesteps + self.hold_timesteps,
        )
        reset = self.timer > self.traj_timesteps + self.hold_timesteps

        reaching_ids = reaching.nonzero(as_tuple=False).squeeze(-1)
        holding_ids = holding.nonzero(as_tuple=False).squeeze(-1)
        reset_ids = reset.nonzero(as_tuple=False).squeeze(-1)

        if len(reaching_ids) > 0:
            self.arm_joint_sub_goal[reaching_ids] = torch.lerp(
                self.arm_joint_start[reaching_ids],
                self.arm_joint_goal[reaching_ids],
                (self.timer / self.traj_timesteps)[reaching_ids].reshape(-1, 1),
            )
            self.leg_joint_sub_goal[reaching_ids] = torch.lerp(
                self.leg_joint_start[reaching_ids],
                self.leg_joint_goal[reaching_ids],
                (self.timer / self.traj_timesteps)[reaching_ids].reshape(-1, 1),
            )

        if len(holding_ids) > 0:
            self.arm_joint_sub_goal[holding_ids] = self.arm_joint_goal[
                holding_ids
            ].clone()
            self.leg_joint_sub_goal[holding_ids] = self.leg_joint_goal[
                holding_ids
            ].clone()

        if len(reset_ids) > 0:
            self._resample(reset_ids)


@configclass
class ArmLegJointBasePoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    class_type: type = ArmLegJointBasePoseCommand

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    trajectory_time: tuple[float, float] = (1.0, 3.0)
    """Length of the trajectory in seconds."""

    hold_time: tuple[float, float] = (0.5, 2.0)
    """Length of the arm holding in positon in seconds."""

    arm_joint_names: tuple[str, ...] = MISSING
    leg_joint_names: dict = MISSING

    command_range_roll: tuple[float, float] = (-0.35, 0.35)
    command_range_pitch: tuple[float, float] = (-0.35, 0.35)
    command_range_height: tuple[float, float] = (0.25, 0.65)
    command_range_arm_joint: tuple[list[float, ...], list[float, ...]] = (
        [-2.61799, -3.14159, 0.0, -2.79252, -1.8326, -2.87988, -1.5708],
        [3.14157, 0.52359, 3.14158, 2.79252, 1.83259, 2.87979, 0.0],
    )
    command_range_leg_joint: tuple[list[float, ...], list[float, ...]] = (
        [-0.7854, -0.89884, -2.7929],
        [0.78539, 2.29511, 0.0],
    )
    """Command sample ranges."""

    command_which_leg: int = 4
    """Which leg to command: -1: no leg; [0, 1, 2, 3]: [FL, FR, HL, HR]; 4: all leg"""


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = isaac_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=isaac_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    arm_leg_joint_base_pose = ArmLegJointBasePoseCommandCfg(
        resampling_time_range=(1e6, 1e6),
        arm_joint_names=ARM_JOINT_NAMES,
        leg_joint_names=LEG_JOINT_NAMES,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = MixedPDArmMultiLegJointPositionActionCfg(
        asset_name="robot",
        #joint_names=LEG_JOINT_NAMES_ALL,
        joint_names=["[fh].*"],
        command_name="arm_leg_joint_base_pose",
        arm_joint_names=ARM_JOINT_NAMES,
        leg_joint_names=LEG_JOINT_NAMES,
        scale=0.2,
        # preserve_order=True,
    )


def time(env):
    sim_time: float = env._sim_step_counter * env.step_dt
    sim_tensor = torch.tensor(sim_time, device=env.device).unsqueeze(0)
    return sim_tensor


def default_joint_pos(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_ids(env, asset_cfg):
    return (
        torch.tensor(asset_cfg.joint_ids, device=env.device)
        .unsqueeze(0)
        .repeat(env.num_envs, 1)
    )


def time_(env: ManagerBasedEnv):
    sim_time: float = env._sim_step_counter * env.step_dt
    sim_tensor = torch.tensor([sim_time], device=env.device).repeat(env.num_envs, 1)
    return sim_tensor

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""


    @configclass
    class SpotObs(ObsGroup):
        """Observations that are the same as the spot robot."""
        sim_time = ObsTerm(func=time_)

        root_lin_vel_w = ObsTerm(
            func=isaac_mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        root_ang_vel_w = ObsTerm(
            func=isaac_mdp.root_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        root_quat_w = ObsTerm(
            func=isaac_mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )
    
        joint_effort = ObsTerm(
            func=isaac_mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )
    
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    @configclass
    class PolicyCfg(ObsGroup):
        """Observations that are required for the policy."""

        base_lin_vel = ObsTerm(
            func=isaac_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=isaac_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=isaac_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=isaac_mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        commands = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "arm_leg_joint_base_pose"},
        )
        joint_pos = ObsTerm(
            func=isaac_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES_ALL + ARM_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=isaac_mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=LEG_JOINT_NAMES_ALL + ARM_JOINT_NAMES,
                    preserve_order=True,
                )
            },
        )
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    class DebugCfg(PolicyCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    spot: SpotObs = SpotObs()
    debug: DebugCfg = DebugCfg()


def reset_joints_to_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    target_pos: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints to a given position and velocity with a given range.

    This function samples random values from the given ranges around the target joint positions and velocities.
    The ranges are clipped to fit inside the soft joint limits. The sampled values are then set into the physics
    simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if target_pos.ndim == 1:
        target_pos = target_pos.unsqueeze(0).expand(len(env_ids), -1).to(env.device)
    target_vel = torch.zeros_like(target_pos)

    asset.write_joint_state_to_sim(
        target_pos, target_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )

@configclass
class EventCfg:
    """Configuration for events."""

    reset_to_default = EventTerm(func=isaac_mdp.reset_scene_to_default, mode="reset")
    reset_arm_joints = EventTerm(
        func=reset_joints_to_position,
        mode="reset",
        params={
            "target_pos": torch.tensor([ARM_STOW_JOINT_POS[k] for k in ARM_JOINT_NAMES]),
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ...


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body"]),
            "threshold": 1.0,
        },
    )


########################################################
# Environment configuration
########################################################


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", {
            "actions": self._env.action_manager.action,
            "leg_processed_actions": self._env.action_manager.get_term("joint_pos").processed_actions,
            "arm_processed_actions": self._env.action_manager.get_term("joint_pos").arm_processed_actions,
        }

class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf


class PostStepProcessedActionsRecorder(RecorderTerm):
    """Recorder term that records processed actions at the end of each step."""

    def record_post_step(self):
        processed_actions = None

        # Loop through active terms and concatenate their processed actions
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(
                term_name
            ).processed_actions.clone()
            if processed_actions is None:
                processed_actions = term_actions
            else:
                processed_actions = torch.cat([processed_actions, term_actions], dim=-1)

        return "processed_actions", processed_actions


##
# State recorders.
##


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = PostStepStatesRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = PreStepActionsRecorder


@configclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatPolicyObservationsRecorder


@configclass
class PostStepProcessedActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term."""

    class_type: type[RecorderTerm] = PostStepProcessedActionsRecorder


##
# Recorder manager configurations.
##


@configclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = (
        PreStepFlatPolicyObservationsRecorderCfg()
    )
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()
    dataset_export_dir_path: str = "datasets"
    dataset_filename: str = (
        f"isaac_spot_dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    export_in_close = True


@configclass
class InterlimbEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: InterlimbSceneCfg = InterlimbSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        # self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.robot_to_ground_contact_forces is not None:
            self.scene.robot_to_ground_contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class SpotInterlimbEnvCfg_Phase_1(InterlimbEnvCfg):
    """Configuration for the phase 1 environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None

        # Don't hold the leg
        self.commands.arm_leg_joint_base_pose.command_which_leg = -1

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 0.5
        self.scene.terrain.terrain_type = "plane"
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.50, 0.50)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.)
        self.commands.base_velocity.debug_vis = True
        self.commands.arm_leg_joint_base_pose.command_range_roll = (0.0, 0.0)
        self.commands.arm_leg_joint_base_pose.command_range_pitch = (0.0, 0.0)
        self.commands.arm_leg_joint_base_pose.command_range_height = (0.0, 0.0)
        self.commands.arm_leg_joint_base_pose.command_range_arm_joint = ( 
            [0.0, -3.1415, 3.1415, 1.5655, 0.00, 0.0, 0.0],
            [0.0, -3.1415, 3.1415, 1.5655, 0.00, 0.0, 0.0],
        )
        self.commands.arm_leg_joint_base_pose.command_which_leg = -1 
        
        