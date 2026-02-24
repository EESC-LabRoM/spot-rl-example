# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
"""Data for Boston Dynamics Spot robot."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
)
from isaaclab.actuators.actuator_cfg import RemotizedPDActuatorCfg
from isaaclab.actuators.actuator_pd import RemotizedPDActuator
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import LinearInterpolation, configclass
from isaaclab.utils.types import ArticulationActions
from torch._tensor import Tensor

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

# That is the arm stowed
SPOT_STOWED_JOINT_POS: dict[str, float] = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.1415,
    "arm_el0": 3.1415,
    "arm_el1": 1.5655,
    "arm_wr0": 0.00,
    "arm_wr1": 1.5692,
    "arm_f1x": 0,
    "[fh]l_hx": 0.1,  # all left hip_x
    "[fh]r_hx": -0.1,  # all right hip_x
    "f[rl]_hy": 0.9,  # front hip_y
    "h[rl]_hy": 1.1,  # hind hip_y
    ".*_kn": -1.5,  # all knees
}

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


# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Custom actuator definitions for Spot"""


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


"""Configuration for the Boston Dynamics robot.

The following configuration parameters are available:

* :obj:`SPOT_ARM_CFG`: The Spot Arm robot with delay PD and remote PD actuators.
"""
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
        semantic_tags=[("class", "robot")],
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=None)
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
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "spot_knee": SpotKneeActuatorCfg(
            joint_names_expr=[".*_kn"],
            joint_parameter_lookup=JOINT_PARAMETER_LOOKUP_TABLE,
            effort_limit=None,  # torque limits are handled based experimental data
            stiffness=KNEE_STIFFNESS,
            damping=KNEE_DAMPING,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
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
