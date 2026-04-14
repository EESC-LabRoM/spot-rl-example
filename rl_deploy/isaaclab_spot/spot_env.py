from __future__ import annotations

from datetime import datetime

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
)
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from rl_deploy.isaaclab_spot.isaac_model import (
    ARM_ARMATURE,
    ARM_DAMPING,
    ARM_EFFORT_LIMIT,
    ARM_STIFFNESS,
    HIP_DAMPING,
    HIP_EFFORT_LIMIT,
    HIP_STIFFNESS,
    JOINT_PARAMETER_LOOKUP_TABLE,
    KNEE_DAMPING,
    KNEE_STIFFNESS,
    SPOT_DEFAULT_JOINT_POS,
    SPOT_DEFAULT_POS,
    SpotKneeActuatorCfg,
)
from rl_deploy.spot.constants import (
    ORDERED_JOINT_NAMES_SPOT,
    ORDERED_JOINT_NAMES_SPOT_ARM,
    ORDERED_JOINT_NAMES_SPOT_BASE,
)

_arm_poses_cache: dict[str, torch.Tensor] = {}


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


@configclass
class SpotActionsDeployCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ORDERED_JOINT_NAMES_SPOT,
        scale=1,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        ),
    )


def time_(env: ManagerBasedEnv):
    sim_time: float = env._sim_step_counter * env.step_dt
    sim_tensor = torch.tensor([sim_time], device=env.device).repeat(env.num_envs, 1)
    return sim_tensor


def default_joint_pos(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_ids(env, asset_cfg):
    return torch.tensor(asset_cfg.joint_ids).unsqueeze(0)


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class SpotObs(ObsGroup):
        """Observations for spot group."""

        sim_time = ObsTerm(func=time_)

        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        root_ang_vel_w = ObsTerm(
            func=mdp.root_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        root_quat_w = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )

        joint_effort = ObsTerm(
            func=mdp.joint_effort,
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
    class DebugObs(ObsGroup):
        """Observations for deploy on spot group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )

        default_joint_pos = ObsTerm(
            func=default_joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT,
                    preserve_order=True,
                )
            },
        )

        joint_ids = ObsTerm(
            func=joint_ids,
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

    # observation groups
    spot: SpotObs = SpotObs()
    debug: DebugObs = DebugObs()


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


ARM_STOW_JOINT_POS: dict[str, float] = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.1415,
    "arm_el0": 3.1415,
    "arm_el1": 1.5655,
    "arm_wr0": 0.00,
    "arm_wr1": -1.5655,
    "arm_f1x": 0,
}


def _load_arm_poses_from_csv(
    csv_path: str,
    arm_joint_names: list[str],
    device: torch.device | str,
    up_sample_stow: int = 1000,
) -> torch.Tensor:
    """Return a ``(N, num_arm_joints)`` tensor with all arm poses from the CSV.

    Results are cached by ``csv_path`` so the file is read only once.
    The ``ARM_STOW_JOINT_POS`` is appended to the loaded poses.
    """
    cache_key = csv_path
    if cache_key not in _arm_poses_cache:
        # df = pd.read_csv(csv_path)
        # csv_poses = torch.tensor(
        #     df[arm_joint_names].values,
        #     dtype=torch.float32,
        # )

        # Create stow pose tensor in the order of arm_joint_names
        stow_pose = torch.tensor(
            [[ARM_STOW_JOINT_POS.get(name, 0.0) for name in arm_joint_names]],
            dtype=torch.float32,
        ).repeat(up_sample_stow, 1)

        # Concatenate stow pose to the loaded poses
        _arm_poses_cache[cache_key] = (
            stow_pose  # torch.cat([stow_pose, csv_poses], dim=0)
        )

    return _arm_poses_cache[cache_key].to(device)


def reset_base_and_arm_joints_from_csv(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    csv_path: str,
    arm_joint_names: list[str],
    base_joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset base joints around their default positions and arm joints to poses sampled from a CSV.

    Base joints are randomized uniformly within ``position_range`` / ``velocity_range`` of their
    default values (identical behaviour to :func:`reset_joints_around_default`).
    Arm joints are set to a randomly chosen row from ``csv_path``, with zero velocity.

    The CSV is read once and cached in memory for the lifetime of the process.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        position_range: ``(min, max)`` offset applied around the default base-joint positions.
        velocity_range: ``(min, max)`` offset applied around the default base-joint velocities.
        csv_path: Absolute path to the CSV file containing arm joint poses.  The file must
            contain the columns listed in ``arm_joint_names`` in that exact order.
        arm_joint_names: Ordered list of arm joint names matching the CSV column headers.
        base_joint_names: Ordered list of base (leg) joint names to randomise.
        asset_cfg: Scene entity config identifying the robot articulation.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    arm_joint_ids, _ = asset.find_joints(arm_joint_names, preserve_order=True)
    base_joint_ids, _ = asset.find_joints(base_joint_names, preserve_order=True)

    # ------------------------------------------------------------------
    # Build full joint state starting from the current sim state so that
    # joints we do NOT touch keep their existing values.
    # ------------------------------------------------------------------
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # -- Base joints: randomise around default --
    base_default_pos = asset.data.default_joint_pos[env_ids][:, base_joint_ids]
    base_default_vel = asset.data.default_joint_vel[env_ids][:, base_joint_ids]

    base_min_pos = base_default_pos + position_range[0]
    base_max_pos = base_default_pos + position_range[1]
    base_min_vel = base_default_vel + velocity_range[0]
    base_max_vel = base_default_vel + velocity_range[1]

    # Clip to soft joint limits
    base_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, base_joint_ids]
    base_min_pos = torch.clamp(
        base_min_pos, min=base_pos_limits[..., 0], max=base_pos_limits[..., 1]
    )
    base_max_pos = torch.clamp(
        base_max_pos, min=base_pos_limits[..., 0], max=base_pos_limits[..., 1]
    )

    base_vel_limits = asset.data.soft_joint_vel_limits[env_ids][:, base_joint_ids]
    base_min_vel = torch.clamp(base_min_vel, min=-base_vel_limits, max=base_vel_limits)
    base_max_vel = torch.clamp(base_max_vel, min=-base_vel_limits, max=base_vel_limits)

    joint_pos[:, base_joint_ids] = sample_uniform(
        base_min_pos, base_max_pos, base_min_pos.shape, base_min_pos.device
    )
    joint_vel[:, base_joint_ids] = sample_uniform(
        base_min_vel, base_max_vel, base_min_vel.shape, base_min_vel.device
    )

    # -- Arm joints: sample a random row from the CSV per environment --
    arm_poses = _load_arm_poses_from_csv(csv_path, arm_joint_names, asset.device)
    num_poses = arm_poses.shape[0]
    num_envs_to_reset = len(env_ids)
    sampled_indices = torch.randint(
        0, num_poses, (num_envs_to_reset,), device=asset.device
    )
    joint_pos[:, arm_joint_ids] = arm_poses[sampled_indices]
    joint_vel[:, arm_joint_ids] = 0.0
    print("Initial Joint Pos: ", joint_pos)
    print("Initial Joint Vel: ", joint_vel)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    reset_robot_joints = EventTerm(
        func=reset_base_and_arm_joints_from_csv,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "csv_path": "stow.csv",
            "arm_joint_names": ORDERED_JOINT_NAMES_SPOT_ARM,
            "base_joint_names": ORDERED_JOINT_NAMES_SPOT_BASE,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf["spot"] | self._env.obs_buf["debug"]


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
class SpotFlatEnvCfg(ManagerBasedEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=2.5)

    curriculum = None

    # Basic settings'
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions = SpotActionsDeployCfg()

    # MDP setting
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer
    viewer = ViewerCfg(
        eye=(5.0, 5.0, 5.0), origin_type="asset_root", env_index=0, asset_name="robot"
    )
    recorders = ActionStateRecorderManagerCfg()

    def __post_init__(self):
        # post init of parent
        self.sim.dt = 0.002  # 200 Hz from 500hz
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
