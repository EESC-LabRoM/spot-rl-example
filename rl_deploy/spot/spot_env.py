import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from rl_deploy.spot.constants import (
    ORDERED_JOINT_NAMES_SPOT_ARM,
    ORDERED_JOINT_NAMES_SPOT_BASE,
)
from rl_deploy.spot.isaac_model import (
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
    SPOT_DEFAULT_POS,
    SPOT_STOWED_JOINT_POS,
    SpotKneeActuatorCfg,
)

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
        joint_pos=SPOT_STOWED_JOINT_POS,
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
        joint_names=ORDERED_JOINT_NAMES_SPOT_BASE,
        scale=1,
        use_default_offset=False,
        preserve_order=True,
    )

    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ORDERED_JOINT_NAMES_SPOT_ARM,
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


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class DeployObs(ObsGroup):
        """Observations for deploy on spot group."""

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
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE
                    + ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE
                    + ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
        )
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
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE,
                    preserve_order=True,
                )
            },
        )
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
        )

        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_BASE,
                    preserve_order=True,
                )
            },
        )

        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=ORDERED_JOINT_NAMES_SPOT_ARM,
                    preserve_order=True,
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    spot: DeployObs = DeployObs()


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


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    reset_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class SpotFlatEnvCfg(ManagerBasedEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

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

    def __post_init__(self):
        # post init of parent
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz from 500hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"


