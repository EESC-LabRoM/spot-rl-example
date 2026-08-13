# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from rl_deploy.orbit.orbit_constants import (
    ORDERED_JOINT_NAMES_ARM_ISAAC,
    ORDERED_JOINT_NAMES_BASE_ISAAC,
)
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from rl_deploy.orbit.orbit_constants import ORDERED_JOINT_NAMES_ISAAC
from rl_deploy.utils.dict_tools import dict_from_lists, set_matching
from rl_deploy.spot.constants import DEFAULT_K_Q_P, DEFAULT_K_QD_P, DOF


class Ref(yaml.YAMLObject):
    yaml_tag = "tag:yaml.org,2002:python/tuple"

    def __init__(self, val):
        self.val = val

    @classmethod
    def from_yaml(cls, loader, node):
        return tuple(node.value)


class Slices(yaml.YAMLObject):
    yaml_tag = "python/object/apply:builtins.slice"

    @classmethod
    def from_yaml(cls, loader, node):
        values = node.value
        if len(values) == 1:
            return slice(values[0].value)
        elif len(values) == 2:
            return slice(values[0].value, values[1].value)
        elif len(values) == 3:
            return slice(values[0].value, values[1].value, values[2].value)


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", Ref.from_yaml)
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", Slices.from_yaml
)


@dataclass
class OrbitConfig:
    """dataclass holding data extracted from orbits training configuration"""

    kp: List[float]
    kd: List[float]
    default_joints: List[float]
    standing_height: float
    action_scale: float
    # Relic Plus policies trained with the 65-D observation schema expect a
    # height command in the [0.5, 0.7] training range.  Keep the deployment
    # default explicit and configurable without changing older env.yaml files.
    height_command: float = 0.6
    gait_frequency: float = 1.5
    foot_height_max: float = 0.15
    foot_radius: float = 0.036
    gait_swing_fraction: float = 0.25
    gait_pattern: str = "static_crawl"
    standing_velocity_threshold: float = 0.05
    control_period_s: float = 0.02
    orientation_command: List[float] = field(default_factory=lambda: [0.0, 0.0])
    policy_inputs: List[str] = field(default_factory=list)
    gait_phase_offsets: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.75, 0.25]
    )


@dataclass(frozen=True)
class PolicyBundle:
    directory: Path
    policy_file: Path
    manifest: dict
    config: OrbitConfig


def add_policy_bundle_argument(parser, default: os.PathLike | str):
    """Add the shared policy-bundle CLI, including the historical demo spelling."""
    parser.add_argument(
        "-policy_file_path",
        "--policy-file-path",
        "--policy-dir",
        dest="policy_dir",
        type=Path,
        default=Path(default),
        help=(
            "Directory containing exactly one ONNX model and policy.yaml. "
            "Defaults to rl_deploy/configs."
        ),
    )
    parser.add_argument(
        "--arm",
        choices=("off", "easy", "full"),
        default="off",
        help=(
            "Replay the arm disturbance the policy was trained against: easy = the arm_easy "
            "tier, full = every tier. Defaults to off, which holds the arm stowed."
        ),
    )


def detect_config_file(directory: os.PathLike | str) -> dict | None:
    """find and parse json or yaml file in policy directory

    arguments
    directory -- path where policy and training configuration can be found

    return dictionary from config file
    """
    files = [f for f in os.listdir(directory) if f.endswith("env.json")]
    if len(files) == 1:
        filepath = os.path.join(directory, files[0])
        with open(filepath) as f:
            return json.load(f)

    files = [f for f in os.listdir(directory) if f.endswith("env.yaml")]
    if len(files) == 1:
        filepath = os.path.join(directory, files[0])
        with open(filepath) as f:
            return yaml.safe_load(f)

    return None


def detect_policy_file(directory: os.PathLike | str) -> str | None:
    """find onnx file in policy directory

    arguments
    directory -- path where policy and training configuration can be found

    return filepath to onnx file
    """
    if os.path.isfile(directory):
        if str(directory).endswith(".onnx"):
            return str(directory)
        raise ValueError(f"Policy path is not an ONNX file: {directory}")
    files = [f for f in os.listdir(directory) if f.endswith(".onnx")]
    if len(files) == 1:
        return os.path.join(directory, files[0])
    raise ValueError(
        f"Expected exactly one ONNX file in {directory}, found {len(files)}"
    )


def load_policy_manifest(
    directory: os.PathLike | str, policy_file: os.PathLike | str | None = None
) -> dict:
    """Load and minimally validate the policy bundle's runtime contract."""
    directory = os.path.dirname(directory) if os.path.isfile(directory) else directory
    path = os.path.join(directory, "policy.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Policy bundle is missing {path}")
    with open(path) as f:
        manifest = yaml.safe_load(f)
    if manifest.get("contract_version") != 2:
        raise ValueError(
            f"Unsupported policy contract version: {manifest.get('contract_version')}"
        )
    declared_model = manifest.get("model", {}).get("file")
    if not declared_model:
        raise ValueError("Policy manifest does not declare model.file")
    expected_model = (
        os.path.basename(policy_file)
        if policy_file is not None
        else os.path.basename(detect_policy_file(directory))
    )
    if declared_model != expected_model:
        raise ValueError(
            f"Policy manifest declares {declared_model}, but selected {expected_model}"
        )
    return manifest


def resolve_policy_bundle(
    policy_location: os.PathLike | str,
    env_directory: os.PathLike | str,
) -> PolicyBundle:
    """Resolve and validate the portable policy bundle before starting a backend."""
    policy_file = Path(detect_policy_file(policy_location)).resolve()
    directory = policy_file.parent
    manifest = load_policy_manifest(directory, policy_file)
    env_config = detect_config_file(env_directory)
    if env_config is None:
        raise FileNotFoundError(
            f"Expected exactly one env.yaml or env.json in {env_directory}"
        )
    config = load_configuration(env_config, manifest)
    return PolicyBundle(directory, policy_file, manifest, config)


def load_configuration(env_config: dict, policy_manifest: dict | None = None) -> OrbitConfig:
    """parse json file and populate an OrbitConfig dataclass

    arguments
    file -- the path to the json file containing training configuration

    return OrbitConfig containing needed training configuration
    """

    joint_kp = dict_from_lists(ORDERED_JOINT_NAMES_ISAAC, [None] * 19)
    joint_kd = dict_from_lists(ORDERED_JOINT_NAMES_ISAAC, [None] * 19)
    joint_offsets = dict_from_lists(ORDERED_JOINT_NAMES_ISAAC, [None] * 19)

    actuators = env_config["scene"]["robot"]["actuators"]

    for group in actuators.keys():
        regex = re.compile(actuators[group]["joint_names_expr"][0])

        set_matching(joint_kp, regex, actuators[group]["stiffness"])
        set_matching(joint_kd, regex, actuators[group]["damping"])

    default_joint_data = env_config["scene"]["robot"]["init_state"]["joint_pos"]
    default_joint_expressions = default_joint_data.keys()
    for expression in default_joint_expressions:
        regex = re.compile(expression)
        set_matching(joint_offsets, regex, default_joint_data[expression])

    action_scale = env_config["actions"]["joint_pos"]["scale"]
    standing_height = env_config["scene"]["robot"]["init_state"]["pos"][2]
    observations = (policy_manifest or {}).get("observations", {})
    gait = (policy_manifest or {}).get("gait", {})
    control = (policy_manifest or {}).get("control", {})
    model = (policy_manifest or {}).get("model", {})
    action_contract = (policy_manifest or {}).get("actions", {})
    height_command = float(
        observations.get(
            "height_command", env_config.get("policy_height_command", 0.6)
        )
    )
    orientation_command = [
        float(value) for value in observations.get("base_orientation_command", [0.0, 0.0])
    ]
    gait_frequency = float(
        gait.get("frequency", env_config.get("policy_gait_frequency", 1.5))
    )
    foot_height_max = float(
        gait.get(
            "foot_height_max", env_config.get("policy_foot_height_max", 0.15)
        )
    )
    foot_radius = float(gait.get("foot_radius", 0.036))
    gait_swing_fraction = float(
        gait.get(
            "swing_fraction",
            env_config.get("policy_gait_swing_fraction", 0.25),
        )
    )
    gait_phase_offsets = [
        float(value)
        for value in gait.get(
            "phase_offsets",
            env_config.get("policy_gait_phase_offsets", [0.0, 0.5, 0.75, 0.25]),
        )
    ]
    gait_pattern = str(gait.get("pattern", "static_crawl"))
    standing_velocity_threshold = float(gait.get("standing_velocity_threshold", 0.05))
    control_period_s = float(control.get("period_s", 0.02))
    if gait_pattern not in ("static_crawl", "trot"):
        raise ValueError(f"Unsupported deployment gait pattern: {gait_pattern}")
    if len(orientation_command) != 2 or len(gait_phase_offsets) != 4:
        raise ValueError("Policy manifest command dimensions are invalid")
    if policy_manifest:
        if control_period_s != 0.02:
            raise ValueError(
                f"Spot deployment requires a 0.02 s control period, got {control_period_s}"
            )
        if action_contract.get("output") != "absolute_joint_positions":
            raise ValueError("Policy must output absolute_joint_positions")
        if action_contract.get("last_actions_reset") != "reference_joint_positions":
            raise ValueError("Unsupported previous-action reset contract")
        if (policy_manifest.get("recurrent_state") or {}).get("reset") != "zeros":
            raise ValueError("Unsupported recurrent-state reset contract")
        if list(action_contract.get("joint_names", [])) != list(
            ORDERED_JOINT_NAMES_BASE_ISAAC
        ):
            raise ValueError("Policy manifest joint ordering does not match deployment")

    # Override the arm with default values for kp, kd
    for joint_name in ORDERED_JOINT_NAMES_ARM_ISAAC:
        joint_kp[joint_name] = DEFAULT_K_Q_P[DOF[joint_name.upper()]]
        joint_kd[joint_name] = DEFAULT_K_QD_P[DOF[joint_name.upper()]]
        print(
            f"Setting {joint_name} kp to {joint_kp[joint_name]} and kd to {joint_kd[joint_name]}"
        )

    return OrbitConfig(
        kp=joint_kp,
        kd=joint_kd,
        default_joints=joint_offsets,
        standing_height=standing_height,
        action_scale=action_scale,
        height_command=height_command,
        gait_frequency=gait_frequency,
        foot_height_max=foot_height_max,
        foot_radius=foot_radius,
        gait_swing_fraction=gait_swing_fraction,
        gait_phase_offsets=gait_phase_offsets,
        gait_pattern=gait_pattern,
        standing_velocity_threshold=standing_velocity_threshold,
        control_period_s=control_period_s,
        orientation_command=orientation_command,
        policy_inputs=list(model.get("inputs", [])),
    )
