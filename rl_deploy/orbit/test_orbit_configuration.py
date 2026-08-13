import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rl_deploy.orbit.orbit_configuration import (
    add_policy_bundle_argument,
    detect_config_file,
    detect_policy_file,
    load_configuration,
    load_policy_manifest,
    resolve_policy_bundle,
)
from rl_deploy.orbit.orbit_constants import ORDERED_JOINT_NAMES_BASE_ISAAC


class PolicyBundleConfigurationTest(unittest.TestCase):
    def test_policy_argument_defaults_to_configs_and_preserves_aliases(self):
        default = Path("/portable/configs")
        for arguments in (
            [],
            ["--policy-dir", "/bundle"],
            ["--policy-file-path", "/bundle"],
            ["-policy_file_path", "/bundle"],
        ):
            parser = argparse.ArgumentParser()
            add_policy_bundle_argument(parser, default)
            options = parser.parse_args(arguments)
            expected = default if not arguments else Path("/bundle")
            self.assertEqual(options.policy_dir, expected)

    def test_arm_argument_defaults_to_off_and_accepts_tiers(self):
        default = Path("/portable/configs")
        for arguments, expected in (
            ([], "off"),
            (["--arm", "easy"], "easy"),
            (["--arm", "full"], "full"),
        ):
            parser = argparse.ArgumentParser()
            add_policy_bundle_argument(parser, default)
            self.assertEqual(parser.parse_args(arguments).arm, expected)

        parser = argparse.ArgumentParser()
        add_policy_bundle_argument(parser, default)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parser.parse_args(["--arm", "bogus"])

    def test_loads_versioned_manifest_for_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "policy.onnx"
            model.touch()
            (root / "policy.yaml").write_text(
                yaml.safe_dump(
                    {
                        "contract_version": 2,
                        "model": {"file": model.name},
                    }
                )
            )

            selected = detect_policy_file(root)
            manifest = load_policy_manifest(root, selected)

            self.assertEqual(selected, str(model))
            self.assertEqual(manifest["contract_version"], 2)

    def test_rejects_old_manifest_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "policy.onnx"
            model.touch()
            (root / "policy.yaml").write_text(
                yaml.safe_dump({"contract_version": 1, "model": {"file": model.name}})
            )
            with self.assertRaisesRegex(ValueError, "version: 1"):
                load_policy_manifest(root, model)

    def test_rejects_manifest_model_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "actual.onnx"
            model.touch()
            (root / "policy.yaml").write_text(
                yaml.safe_dump(
                    {
                        "contract_version": 2,
                        "model": {"file": "other.onnx"},
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "declares other.onnx"):
                load_policy_manifest(root, model)

    def test_rejects_ambiguous_policy_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.onnx").touch()
            (root / "two.onnx").touch()

            with self.assertRaisesRegex(ValueError, "exactly one ONNX"):
                detect_policy_file(root)

    def test_resolver_loads_environment_and_manifest_as_one_preflight(self):
        manifest = {"contract_version": 2}
        config = object()
        with (
            patch(
                "rl_deploy.orbit.orbit_configuration.detect_policy_file",
                return_value="/bundle/policy.onnx",
            ),
            patch(
                "rl_deploy.orbit.orbit_configuration.load_policy_manifest",
                return_value=manifest,
            ),
            patch(
                "rl_deploy.orbit.orbit_configuration.detect_config_file",
                return_value={"env": "config"},
            ),
            patch(
                "rl_deploy.orbit.orbit_configuration.load_configuration",
                return_value=config,
            ),
        ):
            bundle = resolve_policy_bundle("/bundle", "/configs")

        self.assertEqual(bundle.policy_file, Path("/bundle/policy.onnx"))
        self.assertIs(bundle.manifest, manifest)
        self.assertIs(bundle.config, config)

    def test_loads_faster_v2_crawl_and_trot_contracts(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        env_config = detect_config_file(configs)
        base = {
            "contract_version": 2,
            "model": {"file": "policy.onnx", "inputs": ["observations", "hidden_state"]},
            "control": {"period_s": 0.02},
            "observations": {
                "height_command": 0.58,
                "base_orientation_command": [0.0, 0.0],
            },
            "actions": {
                "joint_names": list(ORDERED_JOINT_NAMES_BASE_ISAAC),
                "output": "absolute_joint_positions",
                "last_actions_reset": "reference_joint_positions",
            },
            "recurrent_state": {"reset": "zeros"},
        }
        for pattern, frequency in (("static_crawl", 1.4), ("trot", 2.5)):
            manifest = dict(base)
            manifest["gait"] = {
                "pattern": pattern,
                "frequency": frequency,
                "foot_height_max": 0.14,
                "foot_radius": 0.036,
                "swing_fraction": 0.25 if pattern == "static_crawl" else 0.5,
                "phase_offsets": [0.0, 0.5, 0.75, 0.25],
                "standing_velocity_threshold": 0.05,
            }
            config = load_configuration(env_config, manifest)
            self.assertEqual(config.gait_pattern, pattern)
            self.assertEqual(config.gait_frequency, frequency)
            self.assertEqual(config.foot_radius, 0.036)

    def test_foot_radius_defaults_for_existing_v2_bundle(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        env_config = detect_config_file(configs)
        manifest = {
            "contract_version": 2,
            "model": {"file": "policy.onnx"},
            "control": {"period_s": 0.02},
            "actions": {
                "joint_names": list(ORDERED_JOINT_NAMES_BASE_ISAAC),
                "output": "absolute_joint_positions",
                "last_actions_reset": "reference_joint_positions",
            },
            "recurrent_state": {"reset": "zeros"},
            "gait": {"pattern": "static_crawl"},
        }
        self.assertEqual(load_configuration(env_config, manifest).foot_radius, 0.036)

    def test_rejects_incompatible_v2_runtime_contracts(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        env_config = detect_config_file(configs)
        manifest = {
            "contract_version": 2,
            "model": {"file": "policy.onnx"},
            "control": {"period_s": 0.02},
            "actions": {
                "joint_names": list(ORDERED_JOINT_NAMES_BASE_ISAAC),
                "output": "absolute_joint_positions",
                "last_actions_reset": "reference_joint_positions",
            },
            "recurrent_state": {"reset": "zeros"},
            "gait": {"pattern": "static_crawl"},
        }
        cases = (
            ("gait", "pattern", "pace", "gait pattern"),
            ("actions", "last_actions_reset", "default_leg_joint_offsets", "previous-action"),
            ("actions", "joint_names", list(ORDERED_JOINT_NAMES_BASE_ISAAC) + ["arm_sh0"], "joint ordering"),
        )
        for section, key, value, message in cases:
            original = manifest[section][key]
            manifest[section][key] = value
            with self.assertRaisesRegex(ValueError, message):
                load_configuration(env_config, manifest)
            manifest[section][key] = original


if __name__ == "__main__":
    unittest.main()
