import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rl_deploy.orbit.orbit_configuration import (
    add_policy_bundle_argument,
    detect_policy_file,
    load_policy_manifest,
    resolve_policy_bundle,
)


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

    def test_loads_versioned_manifest_for_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "policy.onnx"
            model.touch()
            (root / "policy.yaml").write_text(
                yaml.safe_dump(
                    {
                        "contract_version": 1,
                        "model": {"file": model.name},
                    }
                )
            )

            selected = detect_policy_file(root)
            manifest = load_policy_manifest(root, selected)

            self.assertEqual(selected, str(model))
            self.assertEqual(manifest["contract_version"], 1)

    def test_rejects_manifest_model_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "actual.onnx"
            model.touch()
            (root / "policy.yaml").write_text(
                yaml.safe_dump(
                    {
                        "contract_version": 1,
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
        manifest = {"contract_version": 1}
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


if __name__ == "__main__":
    unittest.main()
