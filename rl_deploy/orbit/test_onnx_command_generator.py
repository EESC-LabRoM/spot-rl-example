import unittest
from unittest.mock import patch

import numpy as np
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse

from rl_deploy.orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
)
from rl_deploy.orbit.orbit_configuration import OrbitConfig
from rl_deploy.spot.constants import JOINT_LIMITS, ORDERED_JOINT_NAMES_SPOT


class _Node:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _Session:
    def __init__(
        self,
        recurrent=True,
        missing_output=False,
        hidden_shape=None,
        flat_obs_size=None,
        named=False,
        foot_height=False,
    ):
        self.recurrent = recurrent
        self.calls = []
        if flat_obs_size is not None:
            self._inputs = [_Node("obs", [1, flat_obs_size])]
        elif named:
            self._inputs = [
                _Node("base_linear_velocity", [1, 3]),
                _Node("base_angular_velocity", [1, 3]),
                _Node("projected_gravity", [1, 3]),
                _Node("velocity_commands", [1, 3]),
                _Node("joint_positions", [1, 19]),
                _Node("joint_velocities", [1, 19]),
                _Node("last_actions", [1, 12]),
                _Node("height_commands", [1, 1]),
                _Node("base_orientation_commands", [1, 2]),
            ]
            if foot_height:
                self._inputs.append(_Node("foot_height_commands", [1, 4]))
        else:
            self._inputs = [_Node("base_linear_velocity", [1, 3])]
        self._outputs = [_Node("actions", [1, 12])]
        if recurrent:
            self._inputs.append(
                _Node("hidden_state", hidden_shape or [2, "batch_size", 256])
            )
            self._outputs = [_Node("next_hidden_state", [2, "batch_size", 256])]
            if not missing_output:
                self._outputs.append(_Node("actions_output", ["batch_size", 12]))

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _output_names, inputs):
        self.calls.append({key: np.array(value, copy=True) for key, value in inputs.items()})
        if not self.recurrent:
            return [np.full((1, 12), 0.25, dtype=np.float32)]
        outputs = {
            "actions_output": np.full((1, 12), len(self.calls), dtype=np.float32),
            "next_hidden_state": inputs["hidden_state"] + 1.0,
        }
        return [outputs[item.name] for item in self._outputs]


def _config():
    defaults = {
        "fl_hx": 0.12, "fl_hy": 0.5, "fl_kn": -1.0,
        "fr_hx": -0.12, "fr_hy": 0.5, "fr_kn": -1.0,
        "hl_hx": 0.12, "hl_hy": 0.5, "hl_kn": -1.0,
        "hr_hx": -0.12, "hr_hy": 0.5, "hr_kn": -1.0,
        "arm_sh0": 0.0, "arm_sh1": -0.9, "arm_el0": 1.8, "arm_el1": 0.0,
        "arm_wr0": -0.9, "arm_wr1": 0.0, "arm_f1x": -1.54,
    }
    gains = {name: 0.0 for name in ORDERED_JOINT_NAMES_SPOT}
    return OrbitConfig(gains, gains, defaults, 0.65, 0.2)


def _state():
    state = RobotStateStreamResponse()
    state.kinematic_state.odom_tform_body.rotation.w = 1.0
    config = _config()
    state.joint_states.position.extend(
        config.default_joints[name] for name in ORDERED_JOINT_NAMES_SPOT
    )
    state.joint_states.velocity.extend([0.0] * 19)
    state.joint_states.load.extend([0.0] * 19)
    return state


class OnnxCommandGeneratorTest(unittest.TestCase):
    def test_recurrent_state_is_propagated_and_reset(self):
        session = _Session()
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(None, _config(), "policy.onnx", False)

        expected_last_action = [
            _config().default_joints[name] for name in ORDERED_JOINT_NAMES_SPOT[:12]
        ]
        np.testing.assert_allclose(generator._last_action, expected_last_action)
        np.testing.assert_array_equal(generator._hidden_state, np.zeros((2, 1, 256)))

        inputs = {"base_linear_velocity": np.zeros((1, 3), dtype=np.float32)}
        np.testing.assert_array_equal(generator._compute_action(inputs), np.ones(12))
        np.testing.assert_array_equal(generator._compute_action(inputs), np.full(12, 2.0))
        np.testing.assert_array_equal(session.calls[0]["hidden_state"], np.zeros((2, 1, 256)))
        np.testing.assert_array_equal(session.calls[1]["hidden_state"], np.ones((2, 1, 256)))

        generator.reset_policy_state()
        np.testing.assert_array_equal(generator._hidden_state, np.zeros((2, 1, 256)))
        np.testing.assert_allclose(generator._last_action, expected_last_action)

    def test_stateless_policy_behavior_is_unchanged(self):
        session = _Session(recurrent=False)
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(None, _config(), "policy.onnx", False)

        self.assertIsNone(generator._hidden_state)
        np.testing.assert_allclose(
            generator._last_action,
            [_config().default_joints[name] for name in ORDERED_JOINT_NAMES_SPOT[:12]],
        )
        inputs = {"base_linear_velocity": np.zeros((1, 3), dtype=np.float32)}
        np.testing.assert_array_equal(generator._compute_action(inputs), np.full(12, 0.25))
        self.assertNotIn("hidden_state", session.calls[0])

    def test_flat_65_policy_receives_training_observation_and_rescales_action(self):
        session = _Session(recurrent=False, flat_obs_size=65)
        context = OnnxControllerContext()
        context.latest_state = _state()
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(context, _config(), "policy.onnx", False)

        inputs = generator.collect_inputs(context.latest_state, generator._config)
        self.assertEqual(set(inputs), {"obs"})
        self.assertEqual(inputs["obs"].shape, (1, 65))
        np.testing.assert_allclose(inputs["obs"][0, 12:31], 0.0)
        np.testing.assert_allclose(inputs["obs"][0, 50:62], 0.0)
        np.testing.assert_allclose(inputs["obs"][0, 62], 0.6)
        np.testing.assert_allclose(inputs["obs"][0, 63:65], 0.0)
        output = generator._compute_action(inputs)
        action = generator._deployment_action(output)
        expected_legs = np.array(generator.base_offsets_ordered_spot) + 0.05
        np.testing.assert_allclose(action[:12], expected_legs)

    def test_named_policy_receives_in_distribution_height_command(self):
        session = _Session(recurrent=False, named=True)
        context = OnnxControllerContext()
        context.latest_state = _state()
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(context, _config(), "policy.onnx", False)

        inputs = generator.collect_inputs(context.latest_state, generator._config)
        np.testing.assert_allclose(inputs["height_commands"], [[0.6]])
        np.testing.assert_allclose(inputs["base_orientation_commands"], [[0.0, 0.0]])

        custom = _config()
        custom.height_command = 0.5
        inputs = generator.collect_inputs(context.latest_state, custom)
        np.testing.assert_allclose(inputs["height_commands"], [[0.5]])

    def test_flat_84_policy_includes_joint_command_block(self):
        session = _Session(recurrent=False, flat_obs_size=84)
        context = OnnxControllerContext()
        context.latest_state = _state()
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(context, _config(), "policy.onnx", False)

        inputs = generator.collect_inputs(context.latest_state, generator._config)
        self.assertEqual(inputs["obs"].shape, (1, 84))
        self.assertEqual(inputs["obs"][0, 12:34].shape, (22,))
        np.testing.assert_allclose(inputs["obs"][0, 34:53], 0.0)
        np.testing.assert_allclose(inputs["obs"][0, 72:84], 0.0)

    def test_named_policy_receives_gait_foot_height_command(self):
        session = _Session(recurrent=False, named=True, foot_height=True)
        context = OnnxControllerContext()
        context.latest_state = _state()
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(context, _config(), "policy.onnx", False)

        inputs = generator.collect_inputs(context.latest_state, generator._config)
        self.assertEqual(inputs["foot_height_commands"].shape, (1, 4))
        np.testing.assert_allclose(inputs["foot_height_commands"], np.full((1, 4), 0.036))

        context.velocity_cmd = [0.5, 0.0, 0.0]
        generator._gait_phase = 0.125
        inputs = generator.collect_inputs(context.latest_state, generator._config)
        np.testing.assert_allclose(
            inputs["foot_height_commands"], [[0.186, 0.036, 0.036, 0.036]], atol=1e-6
        )

        generator._advance_gait_phase(generator._config)
        self.assertAlmostEqual(generator._gait_phase, 0.155)
        generator.reset_policy_state()
        self.assertEqual(generator._gait_phase, 0.0)

    def test_gait_command_is_suppressed_below_standing_threshold(self):
        session = _Session(recurrent=False, named=True, foot_height=True)
        context = OnnxControllerContext()
        context.latest_state = _state()
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(context, _config(), "policy.onnx", False)

        generator._gait_phase = 0.125
        context.velocity_cmd = [0.049, 0.0, 0.0]
        inputs = generator.collect_inputs(context.latest_state, generator._config)
        np.testing.assert_allclose(inputs["foot_height_commands"], np.full((1, 4), 0.036))

    def test_deployment_action_clamps_leg_targets_and_preserves_arm_stow(self):
        session = _Session(recurrent=False, named=True)
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(None, _config(), "policy.onnx", False)

        action = generator._deployment_action([100.0] * 6 + [-100.0] * 6)
        self.assertEqual(len(action), 19)
        for name, value in zip(ORDERED_JOINT_NAMES_SPOT[:12], action[:12]):
            self.assertGreaterEqual(value, JOINT_LIMITS[name]["lower"])
            self.assertLessEqual(value, JOINT_LIMITS[name]["upper"])
        self.assertEqual(action[12:], generator.arm_offsets_ordered)

    def test_deployment_action_rejects_invalid_output(self):
        session = _Session(recurrent=False, named=True)
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            generator = OnnxCommandGenerator(None, _config(), "policy.onnx", False)

        with self.assertRaisesRegex(ValueError, "shape"):
            generator._deployment_action([0.0] * 11)
        for value in (np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                generator._deployment_action([value] + [0.0] * 11)

    def test_recurrent_policy_requires_named_outputs(self):
        session = _Session(missing_output=True)
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            with self.assertRaisesRegex(ValueError, "actions_output"):
                OnnxCommandGenerator(None, _config(), "policy.onnx", False)

    def test_recurrent_policy_requires_concrete_state_dimensions(self):
        session = _Session(hidden_shape=["layers", "batch_size", 256])
        with patch(
            "rl_deploy.orbit.onnx_command_generator.ort.InferenceSession",
            return_value=session,
        ):
            with self.assertRaisesRegex(ValueError, "hidden_state must have shape"):
                OnnxCommandGenerator(None, _config(), "policy.onnx", False)


if __name__ == "__main__":
    unittest.main()
