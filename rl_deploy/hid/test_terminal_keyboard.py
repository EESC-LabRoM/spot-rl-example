import unittest

from rl_deploy.hid.terminal_keyboard import TerminalKeyboardConfig


class TerminalKeyboardConfigTest(unittest.TestCase):
    def test_defaults_match_faster_training_envelope(self):
        config = TerminalKeyboardConfig()
        self.assertEqual(config.delta_forward_velocity, 0.1)
        self.assertEqual(config.delta_lateral_velocity, 0.1)
        self.assertEqual(config.delta_yaw_velocity, 0.1)
        self.assertEqual(config.max_forward_velocity, 0.55)
        self.assertEqual(config.max_backward_velocity, 0.55)
        self.assertEqual(config.max_lateral_velocity, 0.45)
        self.assertEqual(config.max_yaw_velocity, 0.55)


if __name__ == "__main__":
    unittest.main()
