import unittest
from unittest.mock import patch

from rl_deploy.utils.event_divider import EventDivider


class _Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Event:
    def __init__(self, available=True):
        self.available = available

    def wait(self, _timeout):
        return self.available

    def clear(self):
        pass


class _Context:
    def __init__(self, available=True):
        self.event = _Event(available)
        self.timing_dict = {}


class EventDividerTest(unittest.TestCase):
    def test_limits_commands_to_period_without_catch_up_bursts(self):
        clock = _Clock()
        divider = EventDivider(_Context(), 0.02)
        with (
            patch("rl_deploy.utils.event_divider.time.perf_counter", clock.perf_counter),
            patch("rl_deploy.utils.event_divider.time.sleep", clock.sleep),
        ):
            self.assertTrue(divider())
            self.assertEqual(clock.now, 0.0)
            clock.now += 0.003
            self.assertTrue(divider())
            self.assertAlmostEqual(clock.now, 0.02)
            clock.now += 0.025
            self.assertTrue(divider())
            self.assertAlmostEqual(clock.now, 0.045)
            clock.now += 0.001
            self.assertTrue(divider())
            self.assertAlmostEqual(clock.now, 0.065)

    def test_state_timeout_stops_stream(self):
        divider = EventDivider(_Context(available=False), 0.02)
        self.assertFalse(divider())


if __name__ == "__main__":
    unittest.main()
