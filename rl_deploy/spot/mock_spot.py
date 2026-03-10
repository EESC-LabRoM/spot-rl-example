# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import math
import time
from contextlib import nullcontext
from threading import Thread
from typing import Any, Callable, List

from bosdyn.api.robot_command_pb2 import JointControlStreamRequest
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse


class RepeatedTimer(Thread):
    def __init__(self, dt_seconds: float, target: Callable, args: List[Any] = []) -> None:
        super().__init__()
        self._dt_seconds = dt_seconds
        self._target = target
        self._args = args
        self._stopping = False

    def run(self):
        run_time = time.monotonic()
        while not self._stopping:
            now = time.monotonic()
            num_dt = math.ceil((now - run_time) / self._dt_seconds)
            run_time += num_dt * self._dt_seconds
            time.sleep(run_time - now)
            self._target(*self._args)

    def stop(self):
        self._stopping = True


class MockSpot:
    _command_thread = None
    _state_stream_stopping = False
    _command_stream_stopping = False

    def __init__(self):
        ... 

    def start_state_stream(self, on_state_update: Callable[[RobotStateStreamResponse], None]):
        self._state_msg = RobotStateStreamResponse()
        self._state_msg.kinematic_state.odom_tform_body.rotation.w = 1.0
        # Default positions corresponding to ORDERED_JOINT_NAMES_SPOT in constants.py
        # Legs: fl_hx, fl_hy, fl_kn, fr_hx, fr_hy, fr_kn, hl_hx, hl_hy, hl_kn, hr_hx, hr_hy, hr_kn
        # Arm: arm_sh0, arm_sh1, arm_el0, arm_el1, arm_wr0, arm_wr1, arm_f1x
        default_positions = [
            0.1, 0.9, -1.5,
            -0.1, 0.9, -1.5,
            0.1, 1.1, -1.5,
            -0.1, 1.1, -1.5,
            0.0, -3.1415, 3.1415, 1.5655, 0.0, 1.5692, 0.0
        ]
        self._state_msg.joint_states.position.extend(default_positions)
        self._state_msg.joint_states.velocity.extend([0.0] * 19)
        self._state_msg.joint_states.load.extend([0.0] * 19)

        self._stateUpdates = RepeatedTimer(1 / 333, on_state_update, args=[self._state_msg])
        self._stateUpdates.start()

    def start_command_stream(
        self, command_policy: Callable[[None], JointControlStreamRequest], timing_policy: Callable[[None], None]
    ):
        self._timing_policy = timing_policy
        self._command_generator = command_policy

        self._command_thread = Thread(target=self._commandUpdate)
        self._command_thread.start()

    def lease_keep_alive(self):
        return nullcontext()

    def _commandUpdate(self):
        while not self._command_stream_stopping:
            self._timing_policy()
            self._command_generator()

    def power_on(self):
        pass

    def stand(self, body_height: float):
        pass

    def stop_state_stream(self):
        if self._stateUpdates is not None:
            self._stateUpdates.stop()
            self._stateUpdates.join()

    def stop_command_stream(self):
        if self._command_thread is not None:
            self._command_stream_stopping = True
            self._command_thread.join()
