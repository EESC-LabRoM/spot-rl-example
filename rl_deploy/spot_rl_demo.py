# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import sys
from pathlib import Path

import bosdyn.client.util
import orbit.orbit_configuration
from hid.keyboard import Keyboard
from orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from spot.mock_spot import MockSpot
from spot.spot import Spot
from utils.event_divider import EventDivider


def main():
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("policy_file_path", type=Path)
    parser.add_argument("-m", "--mock", action="store_true")
    options = parser.parse_args()
    
    env_config = orbit.orbit_configuration.detect_config_file(options.policy_file_path)
    policy_file = orbit.orbit_configuration.detect_policy_file(options.policy_file_path)

    config = orbit.orbit_configuration.load_configuration(env_config)
    print("Loaded configs: ", config)

    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    print("Verbose option: ", options.verbose)
    
    command_generator = OnnxCommandGenerator(context, config, policy_file, options.verbose)
    gamepad = Keyboard(context)
    # 333 Hz state update / 6 => ~56 Hz control updates
    timeing_policy = EventDivider(context.event, 6)

    if options.mock:
        spot = MockSpot()
    else:
        spot = Spot(options)

    with spot.lease_keep_alive():
        try:
            spot.power_on()
            spot.stand(0.0)
            print("start state stream")
            spot.start_state_stream(state_handler)

            # input(" OK To enter loop")
            print("start command stream")
            spot.start_command_stream(command_generator, timeing_policy)
            gamepad.listen()

        except KeyboardInterrupt:
            print("killed with ctrl-c")
        finally:
            print("stop command stream")
            spot.stop_command_stream()
            print("stop state stream")
            spot.stop_state_stream()
            print("stop game pad")


if __name__ == "__main__":
    if not main():
        sys.exit(1)
