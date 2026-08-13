# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import atexit
import signal
import sys
from pathlib import Path

import bosdyn.client.util
import orbit.orbit_configuration
from rl_deploy.hid.terminal_keyboard import TerminalKeyboard
from rl_deploy.orbit import arm_motion
from rl_deploy.orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from rl_deploy.spot.mock_spot import MockSpot
from rl_deploy.spot.spot import Spot
from rl_deploy.utils.event_divider import EventDivider
from rl_deploy.utils.hdf5_logger import HDF5Logger

from datetime import datetime


def _safe_save_hdf5(logger: HDF5Logger, reason: str):
    try:
        logger.save(reason=reason)
    except Exception as exc:
        print(f"Failed to save HDF5 log during {reason}: {exc!r}")


def _run_cleanup_step(description: str, cleanup_func):
    print(description)
    try:
        cleanup_func()
    except KeyboardInterrupt:
        print(f"Interrupted during {description}; continuing cleanup.")
    except Exception as exc:
        print(f"Failed during {description}: {exc!r}")


def _register_emergency_hdf5_saves(logger: HDF5Logger):
    def save_on_exit():
        _safe_save_hdf5(logger, "process exit")

    def save_on_signal(signum, frame):
        if logger.is_save_in_progress_on_current_thread():
            print(f"HDF5 save already in progress; ignoring signal {signum}.")
            return
        print(f"Received signal {signum}; saving during shutdown.")
        raise KeyboardInterrupt

    atexit.register(save_on_exit)
    signal.signal(signal.SIGINT, save_on_signal)
    signal.signal(signal.SIGTERM, save_on_signal)


def main():
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    configs_dir = Path(__file__).resolve().parent / "configs"
    orbit.orbit_configuration.add_policy_bundle_argument(
        parser, configs_dir
    )
    parser.add_argument("-m", "--mock", action="store_true")
    parser.add_argument(
        "--hdf5_log",
        type=str,
        default=f"spot_isaac_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5",
        help="Path to save HDF5 log of observations.",
    )
    parser.add_argument(
        "--run_metadata",
        type=Path,
        default=None,
        help="Optional JSON sidecar with run metadata to store as HDF5 attributes.",
    )
    options = parser.parse_args()

    # Resolve the full runtime contract and instantiate ONNX before any robot
    # connection, lease, power-on, or command-stream operation.
    bundle = orbit.orbit_configuration.resolve_policy_bundle(
        options.policy_dir, configs_dir
    )
    print("Loaded policy bundle: ", bundle.directory)
    print("Loaded configs: ", bundle.config)
    arm = arm_motion.from_manifest(bundle.manifest, options.arm)

    context = OnnxControllerContext()
    state_handler = StateHandler(context)
    print("Verbose option: ", options.verbose)

    metadata_path = options.run_metadata
    if metadata_path is None:
        default_metadata_path = Path(options.hdf5_log).with_suffix(".metadata.json")
        metadata_path = default_metadata_path if default_metadata_path.exists() else None

    logger = HDF5Logger(options.hdf5_log, metadata_path=metadata_path)
    _register_emergency_hdf5_saves(logger)
    command_generator = OnnxCommandGenerator(
        context, bundle.config, bundle.policy_file, options.verbose, logger=logger,
        arm_motion=arm,
    )
    gamepad = TerminalKeyboard(context)
    timeing_policy = EventDivider(context, bundle.config.control_period_s)

    if options.mock:
        spot = MockSpot()
    else:
        print("Connecting Spot")
        spot = Spot(options)
        print("OK")

    with spot.lease_keep_alive():
        try:
            print("Powering on and standing")
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
            _safe_save_hdf5(logger, "before stream shutdown")
            _run_cleanup_step("stop command stream", spot.stop_command_stream)
            _run_cleanup_step("stop state stream", spot.stop_state_stream)
            print("stop game pad")
            _safe_save_hdf5(logger, "after stream shutdown")


if __name__ == "__main__":
    if not main():
        sys.exit(1)
