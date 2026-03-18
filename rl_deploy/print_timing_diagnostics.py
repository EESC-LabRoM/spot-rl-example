"""Print average timing diagnostics from HDF5 robot logs.

Parses state and command protobuf data, computes timing metrics,
and validates command key monotonicity and state-command cross-references.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from bosdyn.api import robot_command_pb2
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse

# allow for absolute imports from 'rl_deploy'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys.path[0] != project_root:
    sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingMetric:
    """Describes an expected timing metric with its key and human-readable condition."""
    key: str
    expected_value: float
    expected_condition: str


TIMING_METRICS: list[TimingMetric] = [
    TimingMetric("dt_acquisition_timestamp", 0.02,  "approx 0.02"),
    TimingMetric("dt_total_step",            0.02,  "approx 0.02"),
    TimingMetric("dt_divider_wait",          0.02,  "< 0.02 (varies)"),
    TimingMetric("dt_divider_to_onnx",       0.001, "< 0.001"),
    TimingMetric("dt_state_arrival_to_compute", 0.002, "< 0.002"),
    TimingMetric("dt_onnx_compute",          0.005, "< 0.005"),
    TimingMetric("dt_post_process",          0.001, "< 0.001"),
    TimingMetric("dt_command_latency",       0.005, "< 0.010"),
]

EXPECTED_CONDITIONS: dict[str, str] = {m.key: m.expected_condition for m in TIMING_METRICS}

STANDARD_DT_KEYS: list[str] = [
    "dt_total_step",
    "dt_divider_wait",
    "dt_divider_to_onnx",
    "dt_state_arrival_to_compute",
    "dt_onnx_compute",
    "dt_post_process",
]

MAX_DIAGNOSTIC_EXAMPLES = 10

# ---------------------------------------------------------------------------
# Proto helpers
# ---------------------------------------------------------------------------

def _timestamp_to_seconds(ts: Timestamp) -> float:
    """Convert a protobuf Timestamp to seconds as a float."""
    return ts.seconds + ts.nanos * 1e-9


def _parse_state_protos(
    dataset: h5py.File,
) -> tuple[list[RobotStateStreamResponse], np.ndarray]:
    """Deserialize raw_state_proto_bytes into parsed states and acquisition times."""
    parsed_states: list[RobotStateStreamResponse] = []
    acquisition_times: list[float] = []

    for raw_bytes in dataset["raw_state_proto_bytes"]:
        state = RobotStateStreamResponse()
        state.ParseFromString(raw_bytes.tobytes())
        acquisition_times.append(
            _timestamp_to_seconds(state.joint_states.acquisition_timestamp)
        )
        parsed_states.append(state)

    return parsed_states, np.array(acquisition_times)


def _parse_command_protos(
    dataset: h5py.File,
    indices: np.ndarray,
) -> np.ndarray:
    """Parse proto_bytes entries and return user_command_keys as int64 array."""
    cmd_bytes_list = dataset["proto_bytes"][indices]
    user_keys: list[int] = []

    for cmd_bytes in cmd_bytes_list:
        parsed = robot_command_pb2.JointControlStreamRequest()
        parsed.ParseFromString(cmd_bytes.tobytes())
        user_keys.append(parsed.joint_command.user_command_key)

    return np.array(user_keys, dtype=np.int64)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _compute_valid_indices(
    acquisition_times: np.ndarray,
    margin_seconds: float = 1.0,
) -> np.ndarray:
    """Return indices of samples inside [t_start + margin, t_end - margin].

    Falls back to all indices if the dataset is too short for the given margin.
    """
    t_min = acquisition_times[0] + margin_seconds
    t_max = acquisition_times[-1] - margin_seconds

    if t_min >= t_max:
        print("Warning: Dataset is too short to skip 1s at start and end. Using all data.")
        return np.arange(len(acquisition_times))

    return np.where(
        (acquisition_times >= t_min) & (acquisition_times <= t_max)
    )[0]


# ---------------------------------------------------------------------------
# Metric computation (pure data — no printing)
# ---------------------------------------------------------------------------

@dataclass
class ProtoTimingResult:
    """Aggregated results from parsing state protos over valid indices."""
    avg_dt_acquisition: float | None
    avg_dt_command_latency: float | None
    state_user_command_keys: list[int]


def _compute_proto_timing_metrics(
    parsed_states: list[RobotStateStreamResponse],
    acquisition_times: np.ndarray,
    valid_indices: np.ndarray,
) -> ProtoTimingResult:
    """Compute dt_acquisition_timestamp and dt_command_latency from state protos."""
    state_user_command_keys: list[int] = []
    valid_timestamps: list[float] = []
    command_latencies: list[float] = []
    prev_request_received_time: float | None = None

    for idx in valid_indices:
        state = parsed_states[idx]
        valid_timestamps.append(acquisition_times[idx])
        state_user_command_keys.append(state.last_command.user_command_key)

        cmd_received_time = _timestamp_to_seconds(state.last_command.received_timestamp)
        request_received_time = _timestamp_to_seconds(state.header.request_received_timestamp)

        if (
            cmd_received_time > 0
            and prev_request_received_time is not None
            and prev_request_received_time > 0
        ):
            command_latencies.append(cmd_received_time - prev_request_received_time)

        prev_request_received_time = request_received_time

    avg_dt_acq = float(np.mean(np.diff(valid_timestamps))) if len(valid_timestamps) > 1 else None
    avg_dt_cmd = float(np.mean(command_latencies)) if command_latencies else None

    return ProtoTimingResult(
        avg_dt_acquisition=avg_dt_acq,
        avg_dt_command_latency=avg_dt_cmd,
        state_user_command_keys=state_user_command_keys,
    )


# ---------------------------------------------------------------------------
# Formatted printing helpers
# ---------------------------------------------------------------------------

def _print_metric_row(name: str, value: float | None, expected: str) -> None:
    """Print a single metric row with consistent column formatting."""
    formatted_value = f"{value:<15.6f}" if value is not None else f"{'N/A':<15}"
    print(f"{name:<30} | {formatted_value} | {expected:<15}")


def _print_metric_row_missing(name: str, expected: str) -> None:
    """Print a metric row for a dataset key that is missing."""
    print(f"{name:<30} | {'N/A (Missing)':<15} | {expected:<15}")


def _print_table_header(n_total: int, n_valid: int) -> None:
    """Print the summary header and column titles for the timing table."""
    print(f"Total samples: {n_total}, Valid samples used for average: {n_valid}")
    print(f"{'Variable Name':<30} | {'Average (s)':<15} | {'Expected (s)':<15}")
    print("-" * 65)


# ---------------------------------------------------------------------------
# Command key diagnostics
# ---------------------------------------------------------------------------

def _report_key_issues(
    keys: np.ndarray,
    issue_indices: np.ndarray,
    label: str,
) -> None:
    """Print detailed report for a set of problematic key indices."""
    diffs = np.diff(keys)
    print(f"  [FAIL] user_command_key {label} at {len(issue_indices)} position(s):")

    for idx in issue_indices[:MAX_DIAGNOSTIC_EXAMPLES]:
        print(
            f"    index {idx}: key[{idx}]={keys[idx]} -> "
            f"key[{idx + 1}]={keys[idx + 1]}  (diff={diffs[idx]})"
        )

    remaining = len(issue_indices) - MAX_DIAGNOSTIC_EXAMPLES
    if remaining > 0:
        print(f"    ... and {remaining} more")


def _check_monotonicity(cmd_user_keys: np.ndarray) -> None:
    """Verify command keys are strictly increasing by 1 with no skips."""
    diffs = np.diff(cmd_user_keys)
    non_monotonic = np.where(diffs <= 0)[0]
    skips = np.where(diffs > 1)[0]

    if len(non_monotonic) == 0 and len(skips) == 0:
        print("  [OK] user_command_key is strictly monotonic with no skips (increment = 1 everywhere)")
        return

    if len(non_monotonic) > 0:
        _report_key_issues(cmd_user_keys, non_monotonic, "is NOT monotonic")
    if len(skips) > 0:
        _report_key_issues(cmd_user_keys, skips, f"has {len(skips)} skip(s) (diff > 1)")


def _check_state_command_cross_reference(
    state_command_keys: list[int],
    cmd_user_keys: np.ndarray,
) -> None:
    """Verify state[i].last_command.user_command_key == cmd[i-1].user_command_key."""
    if not state_command_keys:
        print("  [SKIP] No state proto data available for cross-check.")
        return

    state_keys = np.array(state_command_keys, dtype=np.int64)
    n = min(len(state_keys), len(cmd_user_keys))
    state_ack = state_keys[1:n]
    cmd_sent = cmd_user_keys[:n - 1]

    mismatches = np.where(state_ack != cmd_sent)[0]

    if len(mismatches) == 0:
        print(f"  [OK] state.last_command.user_command_key matches previous cmd key for all {len(state_ack)} pairs")
        return

    print(
        f"  [FAIL] {len(mismatches)} mismatch(es) between "
        f"state.last_command.user_command_key and previous cmd key:"
    )
    for idx in mismatches[:MAX_DIAGNOSTIC_EXAMPLES]:
        step = idx + 1
        print(
            f"    step {step}: state.last_command.user_command_key={state_ack[idx]}, "
            f"expected (cmd[{step - 1}].user_command_key)={cmd_sent[idx]}"
        )

    remaining = len(mismatches) - MAX_DIAGNOSTIC_EXAMPLES
    if remaining > 0:
        print(f"    ... and {remaining} more")


def _print_command_key_diagnostics(
    dataset: h5py.File,
    valid_indices: np.ndarray,
    state_command_keys: list[int],
    label: str,
) -> None:
    """Parse command protos and run monotonicity / cross-reference checks."""
    print(f"\n--- Command Key Diagnostics for {label} ---")

    if "proto_bytes" not in dataset:
        print("Warning: 'proto_bytes' not found in dataset. Skipping command key checks.")
        return

    cmd_user_keys = _parse_command_protos(dataset, valid_indices)
    print(
        f"Command key range: {cmd_user_keys[0]} -> {cmd_user_keys[-1]}  "
        f"(total {len(cmd_user_keys)} samples)"
    )

    _check_monotonicity(cmd_user_keys)
    _check_state_command_cross_reference(state_command_keys, cmd_user_keys)


# ---------------------------------------------------------------------------
# Standard HDF5 dt metrics
# ---------------------------------------------------------------------------

def _print_standard_dt_metrics(
    dataset: h5py.File,
    valid_indices: np.ndarray,
) -> None:
    """Print average values for the standard dt_* keys stored in the HDF5 dataset."""
    for key in STANDARD_DT_KEYS:
        expected = EXPECTED_CONDITIONS.get(key, "N/A")
        if key in dataset:
            avg_value = float(np.mean(dataset[key][valid_indices]))
            _print_metric_row(key, avg_value, expected)
        else:
            _print_metric_row_missing(key, expected)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def print_timing_diagnostics(dataset: h5py.File, label: str) -> None:
    """Print complete timing and command-key diagnostics for an HDF5 log."""
    print(f"\n--- Timing Diagnostics for {label} ---")

    if "raw_state_proto_bytes" not in dataset:
        print("Error: 'raw_state_proto_bytes' not found in dataset. Cannot compute diagnostics.")
        return

    n_total = len(dataset["raw_state_proto_bytes"])
    if n_total == 0:
        print("Error: Empty dataset.")
        return

    parsed_states, acquisition_times = _parse_state_protos(dataset)
    valid_indices = _compute_valid_indices(acquisition_times)

    if len(valid_indices) == 0:
        print("Error: No valid data points found after filtering.")
        return

    _print_table_header(n_total, len(valid_indices))

    proto_metrics = _compute_proto_timing_metrics(parsed_states, acquisition_times, valid_indices)

    _print_metric_row(
        "dt_acquisition_timestamp",
        proto_metrics.avg_dt_acquisition,
        EXPECTED_CONDITIONS["dt_acquisition_timestamp"],
    )
    _print_metric_row(
        "dt_command_latency",
        proto_metrics.avg_dt_command_latency,
        EXPECTED_CONDITIONS["dt_command_latency"],
    )

    _print_standard_dt_metrics(dataset, valid_indices)

    _print_command_key_diagnostics(
        dataset, valid_indices, proto_metrics.state_user_command_keys, label
    )


def main() -> None:
    """CLI entry point: parse args and run diagnostics on each HDF5 file."""
    parser = argparse.ArgumentParser(
        description="Print average timing diagnostics from an HDF5 log, skipping the first and last 1s."
    )
    parser.add_argument(
        "--hdf5_files",
        type=Path,
        nargs="+",
        default=[Path("spot_isaac_sim.hdf5"), Path("spot_isaac_real.hdf5")],
        help="Paths to the HDF5 log files.",
    )
    args = parser.parse_args()

    valid_files = [f for f in args.hdf5_files if f.exists()]
    if not valid_files:
        print("Error: None of the specified HDF5 files exist.")
        return

    for file_path in valid_files:
        with h5py.File(file_path, "r") as hdf5_file:
            print_timing_diagnostics(hdf5_file, file_path.name)


if __name__ == "__main__":
    main()
