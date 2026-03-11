import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse

# allow for absolute imports from 'rl_deploy'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys.path[0] != project_root:
    sys.path.insert(0, project_root)

EXPECTED_VALUES = {
    "dt_acquisition_timestamp": 0.02,
    "dt_total_step": 0.02,
    "dt_divider_wait": 0.02,
    "dt_divider_to_onnx": 0.001,
    "dt_state_arrival_to_compute": 0.002,
    "dt_onnx_compute": 0.005,
    "dt_post_process": 0.001,
    "dt_command_latency": 0.005,
}

EXPECTED_CONDITIONS = {
    "dt_acquisition_timestamp": "approx 0.02",
    "dt_total_step": "approx 0.02",
    "dt_divider_wait": "< 0.02 (varies)",
    "dt_divider_to_onnx": "< 0.001",
    "dt_state_arrival_to_compute": "< 0.002",
    "dt_onnx_compute": "< 0.005",
    "dt_post_process": "< 0.001",
    "dt_command_latency": "< 0.010",
}


def print_timing_diagnostics(dataset: h5py.File, label: str):
    print(f"\n--- Timing Diagnostics for {label} ---")
    
    if "response_timestamp" not in dataset:
        print("Error: 'response_timestamp' not found in dataset. Cannot filter by time.")
        return

    # Extract relative timestamps
    response_times = dataset["response_timestamp"][:]
    if len(response_times) == 0:
        print("Error: Empty dataset.")
        return

    # Filter bounds
    t_min = response_times[0] + 1.0
    t_max = response_times[-1] - 1.0

    if t_min >= t_max:
        print("Warning: Dataset is too short to skip 1s at start and end. Using all data.")
        valid_indices = np.arange(len(response_times))
    else:
        valid_indices = np.where((response_times >= t_min) & (response_times <= t_max))[0]

    if len(valid_indices) == 0:
        print("Error: No valid data points found after filtering.")
        return

    print(f"Total samples: {len(response_times)}, Valid samples used for average: {len(valid_indices)}")
    print(f"{'Variable Name':<30} | {'Average (s)':<15} | {'Expected (s)':<15}")
    print("-" * 65)

    # 1. Calculate dt_acquisition_timestamp
    if "raw_state_proto_bytes" in dataset:
        raw_bytes_list = dataset["raw_state_proto_bytes"][valid_indices]
        valid_ts_list = []
        dt_cmd_lat_list = []
        prev_t_req_received = None
        for raw_bytes in raw_bytes_list:
            parsed_state = RobotStateStreamResponse()
            parsed_state.ParseFromString(raw_bytes.tobytes())
            sec = parsed_state.joint_states.acquisition_timestamp.seconds
            nanos = parsed_state.joint_states.acquisition_timestamp.nanos
            valid_ts_list.append(sec + nanos * 1e-9)
            
            # 2. Calculate dt_command_latency
            # Shifted calculation: (last_command.received_timestamp[i]) - (header.request_received_timestamp[i-1])
            # This measures the time from the robot receiving the previous state request
            # to the robot receiving the command generated from that state.
            
            t_cmd_received = parsed_state.last_command.received_timestamp.seconds + \
                             parsed_state.last_command.received_timestamp.nanos * 1e-9
            t_req_received = parsed_state.header.request_received_timestamp.seconds + \
                             parsed_state.header.request_received_timestamp.nanos * 1e-9
            
            if t_cmd_received > 0 and prev_t_req_received is not None and prev_t_req_received > 0:
                dt_cmd_lat_list.append(t_cmd_received - prev_t_req_received)
            
            prev_t_req_received = t_req_received
        
        if len(valid_ts_list) > 1:
            dt_acq = np.diff(np.array(valid_ts_list))
            avg_dt_acq = np.mean(dt_acq)
            print(f"{'dt_acquisition_timestamp':<30} | {avg_dt_acq:<15.6f} | {EXPECTED_CONDITIONS['dt_acquisition_timestamp']:<15}")
        else:
            print(f"{'dt_acquisition_timestamp':<30} | {'N/A':<15} | {EXPECTED_CONDITIONS['dt_acquisition_timestamp']:<15}")

        if len(dt_cmd_lat_list) > 0:
            avg_dt_cmd = np.mean(dt_cmd_lat_list)
            print(f"{'dt_command_latency':<30} | {avg_dt_cmd:<15.6f} | {EXPECTED_CONDITIONS['dt_command_latency']:<15}")
        else:
            print(f"{'dt_command_latency':<30} | {'N/A':<15} | {EXPECTED_CONDITIONS['dt_command_latency']:<15}")
    else:
         print(f"{'dt_acquisition_timestamp':<30} | {'N/A (Missing)':<15} | {EXPECTED_CONDITIONS['dt_acquisition_timestamp']:<15}")
         print(f"{'dt_command_latency':<30} | {'N/A (Missing)':<15} | {EXPECTED_CONDITIONS['dt_command_latency']:<15}")

    # 2. Iterate standard datasets
    dt_keys = [
        "dt_total_step",
        "dt_divider_wait",
        "dt_divider_to_onnx",
        "dt_state_arrival_to_compute",
        "dt_onnx_compute",
        "dt_post_process"
    ]

    for key in dt_keys:
        if key in dataset:
            vals = dataset[key][valid_indices]
            avg_val = np.mean(vals)
            expected_condition = EXPECTED_CONDITIONS.get(key, "N/A")
            print(f"{key:<30} | {avg_val:<15.6f} | {expected_condition:<15}")
        else:
            print(f"{key:<30} | {'N/A (Missing)':<15} | {EXPECTED_CONDITIONS.get(key, 'N/A'):<15}")

def main():
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

    for f_path in valid_files:
        with h5py.File(f_path, "r") as f:
            print_timing_diagnostics(f, f_path.name)

if __name__ == "__main__":
    main()
