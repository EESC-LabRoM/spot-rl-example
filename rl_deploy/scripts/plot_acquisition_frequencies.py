import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse

# allow for absolute imports from 'rl_deploy'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def plot_acquisition_frequencies(datasets: dict, out_dir: Path):
    """Plot the delta t of the acquisition timestamps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, data in datasets.items():
        ts_list = []
        for raw_bytes in data:
            parsed_state = RobotStateStreamResponse()
            parsed_state.ParseFromString(raw_bytes.tobytes())
            sec = parsed_state.joint_states.acquisition_timestamp.seconds
            nanos = parsed_state.joint_states.acquisition_timestamp.nanos
            ts_list.append(sec + nanos * 1e-9)
        
        if len(ts_list) > 1:
            dt = np.diff(np.array(ts_list))
            ax.plot(dt, label=label)
    
    # Add a horizontal line for 50Hz (0.02s)
    ax.axhline(y=1/50.0, color='r', linestyle='--', label='50Hz (0.02s)')
    
    ax.set_title("Acquisition Timestamp Delta T")
    ax.set_ylabel("Delta Time (s)")
    ax.set_xlabel("Time step")
    ax.legend()
    plt.tight_layout()

    out_file = out_dir / "dt_acquisition_timestamp.png"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved plot to {out_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot acquisition frequencies from HDF5 log."
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

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True, parents=True)

    open_files = {}
    datasets = {}
    
    for f_path in valid_files:
        f = h5py.File(f_path, "r")
        open_files[f_path.name] = f
        if "raw_state_proto_bytes" in f:
            datasets[f_path.name] = f["raw_state_proto_bytes"][:]

    if not datasets:
        print("Error: None of the HDF5 files contain 'raw_state_proto_bytes'.")
    else:
        print("Plotting dt_acquisition_timestamp...")
        plot_acquisition_frequencies(datasets, out_dir)

    for f in open_files.values():
        f.close()


if __name__ == "__main__":
    main()
