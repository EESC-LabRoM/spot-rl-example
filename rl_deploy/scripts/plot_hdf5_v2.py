import argparse
import os
import sys
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np

# allow for absolute imports from 'rl_deploy'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rl_deploy.orbit.orbit_constants import (
    ORDERED_JOINT_NAMES_ARM_ISAAC,
    ORDERED_JOINT_NAMES_BASE_ISAAC,
    ORDERED_JOINT_NAMES_ISAAC,
)

def get_y_label(num_dims, dim_idx):
    """Determine the label for a specific dimension."""
    if num_dims == 3:
        return ["X", "Y", "Z"][dim_idx]
    elif num_dims == 4:
        return ["X", "Y", "Z", "W"][dim_idx]
    elif num_dims == 7:
        return ORDERED_JOINT_NAMES_ARM_ISAAC[dim_idx]
    elif num_dims == 12:
        return ORDERED_JOINT_NAMES_BASE_ISAAC[dim_idx]
    elif num_dims == 19:
        return ORDERED_JOINT_NAMES_ISAAC[dim_idx]
    else:
        return f"Dim {dim_idx}"

def plot_datasets(name, file_data_map, out_dir):
    """Plot the same dataset from multiple files on the same graph."""
    name_clean = name.replace("/", "_")
    fig = None
    
    # Get the first available dataset to check shape
    first_data = next(iter(file_data_map.values()))
    
    if len(first_data.shape) == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for filename, data in file_data_map.items():
            ax.plot(data, label=filename)
        ax.set_title(name)
        ax.set_ylabel("Value")
        ax.set_xlabel("Time step")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
    elif len(first_data.shape) == 2:
        num_dims = first_data.shape[1]
        fig, axes = plt.subplots(
            nrows=num_dims, ncols=1, figsize=(10, 2 * num_dims), sharex=True
        )
        if num_dims == 1:
            axes = [axes]
        
        fig.suptitle(name, fontsize=16)
        for i in range(num_dims):
            for filename, data in file_data_map.items():
                if len(data.shape) == 2 and data.shape[1] > i:
                    axes[i].plot(data[:, i], label=filename)
            axes[i].set_ylabel(get_y_label(num_dims, i))
            axes[i].legend(loc="upper right", fontsize='small')
            axes[i].grid(True, linestyle="--", alpha=0.6)
        
        axes[-1].set_xlabel("Time step")
    else:
        print(f"Skipping {name}: unsupported shape {first_data.shape}")
        return

    plt.tight_layout()
    out_file = out_dir / f"{name_clean}.png"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved plot to {out_file}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Recursively plot and compare variables in HDF5 files.")
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        default=[Path("spot_isaac_real_v2.hdf5"), Path("spot_isaac_sim_v2.hdf5")],
        help="Path to the HDF5 files.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("plots"), help="Directory to save plots.")
    args = parser.parse_args()

    valid_files = [f for f in args.files if f.exists()]
    if not valid_files:
        print(f"Error: No valid files found.")
        return

    if len(valid_files) > 1:
        out_name = "comparison_" + "_vs_".join([f.stem for f in valid_files])
    else:
        out_name = valid_files[0].stem
        
    out_dir = args.out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all dataset names from all files
    all_datasets = set()
    file_handles = {}
    
    for f_path in valid_files:
        try:
            h = h5py.File(f_path, "r")
            file_handles[f_path.name] = h
            def collect_names(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_datasets.add(name)
            h.visititems(collect_names)
        except OSError as e:
            print(f"Warning: Could not open {f_path}: {e}. Skipping.")

    # Plot each dataset found in any of the files
    for name in sorted(list(all_datasets)):
        file_data_map = {}
        for f_name, h in file_handles.items():
            if name in h:
                file_data_map[f_name] = h[name][:]
        
        if file_data_map:
            print(f"Plotting {name}...")
            plot_datasets(name, file_data_map, out_dir)

    # Close all handles
    for h in file_handles.values():
        h.close()

if __name__ == "__main__":
    main()
