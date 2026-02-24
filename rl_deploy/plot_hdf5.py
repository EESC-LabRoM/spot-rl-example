import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

# allow for absolute imports from 'rl_deploy'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rl_deploy.orbit.orbit_constants import (
    ORDERED_JOINT_NAMES_ARM_ISAAC,
    ORDERED_JOINT_NAMES_BASE_ISAAC,
    ORDERED_JOINT_NAMES_ISAAC,
)


def plot_dataset(name: str, datasets: dict, out_dir: Path):
    """Plot a single dataset from the HDF5 file."""
    first_data = next(iter(datasets.values()))
    if len(first_data.shape) == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, data in datasets.items():
            ax.plot(data, label=label)
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend()
    elif len(first_data.shape) == 2:
        num_dims = first_data.shape[1]
        fig, axes = plt.subplots(nrows=num_dims, ncols=1, figsize=(10, 2 * num_dims), sharex=True)
        # Ensure axes is iterable even if num_dims == 1
        if num_dims == 1:
            axes = [axes]
            
        fig.suptitle(name.replace("_", " ").title(), fontsize=16)
        
        for i in range(num_dims):
            if num_dims == 3:
                y_label = ["X", "Y", "Z"][i]
            elif num_dims == 4:
                y_label = ["X", "Y", "Z", "W"][i]
            elif num_dims == 7:
                y_label = ORDERED_JOINT_NAMES_ARM_ISAAC[i]
            elif num_dims == 12:
                y_label = ORDERED_JOINT_NAMES_BASE_ISAAC[i]
            elif num_dims == 19:
                y_label = ORDERED_JOINT_NAMES_ISAAC[i]
            else:
                y_label = f"Dim {i}"
                
            for label, data in datasets.items():
                if len(data.shape) == 2 and data.shape[1] > i:
                    axes[i].plot(data[:, i], label=label)
            axes[i].set_ylabel(y_label)
            axes[i].grid(True, linestyle="--", alpha=0.6)
            axes[i].legend(loc="upper right")
            
        axes[-1].set_xlabel("Time step")
    else:
        print(f"Cannot plot dataset {name} with shape {first_data.shape}")
        return
        
    plt.tight_layout()
         
    out_file = out_dir / f"{name}.png"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Saved plot to {out_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot robot observations from HDF5 log.")
    parser.add_argument("--hdf5_files", type=Path, nargs="+", default=[Path("spot_isaac_sim.hdf5"), Path("spot_isaac_real.hdf5")], help="Paths to the HDF5 log files.")
    args = parser.parse_args()

    valid_files = [f for f in args.hdf5_files if f.exists()]
    if not valid_files:
        print("Error: None of the specified HDF5 files exist.")
        return

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Collect all unique keys
    all_keys = set()
    open_files = {}
    for f_path in valid_files:
        f = h5py.File(f_path, "r")
        open_files[f_path.name] = f
        all_keys.update(f.keys())

    # Plot each key
    for key in sorted(all_keys):
        datasets = {}
        for name, f in open_files.items():
            if key in f:
                datasets[name] = f[key][:]
        if datasets:
            print(f"Plotting {key}...")
            plot_dataset(key, datasets, out_dir)

    # Close files
    for f in open_files.values():
        f.close()


if __name__ == "__main__":
    main()
