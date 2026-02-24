import argparse
import os
import sys
from pathlib import Path
from operator import sub

import h5py
import matplotlib.pyplot as plt

# allow for absolute imports from 'rl_deploy'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rl_deploy.orbit.orbit_configuration import (
    OrbitConfig,
    load_configuration,
    detect_config_file,
)
from rl_deploy.orbit.orbit_constants import (
    ORDERED_JOINT_NAMES_ARM_ISAAC,
    ORDERED_JOINT_NAMES_BASE_ISAAC,
    ORDERED_JOINT_NAMES_ISAAC,
)
from rl_deploy.spot.constants import (
    JOINT_LIMITS,
    JOINT_SOFT_LIMITS,
    ORDERED_JOINT_NAMES_SPOT,
)
from rl_deploy.utils.dict_tools import dict_to_list, find_ordering, reorder


# [SAFETY STOP] Joint fl_kn value -1.8846 outside safe range [-1.8814, -0.2471]
# [SAFETY STOP] Joint fl_hx value 0.4965 outside safe range [-0.4253, 0.4948]
def get_safety_limits(config: OrbitConfig):
    """Generate safe limits for each joint in Isaac order, shifted by default offsets."""
    safe_limits_min = {}
    safe_limits_max = {}

    for joint_name in ORDERED_JOINT_NAMES_SPOT:
        if joint_name in JOINT_SOFT_LIMITS:
            lower = JOINT_LIMITS[joint_name]["lower"]
            upper = JOINT_LIMITS[joint_name]["upper"]
            middle = (lower + upper) / 2
            full_range = upper - lower

            min_margin, max_margin = JOINT_SOFT_LIMITS[joint_name]
            min_val = middle - (min_margin * full_range / 2)
            max_val = middle + (max_margin * full_range / 2)

            safe_limits_min[joint_name] = min_val
            safe_limits_max[joint_name] = max_val
        else:
            # For joints without soft limits (like arm joints), use the hard limits
            safe_limits_min[joint_name] = JOINT_LIMITS[joint_name]["lower"]
            safe_limits_max[joint_name] = JOINT_LIMITS[joint_name]["upper"]

    # Reorder to Isaac
    spot_to_isaac = find_ordering(ORDERED_JOINT_NAMES_SPOT, ORDERED_JOINT_NAMES_ISAAC)

    min_vals_spot = [safe_limits_min[name] for name in ORDERED_JOINT_NAMES_SPOT]
    max_vals_spot = [safe_limits_max[name] for name in ORDERED_JOINT_NAMES_SPOT]

    min_vals_isaac = reorder(min_vals_spot, spot_to_isaac)
    max_vals_isaac = reorder(max_vals_spot, spot_to_isaac)

    # Subtract default offsets
    default_joints = dict_to_list(config.default_joints, ORDERED_JOINT_NAMES_ISAAC)

    min_vals_shifted = list(map(sub, min_vals_isaac, default_joints))
    max_vals_shifted = list(map(sub, max_vals_isaac, default_joints))

    return min_vals_shifted, max_vals_shifted


def plot_dataset(name: str, datasets: dict, out_dir: Path, safety_limits=None):
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
        fig, axes = plt.subplots(
            nrows=num_dims, ncols=1, figsize=(10, 2 * num_dims), sharex=True
        )
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

            if safety_limits and name in [
                "preprocessed_joint_positions",
                "raw_joint_positions",
            ]:
                lower, upper = safety_limits
                axes[i].axhline(
                    y=lower[i],
                    color="r",
                    linestyle="--",
                    alpha=0.5,
                    label="Safety Limit" if i == 0 else "",
                )
                axes[i].axhline(y=upper[i], color="r", linestyle="--", alpha=0.5)

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
    parser = argparse.ArgumentParser(
        description="Plot robot observations from HDF5 log."
    )
    parser.add_argument(
        "--hdf5_files",
        type=Path,
        nargs="+",
        default=[Path("spot_isaac_sim.hdf5"), Path("spot_isaac_real.hdf5")],
        help="Paths to the HDF5 log files.",
    )
    parser.add_argument(
        "--policy_file_path",
        type=Path,
        default=Path(__file__).parent / "configs",
        help="Path to the policy directory.",
    )
    args = parser.parse_args()

    # Load Orbit config and safety limits
    env_config = detect_config_file(args.policy_file_path)
    safety_limits = None
    if env_config:
        config = load_configuration(env_config)
        safety_limits = get_safety_limits(config)
    else:
        print(
            f"Warning: Could not find orbit config in {args.policy_file_path}. Safety limits will not be plotted."
        )

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
            plot_dataset(key, datasets, out_dir, safety_limits=safety_limits)

    # Close files
    for f in open_files.values():
        f.close()


if __name__ == "__main__":
    main()
