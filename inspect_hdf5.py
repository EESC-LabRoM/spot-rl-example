import h5py
import numpy as np
from pathlib import Path
from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT
hdf5_path = Path("spot_isaac_real.hdf5")
if hdf5_path.exists():
    with h5py.File(hdf5_path, "r") as f:
        print("Keys:", list(f.keys()))
        joint_positions = f["raw_joint_positions"][:]
        commanded_action = f["commanded_action"][:]
        knee_indices = [ORDERED_JOINT_NAMES_SPOT.index(name) for name in ORDERED_JOINT_NAMES_SPOT if name.endswith("_kn")]
        print("Knee indices:", knee_indices)
        print("Joint positions [0]:", joint_positions[0, knee_indices])
        print("Commanded action [0]:", commanded_action[0, knee_indices])
        print("Commanded action shape:", commanded_action.shape)
        
        err = commanded_action[:, knee_indices] - joint_positions[:, knee_indices]
        print("Mean error:", np.mean(err, axis=0))
else:
    print("No HDF5 file.")
