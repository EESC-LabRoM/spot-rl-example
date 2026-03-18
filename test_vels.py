import h5py
with h5py.File("spot_isaac_real.hdf5", "r") as f:
    print("spot_current_velocities:", f["spot_current_velocities"][0, 8:12])
    print("raw_joint_velocities:", f["raw_joint_velocities"][0, 8:12])
    print("commanded_action:", f["commanded_action"][0, 8:12])
