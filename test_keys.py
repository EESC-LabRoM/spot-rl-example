import h5py
with h5py.File("spot_isaac_real.hdf5", "r") as f:
    print("spot_current_positions:", f["spot_current_positions"][0, 8:12]) # Just checking some indices
    if "preprocessed_joint_positions" in f:
        print("preprocessed_joint_positions:", f["preprocessed_joint_positions"][0, 8:12])
