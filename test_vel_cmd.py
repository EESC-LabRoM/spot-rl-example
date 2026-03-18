import h5py
with h5py.File("spot_isaac_real.hdf5", "r") as f:
    if "preprocessed_velocity_cmd" in f:
        print("preprocessed_velocity_cmd shape:", f["preprocessed_velocity_cmd"].shape)
        print("sample:", f["preprocessed_velocity_cmd"][0:5])
