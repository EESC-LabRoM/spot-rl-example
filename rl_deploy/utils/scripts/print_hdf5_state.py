import sys
import os
import random
import h5py

# Add the project root to the path so we can import from rl_deploy and bosdyn if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse

def main():
    # Path to spot_isaac_real.hdf5 in the root of the project
    hdf5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../spot_isaac_real.hdf5'))
    
    if not os.path.exists(hdf5_path):
        print(f"Error: Could not find HDF5 file at {hdf5_path}")
        sys.exit(1)

    with h5py.File(hdf5_path, "r") as f:
        if "raw_state_proto_bytes" not in f:
            print("Error: 'raw_state_proto_bytes' dataset not found in HDF5 file.")
            sys.exit(1)
            
        dataset = f["raw_state_proto_bytes"]
        num_timesteps = dataset.shape[0]
        
        if num_timesteps == 0:
            print("Error: The 'raw_state_proto_bytes' dataset is empty.")
            sys.exit(1)

        # Select a random timestep
        random_idx = random.randint(0, num_timesteps - 1)
        print(f"Loaded HDF5 file with {num_timesteps} timesteps.")
        print(f"Displaying data for random timestep: {random_idx}\n")
        
        # Read the bytes data for the random timestep
        bytes_data = dataset[random_idx]
        parsed_bytes = bytes_data.tobytes()
        
        from google.protobuf.json_format import MessageToJson
        
        # Parse the protobuf
        parsed_state = RobotStateStreamResponse()
        parsed_state.ParseFromString(parsed_bytes)
        
        # Convert to JSON and save to file
        json_str = MessageToJson(parsed_state)
        json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../robot_state.json'))
        
        with open(json_path, 'w') as json_file:
            json_file.write(json_str)
            
        print(f"Robot state for timestep {random_idx} saved to {json_path}")

if __name__ == "__main__":
    main()
