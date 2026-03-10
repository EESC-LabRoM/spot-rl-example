import sys
import os
from datetime import datetime
import numpy as np
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from rl_deploy.utils.hdf5_logger import HDF5Logger
from bosdyn.api import robot_state_pb2
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse


def test_hdf5_logger_protobuf():
    print("Creating logger...")
    logger = HDF5Logger("test_log.h5")
    
    # Create fake RobotStateStreamResponse
    fake_state = RobotStateStreamResponse()
    fake_state.joint_states.position.extend([1.0, 2.0, 3.0])
    
    print("Logging state...")
    logger.log_state(
        raw_base_linear_velocity=[0.1, 0.2, 0.3],
        raw_base_angular_velocity=[0.1, 0.2, 0.3],
        raw_projected_gravity=[0.0, 0.0, -9.8],
        raw_joint_positions=[1.0, 2.0, 3.0],
        raw_joint_velocities=[0.1, 0.2, 0.3],
        raw_joint_loads=[0.1, 0.2, 0.3],
        spot_current_positions=[1.0, 2.0, 3.0],
        spot_current_velocities=[0.1, 0.2, 0.3],
        preprocessed_base_linear_velocity=[0.1, 0.2, 0.3],
        preprocessed_base_angular_velocity=[0.1, 0.2, 0.3],
        preprocessed_projected_gravity=[0.0, 0.0, -1.0],
        preprocessed_velocity_cmd=[1.0, 0.0, 0.0],
        preprocessed_joint_positions=[1.0, 2.0, 3.0],
        preprocessed_joint_velocities=[0.1, 0.2, 0.3],
        preprocessed_last_action=[0.0] * 12,
        commanded_action=[0.0] * 12,
        response_timestamp=datetime.now(),
        dt_divider_wait=0.01,
        dt_divider_to_onnx=0.02,
        dt_onnx_compute=0.03,
        dt_post_process=0.04,
        dt_total_step=0.1,
        dt_state_arrival_to_compute=0.05,
        raw_state_proto_bytes=fake_state.SerializeToString()
    )
    
    print("Saving logger...")
    logger.save()
    
    print("Reading logger back...")
    with h5py.File("test_log.h5", "r") as f:
        bytes_data = f["raw_state_proto_bytes"][0]
        # convert from uint8 numpy array to bytes
        parsed_bytes = bytes_data.tobytes()
        
        parsed_state = RobotStateStreamResponse()
        parsed_state.ParseFromString(parsed_bytes)
        
        print("Parsed state joint positions:", list(parsed_state.joint_states.position))
        assert list(parsed_state.joint_states.position) == [1.0, 2.0, 3.0]
        print("Test passed successfully!")

if __name__ == "__main__":
    test_hdf5_logger_protobuf()
