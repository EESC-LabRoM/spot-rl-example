# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


class HDF5Logger:
    """Class to buffer and save robot states and observations to an HDF5 file."""

    def __init__(self, log_path: str, metadata_path: str | os.PathLike | None = None):
        self.log_path = log_path
        self._first_timestamp = None
        self.metadata = self._load_metadata(metadata_path)
        self.data: Dict[str, List] = {
            "raw_base_linear_velocity": [],
            "raw_base_angular_velocity": [],
            "raw_projected_gravity": [],
            "raw_joint_positions": [],
            "raw_joint_velocities": [],
            "raw_joint_loads": [],
            "commanded_action": [],
            "spot_current_positions": [],
            "spot_current_velocities": [],
            "preprocessed_base_linear_velocity": [],
            "preprocessed_base_angular_velocity": [],
            "preprocessed_projected_gravity": [],
            "preprocessed_velocity_cmd": [],
            "preprocessed_joint_positions": [],
            "preprocessed_joint_velocities": [],
            "preprocessed_last_action": [],
            "response_timestamp": [],
            "dt_divider_wait": [],
            "dt_divider_to_onnx": [],
            "dt_onnx_compute": [],
            "dt_post_process": [],
            "dt_total_step": [],
            "dt_state_arrival_to_compute": [],
            "raw_state_proto_bytes": [],
            "proto_bytes": [],
            "foot_contact_enum": [],
            "foot_contact_binary": [],
            "foot_contact_transition": [],
            "imu_linear_acceleration": [],
            "imu_angular_velocity": [],
            "imu_packet_timestamp": [],
            "command_user_key": [],
            "command_request_timestamp": [],
            "command_end_time": [],
            "tracking_error_velocity": [],
            "joint_position_error": [],
        }

    def _load_metadata(
        self, metadata_path: str | os.PathLike | None
    ) -> Dict[str, Any]:
        if metadata_path is None:
            return {}

        path = Path(metadata_path)
        if not path.exists():
            print(f"Metadata sidecar not found: {path}")
            return {}

        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata sidecar must contain a JSON object: {path}")

        metadata["metadata_json_raw"] = json.dumps(metadata, sort_keys=True)
        return metadata

    def _timestamp_to_seconds(self, timestamp) -> float:
        if hasattr(timestamp, "timestamp"):
            return float(timestamp.timestamp())
        return float(timestamp.seconds) + float(timestamp.nanos) * 1e-9

    def _metadata_attr_value(self, value: Any):
        if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
            return value
        return json.dumps(value, sort_keys=True)

    def log_state(
        self,
        raw_base_linear_velocity: List[float],
        raw_base_angular_velocity: List[float],
        raw_projected_gravity: List[float],
        raw_joint_positions: List[float],
        raw_joint_velocities: List[float],
        raw_joint_loads: List[float],
        spot_current_positions: List[float],
        spot_current_velocities: List[float],
        preprocessed_base_linear_velocity: List[float],
        preprocessed_base_angular_velocity: List[float],
        preprocessed_projected_gravity: List[float],
        preprocessed_velocity_cmd: List[float],
        preprocessed_joint_positions: List[float],
        preprocessed_joint_velocities: List[float],
        preprocessed_last_action: List[float],
        commanded_action: List[float],
        response_timestamp: datetime.datetime,
        dt_divider_wait: float,
        dt_divider_to_onnx: float,
        dt_onnx_compute: float,
        dt_post_process: float,
        dt_total_step: float,
        dt_state_arrival_to_compute: float,
        raw_state_proto_bytes: bytes,
        proto_bytes: bytes,
        foot_contact_enum: List[int] | None = None,
        foot_contact_binary: List[int] | None = None,
        foot_contact_transition: List[int] | None = None,
        imu_linear_acceleration: List[float] | None = None,
        imu_angular_velocity: List[float] | None = None,
        imu_packet_timestamp: float | None = None,
        command_user_key: int | None = None,
        command_request_timestamp: float | None = None,
        command_end_time: float | None = None,
    ):
        """Append a single step of data to the buffers."""
        self.data["raw_base_linear_velocity"].append(raw_base_linear_velocity)
        self.data["raw_base_angular_velocity"].append(raw_base_angular_velocity)
        self.data["raw_projected_gravity"].append(raw_projected_gravity)
        self.data["raw_joint_positions"].append(raw_joint_positions)
        self.data["raw_joint_velocities"].append(raw_joint_velocities)
        self.data["raw_joint_loads"].append(raw_joint_loads)
        self.data["spot_current_positions"].append(spot_current_positions)
        self.data["spot_current_velocities"].append(spot_current_velocities)
        self.data["preprocessed_base_linear_velocity"].append(
            preprocessed_base_linear_velocity
        )
        self.data["preprocessed_base_angular_velocity"].append(
            preprocessed_base_angular_velocity
        )
        self.data["preprocessed_projected_gravity"].append(
            preprocessed_projected_gravity
        )
        self.data["preprocessed_velocity_cmd"].append(preprocessed_velocity_cmd)
        self.data["preprocessed_joint_positions"].append(preprocessed_joint_positions)
        self.data["preprocessed_joint_velocities"].append(preprocessed_joint_velocities)
        self.data["preprocessed_last_action"].append(preprocessed_last_action)
        self.data["commanded_action"].append(commanded_action)
        self.data["dt_divider_wait"].append(dt_divider_wait)
        self.data["dt_divider_to_onnx"].append(dt_divider_to_onnx)
        self.data["dt_onnx_compute"].append(dt_onnx_compute)
        self.data["dt_post_process"].append(dt_post_process)
        self.data["dt_total_step"].append(dt_total_step)
        self.data["dt_state_arrival_to_compute"].append(dt_state_arrival_to_compute)
        if self._first_timestamp is None:
            self._first_timestamp = response_timestamp
        delta_time = self._timestamp_to_seconds(response_timestamp) - self._timestamp_to_seconds(
            self._first_timestamp
        )
        self.data["response_timestamp"].append(delta_time)
        self.data["raw_state_proto_bytes"].append(raw_state_proto_bytes)
        self.data["proto_bytes"].append(proto_bytes)
        self.data["foot_contact_enum"].append(foot_contact_enum or [0, 0, 0, 0])
        self.data["foot_contact_binary"].append(foot_contact_binary or [0, 0, 0, 0])
        self.data["foot_contact_transition"].append(
            foot_contact_transition or [0, 0, 0, 0]
        )
        self.data["imu_linear_acceleration"].append(
            imu_linear_acceleration or [0.0, 0.0, 0.0]
        )
        self.data["imu_angular_velocity"].append(imu_angular_velocity or [0.0, 0.0, 0.0])
        self.data["imu_packet_timestamp"].append(
            np.nan if imu_packet_timestamp is None else imu_packet_timestamp
        )
        self.data["command_user_key"].append(
            -1 if command_user_key is None else command_user_key
        )
        self.data["command_request_timestamp"].append(
            np.nan if command_request_timestamp is None else command_request_timestamp
        )
        self.data["command_end_time"].append(
            np.nan if command_end_time is None else command_end_time
        )

        velocity_cmd = np.array(preprocessed_velocity_cmd, dtype=np.float32).reshape(-1)
        base_velocity = np.array(raw_base_linear_velocity, dtype=np.float32).reshape(-1)
        self.data["tracking_error_velocity"].append(velocity_cmd[:3] - base_velocity[:3])

        action = np.array(commanded_action, dtype=np.float32).reshape(-1)
        joint_pos = np.array(raw_joint_positions, dtype=np.float32).reshape(-1)
        self.data["joint_position_error"].append(action[:12] - joint_pos[:12])

    def save(self):
        """Write all buffered data to the HDF5 file."""
        if not self.log_path:
            return

        print(f"Saving HDF5 log to {self.log_path}...")
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        # Create variable-length datatype for raw bytes arrays
        vlen_bytes_dtype = h5py.vlen_dtype(np.uint8)
        with h5py.File(self.log_path, "w") as f:
            for key, value in self.metadata.items():
                f.attrs[key] = self._metadata_attr_value(value)

            for key, val in self.data.items():
                if len(val) > 0:
                    if key == "raw_state_proto_bytes" or key == "proto_bytes":
                        # Convert bytes strings into variable length numpy arrays of uint8
                        byte_arrays = [np.frombuffer(b, dtype=np.uint8) for b in val]
                        ds = f.create_dataset(key, (len(val),), dtype=vlen_bytes_dtype)
                        for i, arr in enumerate(byte_arrays):
                            ds[i] = arr
                    elif key in {
                        "foot_contact_enum",
                        "foot_contact_binary",
                        "foot_contact_transition",
                        "command_user_key",
                    }:
                        f.create_dataset(key, data=np.array(val, dtype=np.int32))
                    elif key in {
                        "imu_packet_timestamp",
                        "command_request_timestamp",
                        "command_end_time",
                    }:
                        f.create_dataset(key, data=np.array(val, dtype=np.float64))
                    else:
                        f.create_dataset(key, data=np.array(val, dtype=np.float32))
        print("HDF5 log saved successfully.")
