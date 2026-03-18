"""Parse proto_bytes from an HDF5 log and save as a JSON list."""

import argparse
import json
import sys
from pathlib import Path

import h5py
from bosdyn.api import robot_command_pb2
from google.protobuf.json_format import MessageToDict


def main():
    parser = argparse.ArgumentParser(
        description="Parse proto_bytes (JointControlStreamRequest) from an HDF5 file and save as a JSON list."
    )
    parser.add_argument(
        "--hdf5_file",
        type=Path,
        default=Path("spot_isaac_real.hdf5"),
        help="Path to the HDF5 log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path. Defaults to <hdf5_basename>_commands.json.",
    )
    args = parser.parse_args()

    if not args.hdf5_file.exists():
        print(f"Error: HDF5 file not found: {args.hdf5_file}")
        sys.exit(1)

    output_path = args.output or args.hdf5_file.with_name(
        args.hdf5_file.stem + "_commands.json"
    )

    with h5py.File(args.hdf5_file, "r") as f:
        if "proto_bytes" not in f:
            print("Error: 'proto_bytes' dataset not found in HDF5 file.")
            sys.exit(1)

        dataset = f["proto_bytes"]
        n = dataset.shape[0]
        print(f"Parsing {n} command protos from '{args.hdf5_file}'...")

        commands = []
        for i in range(n):
            parsed = robot_command_pb2.JointControlStreamRequest()
            parsed.ParseFromString(dataset[i].tobytes())
            commands.append(MessageToDict(parsed))

    with open(output_path, "w") as out:
        json.dump(commands, out, indent=2)

    print(f"Saved {n} commands to '{output_path}'")


if __name__ == "__main__":
    main()
