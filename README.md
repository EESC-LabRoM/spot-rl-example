## Docker
```bash
docker compose -f docker/docker-compose.yaml up -d
docker exec -it docker-isaaclab-1 bash
```
# Spot-RL
Code & Dockerfile for Spot Reinforcement Learning demo

## Deploy a faster policy

Faster exports a deployment bundle containing exactly one ONNX model and its versioned
`policy.yaml` runtime contract. Point either deployment entry point at that directory; do not copy
policy commands into this repository's `env.yaml`.

```bash
# Portable/default flow: first place the matching ONNX + policy.yaml in rl_deploy/configs.
uv run rl_deploy/spot_rl_isaac.py
uv run rl_deploy/spot_rl_demo.py ROBOT_IP

# Or use an external bundle directly.
uv run rl_deploy/spot_rl_isaac.py --policy_dir /path/to/run/exported
uv run rl_deploy/spot_rl_demo.py ROBOT_IP --policy_dir /path/to/run/exported
```

Download a run's model artifact from W&B into `external/<run-name>` and print the resulting
`--policy_dir` path:

```bash
uv run python import.py RUN_ID
```

Recurrent MinGRU state is detected from the ONNX inputs and carried between control steps
automatically.

Policies trained against a moving arm ship an `arm:` block in `policy.yaml` holding the cuRobo
reachability poses, grouped into the training difficulty tiers. `--arm` replays that disturbance
while you drive the robot. Motion begins after two seconds at stow and uses a seeded sequence;
Isaac and the robot match when they execute the same number of 50 Hz command steps. The default
keeps the arm stowed.

```bash
uv run rl_deploy/spot_rl_isaac.py --policy_dir /path/to/run/exported --arm easy  # arm_easy tier
uv run rl_deploy/spot_rl_demo.py ROBOT_IP --policy_dir /path/to/run/exported --arm full  # all tiers
```

See `rl_deploy/configs/README.md` for the exact portable bundle layout. The Faster W&B model
artifact contains both required policy files.

The loader also recognizes the older flat `obs` policies: 65-value locomotion policies and
84-value command-conditioned policies are packed using their training layouts, and their normalized
actions are scaled and shifted into absolute Spot joint targets.

Keep the existing `env.yaml`: it owns Spot-specific joint defaults, gains, standing height, action
scale, and simulator setup. The policy bundle owns observation commands, gait clock parameters,
action semantics, control timing, and recurrent reset behavior. `agent.yaml` is training metadata
and is not read by the deployment pipeline.


# Example of mocked
```bash
export BOSDYN_CLIENT_USERNAME=admin
export BOSDYN_CLIENT_PASSWORD=spotadmin2017
uv run rl_deploy/spot_rl_demo.py  10.0.0.3 --mock
uv run rl_deploy/spot_rl_isaac.py
```


## Plot Acquisition Frequencies
```bash
uv run rl_deploy/plot_acquisition_frequencies.py --hdf5_files spot_isaac_real.hdf5
```

## Timing Diagnostics (dt_*)
The codebase tracks several `dt` variables to monitor system timings and ensure it meets real-time requirements. For a 50Hz system, the expected cycle time is 0.02s (20ms).
- `dt_acquisition_timestamp`: Time between consecutive state observations from the robot. Evaluated from `RobotStateStreamResponse`. Expected: **0.02s**.
- `dt_total_step`: Total latency of the entire control step cycle. It represents the time between consecutive calls to the `OnnxCommandGenerator`. Expected: **~0.02s**.
- `dt_divider_wait`: Time spent waiting for the `EventDivider` to hit the required trigger factor (waiting for new robot states). Expected: **Varies, typically closely follows the period of the state arrivals ~0.02s**.
- `dt_divider_to_onnx`: The gap from when the `EventDivider` finishes waiting until the `OnnxCommandGenerator` starts computing the action. Expected: **Very small (< 0.001s)**.
- `dt_state_arrival_to_compute`: Latency from the exact moment the state arrives via callback in the `StateHandler` until the ONNX action computation starts. Expected: **Very small (< 0.002s)**.
- `dt_onnx_compute`: Time spent strictly running the ONNX model inference. Expected: **Depends on the model, usually < 0.005s**.
- `dt_post_process`: Time taken to process the raw output of the ONNX model into a Spot-ready command. Expected: **Very small (< 0.001s)**.
- `dt_command_latency`: Total round-trip latency from when the robot captures state/receives request to when it receives the corresponding command (computed as `last_command.received_timestamp[i] - header.request_received_timestamp[i-1]`). Expected: **~0.02s - 0.04s**.

To check these statistics across a pre-logged dataset (excluding the turbulent first and last seconds), run:
```bash
uv run rl_deploy/print_timing_diagnostics.py --hdf5_files spot_isaac_real.hdf5
```

Example output:
```
--- Timing Diagnostics for spot_isaac_real.hdf5 ---
Warning: Dataset is too short to skip 1s at start and end. Using all data.
Total samples: 1131, Valid samples used for average: 1131
Variable Name                  | Average (s)     | Expected (s)   
-----------------------------------------------------------------
dt_acquisition_timestamp       | 0.018008        | approx 0.02    
dt_total_step                  | 0.017996        | approx 0.02    
dt_divider_wait                | 0.016983        | < 0.02 (varies)
dt_divider_to_onnx             | 0.000005        | < 0.001        
dt_state_arrival_to_compute    | 0.001178        | < 0.002        
dt_onnx_compute                | 0.000045        | < 0.005        
dt_post_process                | 0.000008        | < 0.001        

--- Command Key Diagnostics for spot_isaac_real.hdf5 ---
Command key range: 51 -> 1181  (total 1131 samples)
  [OK] user_command_key is strictly monotonic with no skips (increment = 1 everywhere)
  [OK] state.last_command.user_command_key matches previous cmd key for all 1130 pairs
```

The script also performs **Command Key Diagnostics** by loading the `proto_bytes` dataset (serialized `JointControlStreamRequest`):
- **Monotonicity check**: Verifies `user_command_key` increments by exactly 1 with no skips or regressions.
- **Cross-check with state proto**: Confirms that `state[i].last_command.user_command_key == cmd[i-1].user_command_key`, ensuring the robot acknowledges each command in sequence.

## Export Command Protos to JSON
Parse the `proto_bytes` dataset (`JointControlStreamRequest`) from an HDF5 log and save all entries as a JSON list:
```bash
uv run rl_deploy/utils/scripts/export_command_protos_to_json.py --hdf5_file spot_isaac_real.hdf5
# Output: spot_isaac_real_commands.json
```

## Test Knee Actuator
You can validate the computed values of the spot knee actuator with positional torque speed limits using the test script, which generates a plot of the limits vs requested actions.
```bash
uv run rl_deploy/isaaclab/test_knee_actuator.py
```

## Compare Actuator Loads
You can plot the actual load against the predicted load from IsaacLab, as well as position and velocity errors, for each knee actuator individually. This will generate multiple plot figures in the `logs` directory.
```bash
uv run rl_deploy/compare_actuator_loads.py --hdf5_file spot_isaac_real.hdf5
```

## Plot All HDF5 Variables (Recursive)
You can recursively plot all variables in an HDF5 file. This is useful for new HDF5 structures like `spot_isaac_real_v2.hdf5`.
```bash
uv run rl_deploy/plot_hdf5_v2.py --file spot_isaac_real_v2.hdf5
```

## Replay and Compare Simulated vs Real Trajectories
You can replay the exact velocity commands recorded in the HDF5 log through the IsaacLab simulation and plot the resulting base velocities and key joint positions side-by-side with the real robot's telemetry. This script generates comparative plots in the `logs` directory.
```bash
uv run rl_deploy/replay_and_compare_sim_real.py --hdf5_file spot_isaac_real.hdf5
```

## Play relic 
```bash
uv run rl_deploy/relic/play.py
```

# Troubleshooting

## Antigravity IDE / VS Code Python Extension Missing `pet` Binary
If you encounter `bash: /home/.../.antigravity/extensions/ms-python.python-.../python-env-tools/bin/pet: No such file or directory` spam in your IDE terminal, you can resolve it by creating a silent dummy `pet` script in the missing directory:

```bash
# Example fix script:
mkdir -p ~/.antigravity/extensions/ms-python.python-2026.2.0-universal/python-env-tools/bin/
echo -e '#!/bin/bash\nexit 0' > ~/.antigravity/extensions/ms-python.python-2026.2.0-universal/python-env-tools/bin/pet
chmod +x ~/.antigravity/extensions/ms-python.python-2026.2.0-universal/python-env-tools/bin/pet
```

# Architecture Notes

## Spot Class Threading (`rl_deploy/spot/spot.py`)
The `Spot` class uses **two threads** for concurrent gRPC streaming:
- **`_state_thread`**: Listens for robot state updates at ~333Hz.
- **`_command_thread`**: Sends joint commands via a streaming RPC.

Joint control activation is handled inline inside the command stream generator (`_command_stream_loop`) after the first command is yielded — no separate thread is needed for this.
