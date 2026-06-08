# Spot-RL

Spot reinforcement learning deployment and IsaacLab validation code.

## Layout

- `docker/`: IsaacLab container setup.
- `scripts/`: main runnable entrypoints only.
- `utils/`: utility scripts for diagnostics, plotting, exports, and inspection.
- `rl_deploy/`: importable runtime package.
- `rl_deploy/orbit/`: active policy/config/observation runtime code despite the old name.
- `rl_deploy/legacy/`: old ReLIC/orbit-era experiments and configs.
- `artifacts/`: generated datasets, logs, policies, exports, and runtime support files.

## Docker

```bash
docker compose -f docker/docker-compose.yaml up -d
docker exec -it docker-isaaclab-1 bash
```

## Main Flows

```bash
export BOSDYN_CLIENT_USERNAME=admin
export BOSDYN_CLIENT_PASSWORD=spotadmin2017

uv run scripts/spot_rl_demo.py 10.0.0.3 --mock
uv run scripts/spot_rl_isaac.py
```

Default HDF5 outputs now go to `artifacts/datasets/`.

## Diagnostics

```bash
uv run utils/plot_acquisition_frequencies.py --hdf5_files artifacts/datasets/spot_isaac_real.hdf5
uv run utils/print_timing_diagnostics.py --hdf5_files artifacts/datasets/spot_isaac_real.hdf5
uv run utils/export_command_protos_to_json.py --hdf5_file artifacts/datasets/spot_isaac_real.hdf5
uv run utils/plot_hdf5_v2.py --files artifacts/datasets/spot_isaac_real_v2.hdf5
uv run utils/replay_and_compare_sim_real.py --hdf5_file artifacts/datasets/spot_isaac_real.hdf5
uv run utils/compare_actuator_loads.py --hdf5_file artifacts/datasets/spot_isaac_real.hdf5
```

Plots and derived outputs go to `artifacts/logs/` or `artifacts/exports/`.

## Legacy

ReLIC playback is retained for reference:

```bash
uv run rl_deploy/legacy/relic/play.py
```

## Architecture Note

`rl_deploy/spot/spot.py` streams robot state and joint commands concurrently. The active control path is:

`scripts/spot_rl_demo.py` -> `rl_deploy/orbit/onnx_command_generator.py` -> `rl_deploy/spot/`.
