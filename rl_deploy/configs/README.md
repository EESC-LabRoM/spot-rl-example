# Policy drop-in directory

Both deployment scripts use this directory when no policy option is passed:

```bash
uv run rl_deploy/spot_rl_isaac.py
uv run rl_deploy/spot_rl_demo.py ROBOT_IP
```

Before running either command, place exactly these two files here:

```text
rl_deploy/configs/
├── <run-name>.onnx
├── policy.yaml
├── env.yaml
└── agent.yaml
```

- Keep exactly one `.onnx` file.
- The ONNX and `policy.yaml` must come from the same Faster `exported/` directory.
- Keep this repository's `env.yaml`; it contains deployment-specific Spot and simulator settings.
- `agent.yaml` is retained metadata and is not read by deployment.

Faster uploads the ONNX and `policy.yaml` together in the run's model artifact, so a deployment
machine does not need the Faster checkout or local run directory. Download both artifact files,
replace the old ONNX and manifest here, and run without `--policy_dir`.

To use a bundle elsewhere without copying it here:

```bash
uv run rl_deploy/spot_rl_isaac.py --policy_dir /path/to/exported
uv run rl_deploy/spot_rl_demo.py ROBOT_IP --policy_dir /path/to/exported
```
