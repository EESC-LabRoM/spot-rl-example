# Agent Map

Purpose: keep token use low. Read this first, then inspect only the touched path.

## Repo Map

- `scripts/`: main entrypoints only: `spot_rl_demo.py`, `spot_rl_isaac.py`.
- `utils/`: one-off/diagnostic scripts. These are executable helpers, not importable package utilities.
- `rl_deploy/`: package code. Runtime imports should stay package-qualified (`rl_deploy...`).
- `rl_deploy/orbit/`: active runtime policy/config/obs code. Name is old; do not treat as dead.
- `rl_deploy/spot/`: real/mock Spot API, constants, URDF, meshes.
- `rl_deploy/isaaclab_spot/`: IsaacLab env, mock spot bridge, actuator model.
- `rl_deploy/utils/`: importable helpers only. Do not put generated artifacts here.
- `rl_deploy/legacy/`: old ReLIC/old configs. Avoid unless user asks.
- `docker/`: compose + IsaacLab image.
- `artifacts/`: generated/large/output files. Subdirs: `datasets`, `logs`, `policies`, `exports`, `runtime`.

## Main Commands

```bash
uv run scripts/spot_rl_demo.py 10.0.0.3 --mock
uv run scripts/spot_rl_isaac.py
uv run utils/print_timing_diagnostics.py --hdf5_files artifacts/datasets/spot_isaac_real.hdf5
python -m compileall rl_deploy scripts utils
```

## Do Not Break

- Keep only main entrypoints in `scripts/`.
- Keep runtime package imports rooted at `rl_deploy`.
- Keep `rl_deploy/orbit` import paths stable; it is still used by demo + Isaac.
- Keep configs at `rl_deploy/configs`.
- Keep generated files under `artifacts`, not repo root.
- `artifacts/runtime/stow.csv` is used by `rl_deploy/isaaclab_spot/spot_env.py`.

## Common Edits

- New diagnostics: add to top-level `utils/`, default outputs to `artifacts/logs` or `artifacts/exports`.
- New runtime helper: add to `rl_deploy/utils/`, not top-level `utils`.
- New data/policies/logs: use `artifacts/datasets`, `artifacts/policies`, `artifacts/logs`.
- Legacy experiments: put under `rl_deploy/legacy`.
