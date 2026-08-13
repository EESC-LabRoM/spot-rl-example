import argparse
from pathlib import Path

import wandb
import yaml


parser = argparse.ArgumentParser(description="Download a FASTER policy from W&B.")
parser.add_argument("run_id", help="W&B run ID or entity/project/run ID")
args = parser.parse_args()

run_path = args.run_id if "/" in args.run_id else f"tommaselli/faster/{args.run_id}"
run = wandb.Api().run(run_path)
artifact = next(item for item in run.logged_artifacts() if item.type == "model")
files = artifact.files()
onnx = next(item for item in files if item.name.endswith(".onnx"))
manifest = next(item for item in files if Path(item.name).name == "policy.yaml")
policy_dir = Path(__file__).parent / "external" / run.name
policy_dir.mkdir(parents=True, exist_ok=False)
onnx_path = Path(onnx.download(root=policy_dir))
manifest_path = Path(manifest.download(root=policy_dir))
target = policy_dir / f"{run.name}.onnx"
onnx_path.rename(target)
config = yaml.safe_load(manifest_path.read_text())
config["model"]["file"] = target.name
(policy_dir / "policy.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
if manifest_path != policy_dir / "policy.yaml":
    manifest_path.unlink()
print(policy_dir.resolve())
