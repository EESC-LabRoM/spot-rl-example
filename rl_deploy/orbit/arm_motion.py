"""Replay the arm disturbance a policy was trained against.

Single-robot port of faster's `events.ArmReachHold`: lerp the arm target to a sampled cuRobo
reachability pose over U(*traj_range_s), hold it for U(*hold_range_s), then resample from wherever
the target currently is. The poses and timing ride in the exported policy.yaml `arm:` block, so a
bundle describes its own arm motion.

The RNG is seeded, so the Isaac twin and the real robot replay the identical sequence.
"""

import random

from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT_ARM


class ArmMotion:
    def __init__(
        self,
        poses,
        traj_range_s,
        hold_range_s,
        stow,
        step_dt: float = 0.02,
        seed: int = 0,
        start_hold_s: float = 2.0,
    ):
        self.poses = [list(pose) for pose in poses]
        self.traj_range = tuple(traj_range_s)
        self.hold_range = tuple(hold_range_s)
        self.stow = list(stow)
        self.step_dt = step_dt
        self.rng = random.Random(seed)
        # Open on a degenerate segment at stow: the first ticks emit exactly the pose the arm is
        # already holding, so enabling motion never steps the target.
        self.start = list(stow)
        self.goal = list(stow)
        self.traj_steps = 1.0
        self.hold_steps = start_hold_s / step_dt
        self.elapsed = 0.0

    def _resample(self, current):
        self.start = list(current)
        self.goal = self.poses[self.rng.randrange(len(self.poses))]
        self.traj_steps = max(self.rng.uniform(*self.traj_range) / self.step_dt, 1.0)
        self.hold_steps = max(self.rng.uniform(*self.hold_range) / self.step_dt, 0.0)
        self.elapsed = 0.0

    def step(self):
        """Advance one control step and return the 7 arm joint targets."""
        self.elapsed += 1.0
        alpha = min(self.elapsed / self.traj_steps, 1.0)
        target = [s + alpha * (g - s) for s, g in zip(self.start, self.goal)]
        if self.elapsed >= self.traj_steps + self.hold_steps:
            self._resample(target)
        return target


def from_manifest(manifest, mode: str, seed: int = 0):
    """Build the ArmMotion a policy bundle asks for, or None when the arm stays stowed."""
    if mode == "off":
        return None

    arm = (manifest or {}).get("arm")
    if not arm:
        raise ValueError(
            "--arm requested but policy.yaml has no 'arm' block: this policy was not "
            "trained with a moving arm"
        )

    joint_names = list(arm["joint_names"])
    if joint_names != ORDERED_JOINT_NAMES_SPOT_ARM:
        raise ValueError(
            f"policy.yaml arm joint order {joint_names} does not match deployment "
            f"{ORDERED_JOINT_NAMES_SPOT_ARM}"
        )

    tiers = arm["tiers"]
    if mode == "easy" and "arm_easy" not in tiers:
        raise ValueError(f"policy.yaml has no arm_easy tier; available tiers: {list(tiers)}")
    poses = (
        list(tiers["arm_easy"])
        if mode == "easy"
        else [pose for tier in tiers.values() for pose in tier]
    )

    print(
        f"[arm] mode={mode} poses={len(poses)} traj={arm['traj_range_s']}s "
        f"hold={arm['hold_range_s']}s seed={seed}"
    )
    return ArmMotion(poses, arm["traj_range_s"], arm["hold_range_s"], arm["stow"], seed=seed)
