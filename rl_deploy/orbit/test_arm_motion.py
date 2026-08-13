import unittest

from rl_deploy.orbit.arm_motion import ArmMotion, from_manifest
from rl_deploy.spot.constants import ORDERED_JOINT_NAMES_SPOT_ARM

STOW = [0.0, -3.1415, 3.1415, 1.5655, 0.0, -1.5655, 0.0]
POSE_A = [0.1] * 7
POSE_B = [0.2] * 7


def _manifest(tiers=None, joint_names=None):
    return {
        "arm": {
            "joint_names": joint_names or list(ORDERED_JOINT_NAMES_SPOT_ARM),
            "stow": list(STOW),
            "traj_range_s": [2.5, 4.0],
            "hold_range_s": [4.0, 7.0],
            "tiers": tiers if tiers is not None else {"arm_easy": [POSE_A], "arm_hard": [POSE_B]},
        }
    }


class FromManifestTest(unittest.TestCase):
    def test_off_returns_none(self):
        self.assertIsNone(from_manifest(_manifest(), "off"))

    def test_missing_arm_block_raises(self):
        with self.assertRaisesRegex(ValueError, "not.*trained with a moving arm"):
            from_manifest({"contract_version": 2}, "easy")

    def test_missing_easy_tier_raises(self):
        with self.assertRaisesRegex(ValueError, "no arm_easy tier"):
            from_manifest(_manifest(tiers={"all": [POSE_A]}), "easy")

    def test_joint_order_mismatch_raises(self):
        names = list(reversed(ORDERED_JOINT_NAMES_SPOT_ARM))
        with self.assertRaisesRegex(ValueError, "does not match deployment"):
            from_manifest(_manifest(joint_names=names), "full")

    def test_easy_uses_only_easy_tier(self):
        self.assertEqual(from_manifest(_manifest(), "easy").poses, [POSE_A])

    def test_full_concatenates_all_tiers(self):
        self.assertEqual(from_manifest(_manifest(), "full").poses, [POSE_A, POSE_B])


class ArmMotionTest(unittest.TestCase):
    def test_first_ticks_emit_stow(self):
        motion = ArmMotion([POSE_A], [2.5, 4.0], [4.0, 7.0], STOW)

        for _ in range(100):
            self.assertEqual(motion.step(), STOW)

    def test_lerp_reaches_goal_then_resamples_without_a_jump(self):
        motion = ArmMotion(
            [POSE_A], [0.04, 0.04], [0.02, 0.02], [0.0] * 7
        )

        for _ in range(101):
            self.assertEqual(motion.step(), [0.0] * 7)
        self.assertEqual(motion.step(), [0.05] * 7)
        self.assertEqual(motion.step(), POSE_A)
        self.assertEqual(motion.step(), POSE_A)
        self.assertEqual(motion.start, POSE_A)

    def test_same_seed_reproduces_sequence(self):
        args = ([POSE_A, POSE_B], [0.1, 0.4], [0.1, 0.3], STOW)
        left = ArmMotion(*args, seed=7)
        right = ArmMotion(*args, seed=7)

        self.assertEqual(
            [left.step() for _ in range(1000)], [right.step() for _ in range(1000)]
        )

    def test_different_seeds_diverge(self):
        args = ([POSE_A, POSE_B], [0.1, 0.4], [0.1, 0.3], STOW)
        left = ArmMotion(*args, seed=0)
        right = ArmMotion(*args, seed=1)

        self.assertNotEqual(
            [left.step() for _ in range(1000)], [right.step() for _ in range(1000)]
        )


if __name__ == "__main__":
    unittest.main()
