# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""spots base joints in order expected/used by orbit libraries"""

ORDERED_JOINT_NAMES_BASE_ISAAC = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]

ORDERED_JOINT_NAMES_ARM_ISAAC = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]


ORDERED_JOINT_NAMES_ISAAC =  ORDERED_JOINT_NAMES_BASE_ISAAC + ORDERED_JOINT_NAMES_ARM_ISAAC