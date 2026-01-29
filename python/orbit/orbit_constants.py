# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""spots base joints in order expected/used by orbit libraries"""

ORDERED_JOINT_NAMES_BASE_ORBIT = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]

ORDERED_JOINT_NAMES_ARM_ORBIT = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]


ORDERED_JOINT_NAMES_ORBIT =  ORDERED_JOINT_NAMES_BASE_ORBIT + ORDERED_JOINT_NAMES_ARM_ORBIT