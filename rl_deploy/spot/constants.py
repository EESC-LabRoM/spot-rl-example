# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#bosdyn-api-CombinedJointStates
from enum import IntEnum
import os 
from rl_deploy.utils.urdf import parse_urdf_limits
from bosdyn.api.spot import spot_constants_pb2


class DOF(IntEnum):
    FL_HX = spot_constants_pb2.JOINT_INDEX_FL_HX
    FL_HY = spot_constants_pb2.JOINT_INDEX_FL_HY
    FL_KN = spot_constants_pb2.JOINT_INDEX_FL_KN
    FR_HX = spot_constants_pb2.JOINT_INDEX_FR_HX
    FR_HY = spot_constants_pb2.JOINT_INDEX_FR_HY
    FR_KN = spot_constants_pb2.JOINT_INDEX_FR_KN
    HL_HX = spot_constants_pb2.JOINT_INDEX_HL_HX
    HL_HY = spot_constants_pb2.JOINT_INDEX_HL_HY
    HL_KN = spot_constants_pb2.JOINT_INDEX_HL_KN
    HR_HX = spot_constants_pb2.JOINT_INDEX_HR_HX
    HR_HY = spot_constants_pb2.JOINT_INDEX_HR_HY
    HR_KN = spot_constants_pb2.JOINT_INDEX_HR_KN
    # Arm
    ARM_SH0 = spot_constants_pb2.JOINT_INDEX_A0_SH0
    ARM_SH1 = spot_constants_pb2.JOINT_INDEX_A0_SH1
    ARM_EL0 = spot_constants_pb2.JOINT_INDEX_A0_EL0
    ARM_EL1 = spot_constants_pb2.JOINT_INDEX_A0_EL1
    ARM_WR0 = spot_constants_pb2.JOINT_INDEX_A0_WR0
    ARM_WR1 = spot_constants_pb2.JOINT_INDEX_A0_WR1
    # Hand
    ARM_F1X = spot_constants_pb2.JOINT_INDEX_A0_F1X

    # DOF count for strictly the legs.
    N_DOF_LEGS = 12
    # DOF count for all DOF on robot (arms and legs).
    N_DOF = 19

ORDERED_JOINT_NAMES_SPOT_BASE = [
    member.name.lower()
    for member in sorted(DOF, key=lambda x: x.value)
    if member.name not in ["N_DOF_LEGS", "N_DOF"] and not member.name.startswith("ARM_")
]

print("SPOT BASE", ORDERED_JOINT_NAMES_SPOT_BASE)
ORDERED_JOINT_NAMES_SPOT_ARM = [
    member.name.lower()
    for member in sorted(DOF, key=lambda x: x.value)
    if member.name.startswith("ARM_")
]


ORDERED_JOINT_NAMES_SPOT = ORDERED_JOINT_NAMES_SPOT_BASE + ORDERED_JOINT_NAMES_SPOT_ARM


# Default joint gains
DEFAULT_K_Q_P = [0] * DOF.N_DOF
DEFAULT_K_QD_P = [0] * DOF.N_DOF


def set_default_gains():
    # All legs have the same gains
    HX_K_Q_P = 624
    HX_K_QD_P = 5.20
    HY_K_Q_P = 936
    HY_K_QD_P = 5.20
    KN_K_Q_P = 286
    KN_K_QD_P = 2.04

    # Leg gains
    DEFAULT_K_Q_P[DOF.FL_HX] = HX_K_Q_P
    DEFAULT_K_QD_P[DOF.FL_HX] = HX_K_QD_P
    DEFAULT_K_Q_P[DOF.FL_HY] = HY_K_Q_P
    DEFAULT_K_QD_P[DOF.FL_HY] = HY_K_QD_P
    DEFAULT_K_Q_P[DOF.FL_KN] = KN_K_Q_P
    DEFAULT_K_QD_P[DOF.FL_KN] = KN_K_QD_P
    DEFAULT_K_Q_P[DOF.FR_HX] = HX_K_Q_P
    DEFAULT_K_QD_P[DOF.FR_HX] = HX_K_QD_P
    DEFAULT_K_Q_P[DOF.FR_HY] = HY_K_Q_P
    DEFAULT_K_QD_P[DOF.FR_HY] = HY_K_QD_P
    DEFAULT_K_Q_P[DOF.FR_KN] = KN_K_Q_P
    DEFAULT_K_QD_P[DOF.FR_KN] = KN_K_QD_P
    DEFAULT_K_Q_P[DOF.HL_HX] = HX_K_Q_P
    DEFAULT_K_QD_P[DOF.HL_HX] = HX_K_QD_P
    DEFAULT_K_Q_P[DOF.HL_HY] = HY_K_Q_P
    DEFAULT_K_QD_P[DOF.HL_HY] = HY_K_QD_P
    DEFAULT_K_Q_P[DOF.HL_KN] = KN_K_Q_P
    DEFAULT_K_QD_P[DOF.HL_KN] = KN_K_QD_P
    DEFAULT_K_Q_P[DOF.HR_HX] = HX_K_Q_P
    DEFAULT_K_QD_P[DOF.HR_HX] = HX_K_QD_P
    DEFAULT_K_Q_P[DOF.HR_HY] = HY_K_Q_P
    DEFAULT_K_QD_P[DOF.HR_HY] = HY_K_QD_P
    DEFAULT_K_Q_P[DOF.HR_KN] = KN_K_Q_P
    DEFAULT_K_QD_P[DOF.HR_KN] = KN_K_QD_P

    # Arm gains
    DEFAULT_K_Q_P[DOF.ARM_SH0] = 1020
    DEFAULT_K_QD_P[DOF.ARM_SH0] = 10.2
    DEFAULT_K_Q_P[DOF.ARM_SH1] = 255
    DEFAULT_K_QD_P[DOF.ARM_SH1] = 15.3
    DEFAULT_K_Q_P[DOF.ARM_EL0] = 204
    DEFAULT_K_QD_P[DOF.ARM_EL0] = 10.2
    DEFAULT_K_Q_P[DOF.ARM_EL1] = 102
    DEFAULT_K_QD_P[DOF.ARM_EL1] = 2.04
    DEFAULT_K_Q_P[DOF.ARM_WR0] = 102
    DEFAULT_K_QD_P[DOF.ARM_WR0] = 2.04
    DEFAULT_K_Q_P[DOF.ARM_WR1] = 102
    DEFAULT_K_QD_P[DOF.ARM_WR1] = 2.04
    DEFAULT_K_Q_P[DOF.ARM_F1X] = 16.0
    DEFAULT_K_QD_P[DOF.ARM_F1X] = 0.32


# Initialize default gains
set_default_gains()


JOINT_LIMITS = parse_urdf_limits(os.path.join(os.path.dirname(__file__), "..", "spot", "spot_description", "urdf", "spot.urdf"))

JOINT_SOFT_LIMITS = {
"fl_hx": (0.541525, 0.629967),
"fl_hy": (0.244991, 0.450403),
"fl_kn": (0.283921, 1.0),
"fr_hx": (0.667273, 0.500081),
"fr_hy": (0.217532, 0.357181),
"fr_kn": (0.425193, 1.0),
"hl_hx": (0.477865, 0.600584),
"hl_hy": (0.325119, 0.623534),
"hl_kn": (0.442601, 0.860269),
"hr_hx": (0.656397, 0.483916),
"hr_hy": (0.384144, 0.603661),
"hr_kn": (0.195063, 0.865289),
}


"""
Let's say my joint goes from -1 to 1.

I want to safely shrink this range by 10% margin each side.

In another words, the full range is 2, and I want to shrink it to 1.8.

This means that the new range is [-0.9, 0.9].

The inverse math is the following:

new_range = old_range * (1 - margin)
old_range = new_range / (1 - margin)

In the case that the margin is not centered in 0, let's say, from 2 to 4, I want to shrink it to 1.8.

This means that the new range is [2.2, 3.8].

The inverse math is the following:

new_range = old_range * (1 - margin)
old_range = new_range / (1 - margin)

0.
"""