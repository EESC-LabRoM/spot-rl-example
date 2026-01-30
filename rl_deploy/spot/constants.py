# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#bosdyn-api-CombinedJointStates
from enum import IntEnum

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
