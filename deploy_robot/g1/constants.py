from enum import IntEnum

G1_NUM_MOTOR = 29
class G1JointIndex(IntEnum):
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21  # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28  # NOTE: INVALID for g1 23dof

# G1JointNames as strings in the same order as G1JointIndex
G1JointNames = [
    "left_hip_pitch_joint",   # 0 LeftHipPitch
    "left_hip_roll_joint",    # 1 LeftHipRoll
    "left_hip_yaw_joint",     # 2 LeftHipYaw
    "left_knee_joint",        # 3 LeftKnee
    "left_ankle_pitch_joint", # 4 LeftAnklePitch
    "left_ankle_roll_joint",  # 5 LeftAnkleRoll
    "right_hip_pitch_joint",  # 6 RightHipPitch
    "right_hip_roll_joint",   # 7 RightHipRoll
    "right_hip_yaw_joint",    # 8 RightHipYaw
    "right_knee_joint",       # 9 RightKnee
    "right_ankle_pitch_joint",#10 RightAnklePitch
    "right_ankle_roll_joint", #11 RightAnkleRoll
    "waist_yaw_joint",        #12 WaistYaw
    "waist_roll_joint",       #13 WaistRoll
    "waist_pitch_joint",      #14 WaistPitch
    "left_shoulder_pitch_joint",  #15 LeftShoulderPitch
    "left_shoulder_roll_joint",   #16 LeftShoulderRoll
    "left_shoulder_yaw_joint",    #17 LeftShoulderYaw
    "left_elbow_joint",           #18 LeftElbow
    "left_wrist_roll_joint",      #19 LeftWristRoll
    "left_wrist_pitch_joint",     #20 LeftWristPitch
    "left_wrist_yaw_joint",       #21 LeftWristYaw
    "right_shoulder_pitch_joint", #22 RightShoulderPitch
    "right_shoulder_roll_joint",  #23 RightShoulderRoll
    "right_shoulder_yaw_joint",   #24 RightShoulderYaw
    "right_elbow_joint",          #25 RightElbow
    "right_wrist_roll_joint",     #26 RightWristRoll
    "right_wrist_pitch_joint",    #27 RightWristPitch
    "right_wrist_yaw_joint",      #28 RightWristYaw
]

class G1JointArmIndex(IntEnum):
    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

G1JointArmNames = [
    "left_shoulder_pitch_joint",  #15 LeftShoulderPitch
    "left_shoulder_roll_joint",   #16 LeftShoulderRoll
    "left_shoulder_yaw_joint",    #17 LeftShoulderYaw
    "left_elbow_joint",           #18 LeftElbow
    "left_wrist_roll_joint",      #19 LeftWristRoll
    "left_wrist_pitch_joint",     #20 LeftWristPitch
    "left_wrist_yaw_joint",       #21 LeftWristYaw
    "right_shoulder_pitch_joint", #22 RightShoulderPitch
    "right_shoulder_roll_joint",  #23 RightShoulderRoll
    "right_shoulder_yaw_joint",   #24 RightShoulderYaw
    "right_elbow_joint",          #25 RightElbow
    "right_wrist_roll_joint",     #26 RightWristRoll
    "right_wrist_pitch_joint",    #27 RightWristPitch
    "right_wrist_yaw_joint",      #28 RightWristYaw
]

class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

def is_weak_motor(motor_index) -> bool:
    """
    Return True if motor_index corresponds to a 'weak' motor.
    motor_index may be an IntEnum member or an int.
    """
    idx = int(motor_index)
    weak_motors = {
        G1JointIndex.LeftAnklePitch.value,
        G1JointIndex.RightAnklePitch.value,
        # Left arm
        G1JointIndex.LeftShoulderPitch.value,
        G1JointIndex.LeftShoulderRoll.value,
        G1JointIndex.LeftShoulderYaw.value,
        G1JointIndex.LeftElbow.value,
        # Right arm
        G1JointIndex.RightShoulderPitch.value,
        G1JointIndex.RightShoulderRoll.value,
        G1JointIndex.RightShoulderYaw.value,
        G1JointIndex.RightElbow.value,
    }
    return idx in weak_motors


def is_wrist_motor(motor_index) -> bool:
    """
    Return True if motor_index corresponds to a wrist motor.
    motor_index may be an IntEnum member or an int.
    """
    idx = int(motor_index)
    wrist_motors = {
        G1JointIndex.LeftWristRoll.value,
        G1JointIndex.LeftWristPitch.value,
        G1JointIndex.LeftWristYaw.value,
        G1JointIndex.RightWristRoll.value,
        G1JointIndex.RightWristPitch.value,
        G1JointIndex.RightWristYaw.value,
    }
    return idx in wrist_motors