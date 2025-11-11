"""
Beyond Mimic 配置文件
存放路径、常量、关节顺序等配置参数
"""

# 路径配置
LOG_SAVE_PATH = "./logs/BM_log.npz"
MODEL_PATH = "/home/ubuntu/deploy_mini/model/BM_ONNX/dalafan_woyaw.onnx"
MOTION_REF_PATH = "/home/ubuntu/deploy_mini/deploy_mujoco/bm_traj_npz/dalafan.npz"
XML_PATH = '/home/ubuntu/deploy_mini/resources/robots/g1_description/g1_29dof_zy.xml'

# 机器人配置
BODY_NAME = "torso_link"  # robot_ref_body_index=3 motion_ref_body_index=7

# 关节顺序 (XML中的关节顺序)
JOINT_XML = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
]

# 仿真参数
NUM_ACTIONS = 29
NUM_OBS = 148  # 154# with yaw
DEFAULT_NUM_ENVS = 4
CONTROL_DECIMATION = 10
SIMULATION_DT = 0.002
SIMULATION_DURATION = 300.0

# Logger 参数
LOGGER_DT = 0.02

# 环境初始化参数
ENV_X_OFFSET = 0.5  # 每个环境在x方向的位置错开
ENV_Y_OFFSET = 0.5  # 每个环境在y方向的位置错开
