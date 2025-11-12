"""
Beyond Mimic 配置文件
【用户主要修改区域】
所有路径、参数、奖励权重等配置
"""

import numpy as np

# ==================== 路径配置 ====================
MODEL_PATH = "/home/ubuntu/deploy_mini/model/BM_ONNX/dalafan_woyaw.onnx"
MOTION_REF_PATH = "/home/ubuntu/deploy_mini/deploy_mujoco/bm_traj_npz/dalafan.npz"
XML_PATH = '/home/ubuntu/deploy_mini/resources/robots/g1_description/g1_29dof_zy.xml'
LOG_SAVE_PATH = "./logs/BM_log.npz"

# ==================== 策略配置 ====================
POLICY_CONFIG = {
    'type': 'imitation',        # 'imitation' | 'locomotion' | 'custom'
    'enable_ghost': True,       # 是否启用 Ghost 渲染（需要参考运动数据）
}

# ==================== 环境配置 ====================
NUM_ENVS = 4                    # 环境数量
ENV_SPACING = 3.0               # 环境间距（米）
SIMULATION_DT = 0.002           # 仿真时间步长
CONTROL_DECIMATION = 10         # 控制降采样率
SIMULATION_DURATION = 300.0     # 仿真时长（秒）

# 兼容旧代码的常量
DEFAULT_NUM_ENVS = NUM_ENVS

# ==================== 观测配置 ====================
OBS_CONFIG = {
    'type': 'imitation',        # 必须与 POLICY_CONFIG['type'] 一致
    'dim': 148,                 # 观测维度
    'include_motion_ref': True,
    'include_action_history': True,
    'include_proprioception': True,
}

# 兼容旧代码的常量
NUM_OBS = OBS_CONFIG['dim']

# ==================== 动作配置 ====================
NUM_ACTIONS = 29

# ==================== 奖励配置 ====================
# 【用户自定义区域】启用/禁用奖励项，调整权重
REWARD_CONFIG = {
    'enabled': [
        'pos_tracking_global',
        'pos_tracking_local',
        'quat_tracking_global',
        'quat_tracking_local',
        'keypoint_tracking',
    ],
    'weights': {
        'pos_tracking_global': 0.5,
        'pos_tracking_local': 0.3,
        'quat_tracking_global': 0.1,
        'quat_tracking_local': 0.1,
        'keypoint_tracking': 0.1,
    },
    'enable_plotting': True,    # 是否显示奖励曲线
}

# ==================== 外力配置 ====================
FORCE_CONFIG = {
    'anchor_body': 'left_knee_link',  # 施加外力的 body 名称
    'mode': {
        'stop': 'keeping',      # 'fixtime': 固定时间 | 'keeping': 持续切换
        'style': 'spring',      # 'constant': 恒定力 | 'spring': 弹簧力
        'select': 0,            # None: 无 | 0: 仅主环境 | 1: 所有 | [0,2]: 列表
    },
    'constant': {
        'magnitude': 20.0,      # 恒定力大小（N）
        'direction': [0, 1, 0], # 恒定力方向（单位向量）
        'duration': 5.0,        # fixtime 模式的持续时间（秒）
    },
    'spring': {
        'k': 50.0,              # 弹簧系数
        'center_bias': [0.2, 0.2, 0.0],  # 引力中心偏置
    },
}

# ==================== 控制器配置 ====================
CONTROLLER_CONFIG = {
    'type': 'pd',
    'kp': 200.0,                # 比例增益
    'kd': 10.0,                 # 微分增益
}

# ==================== 可视化配置 ====================
VISUALIZATION_CONFIG = {
    'ghost': {
        'color': [0.5, 0.7, 0.5, 0.5],  # 绿色半透明
        'show_main': True,
        'show_others': False,
    },
    'force': {
        'color': [0.8, 0.0, 0.8, 0.6],  # 紫色
        'width': 0.015,
    },
}

# ==================== 日志配置 ====================
LOGGER_DT = 0.02

# ==================== 关节配置 ====================
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
