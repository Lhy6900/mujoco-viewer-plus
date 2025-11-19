"""
观测构造模块
根据策略类型构造不同的观测向量
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

class ObservationBuilder:
    """观测构造器"""
    
    def __init__(self, obs_config, joint_mapping=None):
        """
        初始化观测构造器
        
        Args:
            obs_config: 观测配置字典
                {
                    'type': 'imitation' | 'locomotion' | 'custom',
                    'dim': int,  # 观测维度
                    'include_motion_ref': bool,
                    'include_action_history': bool,
                    'include_proprioception': bool,
                }
            joint_mapping: 关节映射字典（可选）
        """
        self.config = obs_config
        self.obs_type = obs_config.get('type', 'imitation')
        self.obs_dim = obs_config.get('dim', 148)
        self.joint_mapping = joint_mapping
        
        print(f"[观测构造器] 类型: {self.obs_type}, 维度: {self.obs_dim}")
    
    def build(self, data, motion_data=None, action_buffer=None, timestep=0, idx=0):
        """
        【用户自定义区域】构造观测
        
        Args:
            data: mujoco.MjData 对象
            motion_data: 参考运动数据（模仿学习需要）
            action_buffer: 上一步动作
            timestep: 当前时间步
            
        Returns:
            obs: numpy 数组 (obs_dim,)
        """
        if self.obs_type == 'imitation':
            return self._build_imitation(data, motion_data, action_buffer, timestep)
        elif self.obs_type == 'complianceimitation':
            return self._build_complianceimitation(data, motion_data, action_buffer, timestep, idx)
        elif self.obs_type == 'locomotion':
            return self._build_locomotion(data, action_buffer)
        elif self.obs_type == 'custom':
            return self._build_custom(data)
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")
    
    def _build_imitation(self, data, motion_data, action_buffer, timestep):
        """
        【用户自定义区域】模仿学习观测
        
        当前实现：
        - [0:58]: 参考运动输入（关节位置 + 速度）
        - [58:61]: 机器人角速度
        - [61:90]: 关节位置差（当前 - 默认）
        - [90:119]: 关节速度
        - [119:148]: 上一步动作
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # 如果没有参考运动数据，使用零向量填充
        if motion_data is not None and 'joint_pos' in motion_data and 'joint_vel' in motion_data:
            # 参考运动输入
            motion_input = np.concatenate((
                motion_data['joint_pos'][timestep, :],
                motion_data['joint_vel'][timestep, :]
            ), axis=0)
            obs[0:58] = motion_input
        else:
            # 无参考运动时使用零向量
            obs[0:58] = np.zeros(58, dtype=np.float32)
        
        # 角速度
        obs[58:61] = data.qvel[3:6]
        
        # 关节位置和速度（需要关节映射）
        if self.joint_mapping is not None:
            joint_seq = self.joint_mapping['joint_seq']
            joint_xml = self.joint_mapping['joint_xml']
            
            # 获取默认关节位置（如果有）
            default_joint_pos = self.joint_mapping.get('default_joint_pos_seq', np.zeros(len(joint_seq)))
            
            # 关节位置（XML 顺序 -> 策略顺序）
            qpos_xml = data.qpos[7:7 + len(joint_xml)]
            qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
            obs[61:90] = qpos_seq - default_joint_pos
            
            # 关节速度（XML 顺序 -> 策略顺序）
            qvel_xml = data.qvel[6:6 + len(joint_xml)]
            qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
            obs[90:119] = qvel_seq
        
        # 动作历史
        if action_buffer is not None:
            obs[119:148] = action_buffer
        
        return obs
    
    def _build_complianceimitation(self, data, motion_data, action_buffer, timestep, idx):
        """
        【用户自定义区域】模仿学习观测
        
        当前实现：
        - [0:62]: 参考运动输入（关节位置 + 速度）
        - [62:65]: 机器人重力投影
        - [65:68]: 机器人角速度
        - [68:97]: 关节位置差（当前 - 默认）
        - [97:126]: 关节速度
        - [126:155]: 上一步动作
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # 如果没有参考运动数据，使用零向量填充
        if motion_data is not None and 'joint_pos' in motion_data and 'joint_vel' in motion_data:
            # 参考运动输入 --正常跳舞
            motion_input = np.concatenate((
                motion_data['joint_pos'][timestep, :],
                motion_data['joint_vel'][timestep, :]
            ), axis=0)
            # 参考运动输入 --正常跳舞结束
            # # 参考运动输入 --- 原地跳舞 ---
            # joint_pos_timestep = motion_data['joint_pos'][timestep, :]
            # joint_vel_timestep = motion_data['joint_vel'][timestep, :]
            # joint_pos_zero = motion_data['joint_pos'][0, :]
            # joint_vel_zero = motion_data['joint_vel'][0, :]
            # # 将部分关节替换为初始帧（原地）值，得到混合参考位姿
            # # 仅替换下半身 + PELVIS + TORSO 索引（参考 bmconfig 注释）
            # replace_idx = np.array([0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18], dtype=int)
            # joint_pos_mixed = joint_pos_timestep.copy()
            # joint_vel_mixed = joint_vel_timestep.copy()
            # # 防护：确保索引不越界
            # valid_idx = replace_idx[replace_idx < joint_pos_mixed.size]
            # if valid_idx.size > 0:
            #     joint_pos_mixed[valid_idx] = joint_pos_zero[valid_idx]
            #     joint_vel_mixed[valid_idx] = joint_vel_zero[valid_idx]

            # # 使用混合后的关节位置与当前步的关节速度作为参考输入
            # motion_input = np.concatenate((
            #     joint_pos_mixed,
            #     joint_vel_mixed
            # ), axis=0)
            # # 参考运动输入 --- 原地跳舞结束 ---
            obs[0:58] = motion_input
        else:
            # 无参考运动时使用零向量
            obs[0:58] = np.zeros(58, dtype=np.float32)
        # 自定义柔顺性命令
        obs[58:62] = np.zeros(4, dtype=np.float32)
        if idx % 3 ==1:
            obs[61] = 1.0
            obs[59] = 20.0
        elif idx %3 ==2:
            obs[61] = 1.0   
        obs[62:65] = self.project_gravity_to_world(data)
        # 角速度
        obs[65:68] = data.qvel[3:6]
    # debug prints removed
        # 关节位置和速度（需要关节映射）
        if self.joint_mapping is not None:
            joint_seq = self.joint_mapping['joint_seq']
            joint_xml = self.joint_mapping['joint_xml']
            
            # 获取默认关节位置（如果有）
            default_joint_pos = self.joint_mapping.get('default_joint_pos_seq', np.zeros(len(joint_seq)))
            
            # 关节位置（XML 顺序 -> 策略顺序）
            qpos_xml = data.qpos[7:7 + len(joint_xml)]
            qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
            obs[68:97] = qpos_seq - default_joint_pos
            
            # 关节速度（XML 顺序 -> 策略顺序）
            qvel_xml = data.qvel[6:6 + len(joint_xml)]
            qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
            obs[97:126] = qvel_seq
        
        # 动作历史
        if action_buffer is not None:
            obs[126:155] = action_buffer
        
        return obs

    def _build_custom(self, data):
        """
        【用户自定义区域】自定义观测
        用户可以在这里实现自己的观测构造逻辑
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        # TODO: 用户自定义实现
        return obs
    
    def set_joint_mapping(self, joint_seq, joint_xml, default_joint_pos_seq=None):
        """
        设置关节映射
        
        Args:
            joint_seq: 策略中的关节顺序
            joint_xml: XML 中的关节顺序
            default_joint_pos_seq: 默认关节位置（策略顺序）
        """
        self.joint_mapping = {
            'joint_seq': joint_seq,
            'joint_xml': joint_xml
        }
        if default_joint_pos_seq is not None:
            self.joint_mapping['default_joint_pos_seq'] = default_joint_pos_seq

    def project_gravity_to_world(self, data):
        """
        计算重力向量在机体坐标系中的表示，使用scipy的Rotation
        data: mujoco.MjData对象，包含机器人状态
        返回: 重力向量在机体坐标系中的表示
        """
        # MuJoCo中四元数的顺序是[w, x, y, z]，而scipy.Rotation需要[x, y, z, w]
        quat = np.array([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])  # 转换为[x, y, z, w]格式
        
        # 使用scipy的Rotation计算旋转矩阵
        rot_mat = R.from_quat(quat).as_matrix()
        
        # 计算重力向量在机体坐标系中的表示
        base_proj_gravity = np.matmul(rot_mat.T, np.array([0, 0, -1.0]))
        
        return base_proj_gravity