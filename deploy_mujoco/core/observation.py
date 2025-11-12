"""
观测构造模块
根据策略类型构造不同的观测向量
"""

import numpy as np
import torch


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
    
    def build(self, data, motion_data=None, action_buffer=None, timestep=0):
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
