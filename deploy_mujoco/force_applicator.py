"""
外力施加器模块
用于在指定的 body 上施加定时外力
"""

import numpy as np
import mujoco
import time


class ForceApplicator:
    """
    外力施加器类
    支持在多个环境的指定 body 上施加定时外力
    """
    
    def __init__(self, model, body_name="pelvis", force_anchor_bodyidx=None, force_mode=None):
        """
        初始化外力施加器
        
        Args:
            model: mujoco.MjModel 对象
            body_name: 要施加外力的 body 名称，默认 "pelvis"（用于查找 body_id）
            force_anchor_bodyidx: 施加外力的锚点 body 索引，如果为 None 则使用 body_name 查找
            force_mode: 外力模式字典，支持的键：
                - 'stop': 停止模式，可选值：
                    * 'fixtime': 固定时间后自动停止（默认）
                    * 'keeping': 持续施加直到手动停止（再次按 Ctrl+F）
        """
        self.model = model
        self.body_name = body_name
        
        # 确定要施加外力的 body ID
        if force_anchor_bodyidx is not None:
            # 使用指定的 body 索引
            if force_anchor_bodyidx < 0 or force_anchor_bodyidx >= model.nbody:
                raise ValueError(f"force_anchor_bodyidx={force_anchor_bodyidx} 超出范围 [0, {model.nbody})")
            self.body_id = force_anchor_bodyidx
            # 从索引获取 body 名称
            self.body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, self.body_id)
            if self.body_name is None:
                self.body_name = f"body_{self.body_id}"
        else:
            # 从 body 名称查找 ID
            self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if self.body_id == -1:
                raise ValueError(f"Body '{body_name}' not found in model")
        
        # 外力模式配置
        if force_mode is None:
            self.force_mode = {'stop': 'fixtime'}  # 默认固定时间模式
        else:
            self.force_mode = force_mode.copy()
            # 验证 stop 模式
            if 'stop' not in self.force_mode:
                self.force_mode['stop'] = 'fixtime'
            elif self.force_mode['stop'] not in ['fixtime', 'keeping']:
                raise ValueError(f"Invalid force_mode['stop']='{self.force_mode['stop']}', must be 'fixtime' or 'keeping'")
        
        # 外力施加状态
        self.is_active = False
        self.start_time = 0.0
        self.duration = 5.0  # 默认持续时间 5 秒（仅在 fixtime 模式下使用）
        
        # 外力参数
        self.force_magnitude = 20.0  # N
        self.force_direction = np.array([0.0, 1.0, 0.0])  # Y 正方向
        
        print(f"[ForceApplicator] 已初始化，目标 body: {self.body_name} (ID: {self.body_id})")
        print(f"  - 停止模式: {self.force_mode['stop']}")

    
    def trigger(self, duration=5.0, force_magnitude=20.0, force_direction=None, data_list=None):
        """
        触发外力施加（或在 keeping 模式下切换开/关）
        
        Args:
            duration: 持续时间（秒），仅在 fixtime 模式下使用
            force_magnitude: 力的大小（N）
            force_direction: 力的方向（归一化的 3D 向量），默认 Y 正方向
            data_list: mujoco.MjData 对象列表（用于 keeping 模式切换时清除外力）
        """
        # 在 keeping 模式下，如果已经激活，则切换为停止
        if self.force_mode['stop'] == 'keeping' and self.is_active:
            if data_list is not None:
                self.stop(data_list)
            else:
                self.is_active = False
                print(f"[ForceApplicator] 外力施加已停止（keeping 模式）")
            return
        
        # 激活外力
        self.is_active = True
        self.start_time = time.time()
        self.duration = duration
        self.force_magnitude = force_magnitude
        
        if force_direction is not None:
            # 归一化方向向量
            norm = np.linalg.norm(force_direction)
            if norm > 1e-6:
                self.force_direction = force_direction / norm
            else:
                self.force_direction = np.array([0.0, 1.0, 0.0])
        
        print(f"[ForceApplicator] 触发外力施加：")
        print(f"  - 模式: {self.force_mode['stop']}")
        if self.force_mode['stop'] == 'fixtime':
            print(f"  - 持续时间: {duration:.1f}s")
        else:
            print(f"  - 持续时间: 持续施加直到再次按 Ctrl+F")
        print(f"  - 力大小: {force_magnitude:.1f}N")
        print(f"  - 力方向: {self.force_direction}")
    
    def update(self, data_list):
        """
        更新外力施加状态，并应用到所有环境
        
        Args:
            data_list: mujoco.MjData 对象列表（多环境）
        
        Returns:
            is_active: 是否仍在施加外力
        """
        if not self.is_active:
            return False
        
        # 在 fixtime 模式下检查是否超时
        if self.force_mode['stop'] == 'fixtime':
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.duration:
                self.stop(data_list)
                return False
        
        # 计算外力向量
        force_vector = self.force_direction * self.force_magnitude
        
        # 对所有环境施加外力
        for data in data_list:
            # xfrc_applied 的形状是 (nbody, 6)
            # 前3个是力 (force)，后3个是力矩 (torque)
            data.xfrc_applied[self.body_id, 0:3] = force_vector
            data.xfrc_applied[self.body_id, 3:6] = 0.0  # 不施加力矩
        
        return True
    
    def stop(self, data_list):
        """
        停止外力施加，清除所有环境的外力
        
        Args:
            data_list: mujoco.MjData 对象列表（多环境）
        """
        if not self.is_active:
            return
        
        self.is_active = False
        
        # 清除所有环境的外力
        for data in data_list:
            data.xfrc_applied[self.body_id, :] = 0.0
        
        print(f"[ForceApplicator] 外力施加结束")
    
    def get_force_info(self):
        """
        获取当前外力信息（用于可视化）
        
        Returns:
            dict: 包含 body_id, force_vector, is_active, remaining_time 等信息
                  如果未激活，返回 None
        """
        if not self.is_active:
            return None
        
        elapsed_time = time.time() - self.start_time
        force_vector = self.force_direction * self.force_magnitude
        
        # 根据模式计算剩余时间
        if self.force_mode['stop'] == 'fixtime':
            remaining_time = max(0.0, self.duration - elapsed_time)
        else:  # keeping 模式
            remaining_time = float('inf')  # 无限持续
        
        return {
            'body_id': self.body_id,
            'body_name': self.body_name,
            'force_vector': force_vector,
            'is_active': self.is_active,
            'elapsed_time': elapsed_time,
            'remaining_time': remaining_time,
            'duration': self.duration if self.force_mode['stop'] == 'fixtime' else None,
            'mode': self.force_mode['stop']
        }
