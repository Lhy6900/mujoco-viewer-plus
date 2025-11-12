"""
环境管理模块
管理多个 MuJoCo 环境的创建、重置、步进
"""

import numpy as np
import mujoco


class EnvironmentManager:
    """管理多环境仿真"""
    
    def __init__(self, model, num_envs, env_spacing=3.0):
        """
        初始化多环境管理器
        
        Args:
            model: mujoco.MjModel 对象
            num_envs: 环境数量
            env_spacing: 环境间距（米）
        """
        self.model = model
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        
        # 创建多个 MjData 实例
        self.data_list = [mujoco.MjData(model) for _ in range(num_envs)]
        
        # 计算网格布局
        self.env_origins = self._compute_grid_layout(num_envs, env_spacing)
        
        print(f"[环境管理器] 已创建 {num_envs} 个环境")
        print(f"[环境管理器] 网格布局: {self._get_grid_dimensions(num_envs)}")
        
    def _get_grid_dimensions(self, num_envs):
        """获取网格维度"""
        num_rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
        num_cols = int(np.ceil(num_envs / num_rows))
        return f"{num_rows} 行 × {num_cols} 列"
    
    def _compute_grid_layout(self, num_envs, spacing):
        """
        计算网格布局的环境原点位置
        
        Args:
            num_envs: 环境数量
            spacing: 环境间距
            
        Returns:
            env_origins: (num_envs, 2) 数组，每行为一个环境的 (x, y) 坐标
        """
        env_origins = np.zeros((num_envs, 2))
        
        # 计算网格行列数：尽可能接近正方形
        num_rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
        num_cols = int(np.ceil(num_envs / num_rows))
        
        # 生成网格索引
        ii, jj = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
        
        # 计算每个环境的位置（居中布局）
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * spacing  # x
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * spacing   # y
        
        return env_origins
    
    def reset_env(self, env_idx, qpos=None, qvel=None):
        """
        重置指定环境
        
        Args:
            env_idx: 环境索引
            qpos: 可选的初始位置
            qvel: 可选的初始速度
        """
        if qpos is not None:
            self.data_list[env_idx].qpos[:] = qpos
        else:
            self.data_list[env_idx].qpos[:] = self.model.qpos0
            
        if qvel is not None:
            self.data_list[env_idx].qvel[:] = qvel
        else:
            self.data_list[env_idx].qvel[:] = 0
        
        # 应用环境偏移
        self.data_list[env_idx].qpos[0] += self.env_origins[env_idx, 0]  # x
        self.data_list[env_idx].qpos[1] += self.env_origins[env_idx, 1]  # y
        
        mujoco.mj_forward(self.model, self.data_list[env_idx])
    
    def reset_all(self, qpos=None, qvel=None):
        """重置所有环境"""
        for i in range(self.num_envs):
            self.reset_env(i, qpos, qvel)
    
    def step_env(self, env_idx, ctrl=None):
        """
        单个环境步进一次
        
        Args:
            env_idx: 环境索引
            ctrl: 控制输入
        """
        if ctrl is not None:
            self.data_list[env_idx].ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data_list[env_idx])
    
    def step_all(self, ctrl_list=None):
        """
        所有环境步进一次
        
        Args:
            ctrl_list: 控制输入列表，长度为 num_envs
        """
        if ctrl_list is None:
            for i in range(self.num_envs):
                mujoco.mj_step(self.model, self.data_list[i])
        else:
            for i, ctrl in enumerate(ctrl_list):
                self.data_list[i].ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data_list[i])
    
    def get_state(self, env_idx):
        """
        获取指定环境的状态
        
        Returns:
            dict: 包含 qpos, qvel, xpos 等状态信息
        """
        data = self.data_list[env_idx]
        return {
            'qpos': data.qpos.copy(),
            'qvel': data.qvel.copy(),
            'xpos': data.xpos.copy(),
            'xquat': data.xquat.copy(),
        }
    
    def get_all_states(self):
        """获取所有环境的状态"""
        return [self.get_state(i) for i in range(self.num_envs)]
    
    def sync_to_main_data(self, main_data, env_idx):
        """
        将指定环境的数据同步到主数据对象（用于 viewer）
        
        Args:
            main_data: mujoco.MjData 主数据对象
            env_idx: 要同步的环境索引
        """
        main_data.qpos[:] = self.data_list[env_idx].qpos[:]
        main_data.qvel[:] = self.data_list[env_idx].qvel[:]
        main_data.ctrl[:] = self.data_list[env_idx].ctrl[:]
        mujoco.mj_forward(self.model, main_data)
    
    def initialize_positions(self, joint_pos_array, joint_xml):
        """
        初始化所有环境的关节位置
        
        Args:
            joint_pos_array: 默认关节位置数组
            joint_xml: 关节顺序列表
        """
        for i in range(self.num_envs):
            self.data_list[i].qpos[7:] = joint_pos_array
            # 应用环境偏移
            self.data_list[i].qpos[0] = self.env_origins[i, 0]  # x
            self.data_list[i].qpos[1] = self.env_origins[i, 1]  # y
