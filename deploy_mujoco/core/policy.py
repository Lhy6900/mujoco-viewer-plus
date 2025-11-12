"""
策略模块
加载和运行 ONNX 策略模型
"""

import numpy as np
import onnxruntime
import onnx


class PolicyRunner:
    """策略推理器（支持单环境和多环境）"""
    
    def __init__(self, model_path, num_envs=1, policy_type="imitation"):
        """
        初始化策略推理器
        
        Args:
            model_path: ONNX 模型路径
            num_envs: 环境数量（为每个环境创建独立的 session）
            policy_type: 策略类型
                - 'imitation': 模仿学习（需要时间步）
                - 'locomotion': 运动控制（不需要时间步）
                - 'custom': 自定义策略
        """
        self.policy_type = policy_type
        self.model_path = model_path
        self.num_envs = num_envs
        
        # 加载 ONNX 模型
        self.onnx_model = onnx.load(model_path)
        
        # 创建多个推理 session（每个环境一个）
        self.sessions = [onnxruntime.InferenceSession(model_path) for _ in range(num_envs)]
        
        # 主 session（用于获取元数据）
        self.session = self.sessions[0] if num_envs > 0 else onnxruntime.InferenceSession(model_path)
        
        # 获取输入输出名称
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"[策略加载] ✓ 成功加载策略: {model_path}")
        print(f"[策略加载] → 环境数量: {num_envs}")
        print(f"[策略加载] → 类型: {policy_type}")
        print(f"[策略加载] → 输入: {self.input_names}")
        print(f"[策略加载] → 输出: {self.output_names}")
        
        # 从模型元数据读取配置
        self._load_metadata()
    
    def _load_metadata(self):
        """从 ONNX 模型元数据加载配置"""
        self.joint_names = None
        self.default_joint_pos = None
        self.joint_seq = None
        self.action_scale = None
        self.stiffness = None
        self.damping = None
        
        for prop in self.onnx_model.metadata_props:
            if prop.key == "joint_names":
                self.joint_seq = prop.value.split(",")
                self.joint_names = self.joint_seq
            if prop.key == "default_joint_pos":
                self.default_joint_pos = np.array([float(x) for x in prop.value.split(",")])
            if prop.key == "action_scale":
                self.action_scale = np.array([float(x) for x in prop.value.split(",")])
            if prop.key == "joint_stiffness":
                self.stiffness = np.array([float(x) for x in prop.value.split(",")])
            if prop.key == "joint_damping":
                self.damping = np.array([float(x) for x in prop.value.split(",")])
        
        if self.joint_names:
            print(f"[策略加载] → 关节数量: {len(self.joint_names)}")
    
    def compute_action(self, obs, timestep=None, output_names=None, env_idx=0):
        """
        计算单个环境的动作
        
        Args:
            obs: 观测 numpy 数组 (batch, obs_dim) 或 (obs_dim,)
            timestep: 时间步（模仿学习需要）
            output_names: 指定输出名称列表，默认返回 'actions'
            env_idx: 使用的环境索引（对应的 session）
            
        Returns:
            actions: numpy 数组
        """
        # 确保 obs 是 (1, obs_dim) 形状
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        # 构造输入字典
        input_dict = {'obs': obs.astype(np.float32)}
        
        # 模仿学习策略需要时间步
        if self.policy_type == "imitation" and timestep is not None:
            if isinstance(timestep, int):
                timestep = np.array([timestep], dtype=np.float32).reshape(1, 1)
            input_dict['time_step'] = timestep.astype(np.float32)
        
        # 推理
        if output_names is None:
            output_names = ['actions']
        
        try:
            session = self.sessions[env_idx] if env_idx < len(self.sessions) else self.session
            outputs = session.run(output_names, input_dict)
            
            # 如果只有一个输出，直接返回
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs
                
        except Exception as e:
            print(f"[策略推理] ✗ 推理失败 (env {env_idx}): {e}")
            print(f"[策略推理] → 输入形状: {[(k, v.shape) for k, v in input_dict.items()]}")
            raise
    
    def compute_actions_batch(self, obs_list, timestep_list=None, output_names=None):
        """
        批量计算多个环境的动作
        
        Args:
            obs_list: 观测列表，每个元素为 (obs_dim,) numpy 数组
            timestep_list: 时间步列表（模仿学习需要）
            output_names: 指定输出名称列表
            
        Returns:
            actions_list: 动作列表
        """
        if output_names is None:
            output_names = ['actions']
        
        actions_list = []
        for i, obs in enumerate(obs_list):
            timestep = timestep_list[i] if timestep_list is not None else None
            action = self.compute_action(obs, timestep, output_names, env_idx=i)
            actions_list.append(action)
        
        return actions_list
    
    def compute_ghost_outputs(self, obs, timestep, env_idx=0):
        """
        计算 Ghost 渲染所需的输出
        
        Args:
            obs: 观测
            timestep: 时间步
            env_idx: 环境索引
            
        Returns:
            tuple: (body_pos_w, body_quat_w)
        """
        if self.policy_type != "imitation":
            raise ValueError("Ghost 渲染仅支持模仿学习策略")
        
        outputs = self.compute_action(
            obs, 
            timestep, 
            output_names=['body_pos_w', 'body_quat_w'],
            env_idx=env_idx
        )
        return outputs[0], outputs[1]  # body_pos_w, body_quat_w
