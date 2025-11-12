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
                - 'style': 力的施加方式，可选值：
                    * 'constant': 恒定力，大小和方向固定（默认）
                    * 'spring': 弹簧力，引力中心固定，力大小和方向动态计算
                - 'select': 选择哪些环境施加外力，可选值：
                    * None: 不对任何环境施加外力
                    * 0: 仅对第0个环境施加外力
                    * 1: 对所有环境施加外力（默认）
                    * [0, 2, 3]: 对列表中索引对应的环境施加外力
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
            self.force_mode = {'stop': 'fixtime', 'style': 'constant', 'select': 1}  # 默认模式
        else:
            self.force_mode = force_mode.copy()
            # 验证 stop 模式
            if 'stop' not in self.force_mode:
                self.force_mode['stop'] = 'fixtime'
            elif self.force_mode['stop'] not in ['fixtime', 'keeping']:
                raise ValueError(f"Invalid force_mode['stop']='{self.force_mode['stop']}', must be 'fixtime' or 'keeping'")
            # 验证 style 模式
            if 'style' not in self.force_mode:
                self.force_mode['style'] = 'constant'
            elif self.force_mode['style'] not in ['constant', 'spring']:
                raise ValueError(f"Invalid force_mode['style']='{self.force_mode['style']}', must be 'constant' or 'spring'")
            # 设置 select 默认值（不在这里验证，因为还不知道环境数量）
            if 'select' not in self.force_mode:
                self.force_mode['select'] = 1  # 默认所有环境
        
        # 外力施加状态
        self.is_active = False
        self.start_time = 0.0
        self.duration = 5.0  # 默认持续时间 5 秒（仅在 fixtime 模式下使用）
        
        # 恒定力参数（constant 模式）
        self.force_magnitude = 20.0  # N
        self.force_direction = np.array([0.0, 1.0, 0.0])  # Y 正方向
        
        # 弹簧力参数（spring 模式）
        self.spring_k = 50.0  # 弹簧系数 K
        self.spring_center_bias = np.array([0.2, 0.2, 0.0])  # 引力中心偏置
        self.spring_centers = []  # 每个环境的引力中心 (num_envs, 3)，在 trigger 时初始化
        
        # 选中的环境列表（根据 select 参数计算）
        self.selected_envs = []  # 当前选中要施加外力的环境索引列表
        
        print(f"[ForceApplicator] 已初始化，目标 body: {self.body_name} (ID: {self.body_id})")
        print(f"  - 停止模式: {self.force_mode['stop']}")
        print(f"  - 力的风格: {self.force_mode['style']}")
        print(f"  - 环境选择: {self.force_mode['select']}")
        if self.force_mode['style'] == 'spring':
            print(f"  - 弹簧系数 K: {self.spring_k}")
            print(f"  - 引力中心偏置: {self.spring_center_bias}")
    
    def _parse_select(self, select_param, num_envs):
        """
        解析 select 参数，返回选中的环境索引列表
        
        Args:
            select_param: select 参数值（None, 0, 1, 或列表）
            num_envs: 环境总数
            
        Returns:
            list: 选中的环境索引列表，例如 [0, 2, 3]
            
        Raises:
            ValueError: 如果参数不合法
        """
        # 情况1: None - 不施加外力
        if select_param is None:
            return []
        
        # 情况2: 0 - 仅第0个环境
        if select_param == 0:
            if num_envs == 0:
                raise ValueError("Cannot select environment 0: num_envs is 0")
            return [0]
        
        # 情况3: 1 - 所有环境
        if select_param == 1:
            return list(range(num_envs))
        
        # 情况4: 列表 - 指定环境索引
        if isinstance(select_param, (list, tuple)):
            # 检查是否全是整数
            if not all(isinstance(i, int) for i in select_param):
                raise ValueError(f"force_mode['select'] 列表必须全是整数，当前值: {select_param}")
            
            # 检查索引范围
            for idx in select_param:
                if idx < 0 or idx >= num_envs:
                    raise ValueError(
                        f"force_mode['select'] 列表中索引 {idx} 超出范围 [0, {num_envs})"
                    )
            
            # 去重并排序
            selected = sorted(set(select_param))
            
            # 如果有重复，给出提示
            if len(selected) < len(select_param):
                print(f"[警告] force_mode['select'] 列表中有重复索引，已自动去重: {select_param} -> {selected}")
            
            return selected
        
        # 其他情况：不合法的参数类型
        raise ValueError(
            f"Invalid force_mode['select']={select_param}, "
            f"must be None, 0, 1, or a list of integers"
        )
    
    def trigger(self, duration=5.0, force_magnitude=20.0, force_direction=None, data_list=None):
        """
        触发外力施加（或在 keeping 模式下切换开/关）
        
        Args:
            duration: 持续时间（秒），仅在 fixtime 模式下使用
            force_magnitude: 力的大小（N），仅在 constant 模式下使用
            force_direction: 力的方向（归一化的 3D 向量），仅在 constant 模式下使用，默认 Y 正方向
            data_list: mujoco.MjData 对象列表（用于 keeping 模式切换和 spring 模式初始化）
        """
        # ===== 新增：解析 select 参数，确定要施加外力的环境列表 =====
        if data_list is not None:
            try:
                self.selected_envs = self._parse_select(self.force_mode['select'], len(data_list))
                if not self.selected_envs:
                    print(f"[ForceApplicator] force_mode['select']={self.force_mode['select']}，不对任何环境施加外力")
                    return
                print(f"[ForceApplicator] 将对以下环境施加外力: {self.selected_envs}")
            except ValueError as e:
                print(f"[错误] 解析 force_mode['select'] 失败: {e}")
                return
        else:
            # 如果没有 data_list，则默认使用之前的 selected_envs（向后兼容）
            if not hasattr(self, 'selected_envs') or not self.selected_envs:
                print(f"[警告] trigger() 未提供 data_list，无法确定环境列表")
                return
        # ===== select 参数解析结束 =====
        
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
        
        # constant 模式：设置恒定力的方向
        if self.force_mode['style'] == 'constant':
            if force_direction is not None:
                # 归一化方向向量
                norm = np.linalg.norm(force_direction)
                if norm > 1e-6:
                    self.force_direction = force_direction / norm
                else:
                    self.force_direction = np.array([0.0, 1.0, 0.0])
        
        # spring 模式：计算选中环境的引力中心
        elif self.force_mode['style'] == 'spring':
            if data_list is None:
                raise ValueError("spring 模式需要提供 data_list 来计算引力中心")
            
            # 只计算选中环境的引力中心
            self.spring_centers = []
            for env_i in self.selected_envs:
                data = data_list[env_i]
                # 获取当前环境中 body 的全局位置
                body_pos = data.xpos[self.body_id].copy()
                # 计算引力中心 = body 位置 + 偏置
                spring_center = body_pos + self.spring_center_bias
                self.spring_centers.append(spring_center)
            
            print(f"[ForceApplicator] Spring 模式：已计算 {len(self.spring_centers)} 个环境的引力中心")
            if len(self.spring_centers) > 0:
                print(f"  - 环境 0 引力中心: {self.spring_centers[0]}")
        
        # 打印触发信息
        print(f"[ForceApplicator] 触发外力施加：")
        print(f"  - 停止模式: {self.force_mode['stop']}")
        print(f"  - 力的风格: {self.force_mode['style']}")
        
        if self.force_mode['stop'] == 'fixtime':
            print(f"  - 持续时间: {duration:.1f}s")
        else:
            print(f"  - 持续时间: 持续施加直到再次按 Ctrl+F")
        
        if self.force_mode['style'] == 'constant':
            print(f"  - 力大小: {force_magnitude:.1f}N")
            print(f"  - 力方向: {self.force_direction}")
        else:  # spring
            print(f"  - 弹簧系数: {self.spring_k}")
            print(f"  - 引力中心偏置: {self.spring_center_bias}")
    
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
        
        # 根据 style 模式计算并施加外力
        if self.force_mode['style'] == 'constant':
            # 恒定力模式：选中的环境施加相同的力
            force_vector = self.force_direction * self.force_magnitude
            
            # 只对选中的环境施加外力
            for env_i in self.selected_envs:
                if env_i < len(data_list):
                    data = data_list[env_i]
                    # xfrc_applied 的形状是 (nbody, 6)
                    # 前3个是力 (force)，后3个是力矩 (torque)
                    data.xfrc_applied[self.body_id, 0:3] = force_vector
                    data.xfrc_applied[self.body_id, 3:6] = 0.0  # 不施加力矩
        
        elif self.force_mode['style'] == 'spring':
            # 弹簧力模式：每个选中环境单独计算力
            for spring_i, env_i in enumerate(self.selected_envs):
                if env_i >= len(data_list):
                    continue
                
                data = data_list[env_i]
                # 获取当前 body 位置
                body_pos = data.xpos[self.body_id]
                
                # 获取该环境的引力中心（使用 spring_i 索引到 spring_centers）
                if spring_i < len(self.spring_centers):
                    spring_center = self.spring_centers[spring_i]
                else:
                    # 如果没有预先计算的引力中心，使用当前位置+偏置
                    spring_center = body_pos + self.spring_center_bias
                
                # 计算从 body 指向引力中心的向量
                direction_vec = spring_center - body_pos
                distance = np.linalg.norm(direction_vec)
                
                if distance > 1e-6:
                    # 归一化方向
                    direction_normalized = direction_vec / distance
                    # 计算弹簧力大小 F = K * distance
                    force_magnitude = self.spring_k * distance
                    # 计算力向量
                    force_vector = direction_normalized * force_magnitude
                else:
                    # 距离太小，不施加力
                    force_vector = np.zeros(3)
                
                # 施加外力
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
        
        # 清零弹簧引力中心（spring 模式）
        self.spring_centers = []
        
        print(f"[ForceApplicator] 外力施加结束")
    
    def get_force_info(self, data_list=None):
        """
        获取当前外力信息（用于可视化）
        
        Args:
            data_list: mujoco.MjData 对象列表（spring 模式需要用于计算实时力向量）
        
        Returns:
            dict: 包含 body_id, force_vectors, is_active, remaining_time 等信息
                  如果未激活，返回 None
                  
                  返回字典结构：
                  - 'body_id': body 的索引
                  - 'body_name': body 的名称
                  - 'is_active': 是否激活
                  - 'mode': 停止模式 ('fixtime' 或 'keeping')
                  - 'style': 力的风格 ('constant' 或 'spring')
                  - 'force_vectors': 力向量列表，每个环境一个 (num_envs, 3)
                  - 'elapsed_time': 已经过时间
                  - 'remaining_time': 剩余时间（fixtime 模式）或 inf（keeping 模式）
                  - 'duration': 持续时间（fixtime 模式）或 None（keeping 模式）
                  - 'spring_centers': 引力中心列表（仅 spring 模式）
        """
        if not self.is_active:
            return None
        
        elapsed_time = time.time() - self.start_time
        
        # 根据模式计算剩余时间
        if self.force_mode['stop'] == 'fixtime':
            remaining_time = max(0.0, self.duration - elapsed_time)
        else:  # keeping 模式
            remaining_time = float('inf')  # 无限持续
        
        # 根据 style 和 select 计算力向量
        # 为所有环境初始化零向量列表
        if data_list is not None:
            num_envs = len(data_list)
        else:
            num_envs = max(self.selected_envs) + 1 if self.selected_envs else 1
        
        force_vectors = [np.zeros(3) for _ in range(num_envs)]
        
        if self.force_mode['style'] == 'constant':
            # 恒定力模式：选中的环境施加相同的力向量
            force_vector = self.force_direction * self.force_magnitude
            
            # 只为选中的环境设置非零力向量
            for env_i in self.selected_envs:
                if env_i < len(force_vectors):
                    force_vectors[env_i] = force_vector.copy()
        
        elif self.force_mode['style'] == 'spring':
            # 弹簧力模式：每个选中环境单独计算力向量
            if data_list is None:
                # 如果没有 data_list，无法计算实时力向量
                pass
            else:
                for spring_i, env_i in enumerate(self.selected_envs):
                    if env_i >= len(data_list):
                        continue
                    
                    data = data_list[env_i]
                    body_pos = data.xpos[self.body_id]
                    
                    # 使用 spring_i 索引到 spring_centers
                    if spring_i < len(self.spring_centers):
                        spring_center = self.spring_centers[spring_i]
                    else:
                        spring_center = body_pos + self.spring_center_bias
                    
                    direction_vec = spring_center - body_pos
                    distance = np.linalg.norm(direction_vec)
                    
                    if distance > 1e-6:
                        direction_normalized = direction_vec / distance
                        force_magnitude = self.spring_k * distance
                        force_vector = direction_normalized * force_magnitude
                    else:
                        force_vector = np.zeros(3)
                    
                    force_vectors[env_i] = force_vector
        
        result = {
            'body_id': self.body_id,
            'body_name': self.body_name,
            'is_active': self.is_active,
            'mode': self.force_mode['stop'],
            'style': self.force_mode['style'],
            'select': self.force_mode['select'],  # 新增：返回 select 配置
            'selected_envs': self.selected_envs,  # 新增：返回实际选中的环境列表
            'force_vectors': force_vectors,  # 每个环境的力向量列表
            'elapsed_time': elapsed_time,
            'remaining_time': remaining_time,
            'duration': self.duration if self.force_mode['stop'] == 'fixtime' else None,
        }
        
        # 如果是 spring 模式，添加引力中心信息
        if self.force_mode['style'] == 'spring':
            result['spring_centers'] = self.spring_centers
        
        return result
