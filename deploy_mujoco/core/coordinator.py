"""
仿真协调器模块
协调多环境仿真、策略推理、可视化渲染等所有组件
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import torch
from multiprocessing import Value

from core.environment import EnvironmentManager
from core.policy import PolicyRunner
from core.observation import ObservationBuilder
from core.controller import pd_control
from visualization.ghost import GhostRenderer
from visualization.force_applicator import ForceApplicator
from visualization.force_visualizer import ForceVisualizer
from visualization.reward_calculator import compute_rewards
from visualization.reward_visualizer import RewardPlotter
from utils.math_utils import quat_invmul, get_orientation_2d_from_quat
from utils.logger import BMLogger


class SimulationCoordinator:
    """仿真协调器 - 管理整个仿真流程"""
    
    def __init__(self, config):
        """
        初始化仿真协调器
        
        Args:
            config: 配置字典，包含所有必要的配置参数
        """
        self.config = config
        self.model_path = config['model_path']
        self.xml_path = config['xml_path']
        self.num_envs = config.get('num_envs', 1)
        self.control_decimation = config.get('control_decimation', 10)
        self.simulation_dt = config.get('simulation_dt', 0.001)
        
        # 加载 MuJoCo 模型
        print("[协调器] 正在加载 MuJoCo 模型...")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = self.simulation_dt
        self.main_data = mujoco.MjData(self.model)
        
        # 初始化环境管理器
        print(f"[协调器] 正在创建 {self.num_envs} 个环境...")
        self.env_manager = EnvironmentManager(
            self.model, 
            self.num_envs, 
            env_spacing=config.get('env_spacing', 3.0)
        )
        
        # 初始化策略
        print("[协调器] 正在加载策略...")
        self.policy = PolicyRunner(
            self.model_path,
            num_envs=self.num_envs,
            policy_type=config.get('policy_type', 'imitation')
        )
        
        # 从策略元数据获取配置
        self.joint_xml = config['joint_xml']
        self.joint_seq = self.policy.joint_seq
        self.action_scale = self.policy.action_scale
        self.stiffness_array = self._map_joints(self.policy.stiffness, self.joint_seq, self.joint_xml)
        self.damping_array = self._map_joints(self.policy.damping, self.joint_seq, self.joint_xml)
        self.joint_pos_array = self._map_joints(self.policy.default_joint_pos, self.joint_seq, self.joint_xml)
        self.joint_pos_array_seq = self.policy.default_joint_pos
        
        # 初始化观测构造器
        self.obs_builder = ObservationBuilder(config.get('obs_config', {}))
        self.obs_builder.set_joint_mapping(
            self.joint_seq, 
            self.joint_xml,
            default_joint_pos_seq=self.joint_pos_array_seq
        )
        
        # 加载参考运动数据（如果需要）
        self.ghost_rendering_enabled = False
        self.motion_data = self._load_motion_data(config.get('motion_ref_path'))
        
        # 初始化 Ghost 渲染器（如果启用）
        self.ghost_renderers = []
        if self.ghost_rendering_enabled:
            for i in range(self.num_envs):
                self.ghost_renderers.append(GhostRenderer(self.model))
            print(f"[协调器] Ghost 渲染已启用")
        
        # 初始化外力系统
        self.force_applicator = ForceApplicator(
            self.model,
            body_name=config['force_config']['anchor_body'],
            force_anchor_bodyidx=mujoco.mj_name2id(
                self.model, 
                mujoco.mjtObj.mjOBJ_BODY, 
                config['force_config']['anchor_body']
            ),
            force_mode=config['force_config']['mode']
        )
        self.force_visualizer = ForceVisualizer()
        print("[协调器] 外力系统已初始化")
        
        # 初始化奖励系统
        self.reward_plotter = RewardPlotter(history_length=300)
        self.reward_plotter.register_terms([
            "smoothness",
            "pos_tracking_global",
            "pos_tracking_local",
            "quat_tracking_global",
            "quat_tracking_local"
        ])
        print("[协调器] 奖励可视化已初始化")
        
        # 初始化日志记录器
        self.logger = BMLogger(config.get('logger_dt', 0.02))
        
        # 初始化状态变量
        self.timestep_list = [0] * self.num_envs
        self.action_buffer_list = [np.zeros(len(self.action_scale), dtype=np.float32) for _ in range(self.num_envs)]
        self.target_dof_pos_list = [self.joint_pos_array.copy() for _ in range(self.num_envs)]
        self.last_qvel_for_reward = None
        
        # 交互状态（共享变量）
        self.current_env_idx = Value('i', 0)
        self.show_other_envs = Value('i', 0)
        self.show_reward_plot = Value('i', 1)
        self.show_main_ghost = Value('i', 1)
        self.show_other_ghosts = Value('i', 0)
        self.ctrl_pressed = Value('i', 0)
        
        # 初始化环境位置
        self.env_manager.initialize_positions(self.joint_pos_array, self.joint_xml)
        
        # 计数器
        self.counter = 0
        
        print("[协调器] ✓ 初始化完成")
    
    def _map_joints(self, values_seq, joint_seq, joint_xml):
        """将关节值从策略顺序映射到 XML 顺序"""
        if values_seq is None:
            return None
        return np.array([values_seq[joint_seq.index(joint)] for joint in joint_xml])
    
    def _load_motion_data(self, motion_ref_path):
        """加载参考运动数据"""
        if motion_ref_path is None:
            print("[协调器] 未提供参考运动路径，Ghost 渲染已禁用")
            return None
        
        try:
            motionref = np.load(motion_ref_path)
            motion_data = {
                'body_pos_w': motionref["body_pos_w"],
                'body_quat_w': motionref["body_quat_w"],
                'joint_pos': motionref["joint_pos"],
                'joint_vel': motionref["joint_vel"],
            }
            self.ghost_rendering_enabled = True
            print(f"[协调器] ✓ 成功加载参考运动数据：{motion_ref_path}")
            return motion_data
        except Exception as e:
            print(f"[协调器] 加载参考运动数据失败：{e}")
            print(f"[协调器] Ghost 渲染已禁用")
            return None
    
    def _build_observation(self, env_idx):
        """构造单个环境的观测"""
        data = self.env_manager.data_list[env_idx]
        timestep = self.timestep_list[env_idx]
        action_buffer = self.action_buffer_list[env_idx]
        
        # 使用 ObservationBuilder 构造观测
        obs = self.obs_builder.build(
            data=data,
            motion_data=self.motion_data,
            action_buffer=action_buffer,
            timestep=timestep,
            idx=env_idx
        )
        
        return obs
    
    def _compute_control(self, env_idx):
        """计算单个环境的控制"""
        obs = self._build_observation(env_idx)
        timestep = self.timestep_list[env_idx]
        
        # 策略推理
        action = self.policy.compute_action(obs, timestep, env_idx=env_idx)
        action = np.asarray(action).reshape(-1)
        
        # 保存动作历史
        self.action_buffer_list[env_idx] = action.copy()
        
        # 计算目标关节位置（策略顺序）
        target_dof_pos_seq = action * self.action_scale + self.joint_pos_array_seq
        
        # 转换为 XML 顺序
        target_dof_pos = np.array([
            target_dof_pos_seq[self.joint_seq.index(joint)] 
            for joint in self.joint_xml
        ])
        
        self.target_dof_pos_list[env_idx] = target_dof_pos
        
        # PD 控制
        data = self.env_manager.data_list[env_idx]
        tau = pd_control(
            target_dof_pos,
            data.qpos[7:],
            self.stiffness_array,
            np.zeros_like(self.damping_array),
            data.qvel[6:],
            self.damping_array
        )
        
        return tau
    
    def _update_ghost(self, env_idx):
        """更新单个环境的 Ghost"""
        if not self.ghost_rendering_enabled:
            return
        
        try:
            obs = self._build_observation(env_idx)
            timestep = self.timestep_list[env_idx]
            
            # 获取 ghost 输出
            policy_body_pos_w, policy_body_quat_w = self.policy.compute_ghost_outputs(
                obs, timestep, env_idx=env_idx
            )
            
            # 提取基座位置和姿态
            base_pos_policy = policy_body_pos_w[0, 0, :]
            base_quat_policy = policy_body_quat_w[0, 0, :]
            
            # 应用环境偏移
            base_pos_with_offset = base_pos_policy.copy()
            base_pos_with_offset[0] += self.env_manager.env_origins[env_idx, 0]
            base_pos_with_offset[1] += self.env_manager.env_origins[env_idx, 1]
            
            # 提取参考关节位置
            joint_pos_ref_seq = self.motion_data['joint_pos'][timestep, :]
            ghost_joint_pos_dof = np.array([
                joint_pos_ref_seq[self.joint_seq.index(joint)] 
                for joint in self.joint_xml
            ])
            
            # 构造 ghost qpos
            ghost_qpos = self.ghost_renderers[env_idx].construct_ghost_qpos(
                base_pos=base_pos_with_offset,
                base_quat=base_quat_policy,
                joint_pos_dof_order=ghost_joint_pos_dof,
                current_qpos=self.env_manager.data_list[env_idx].qpos
            )
            
            self.ghost_renderers[env_idx].set_ghost_qpos(ghost_qpos)
            
        except Exception as e:
            if timestep % 100 == 0:
                print(f"[警告] 环境 {env_idx} Ghost 更新失败: {e}")
    
    def _compute_rewards(self):
        """计算奖励"""
        if not self.ghost_rendering_enabled:
            return
        
        try:
            idx = self.current_env_idx.value
            timestep = self.timestep_list[idx]
            
            rewards = compute_rewards(
                timestep=timestep,
                current_env_idx=idx,
                dlist=self.env_manager.data_list,
                target_dof_pos_list=self.target_dof_pos_list,
                motionrefinputpos=self.motion_data['joint_pos'],
                motionrefinputvel=self.motion_data['joint_vel'],
                joint_xml=self.joint_xml,
                joint_seq=self.joint_seq,
                last_qvel=self.last_qvel_for_reward
            )
            
            # 更新速度历史
            current_joint_vel_seq = np.array([
                self.env_manager.data_list[idx].qvel[6 + self.joint_xml.index(joint)]
                for joint in self.joint_seq
            ])
            self.last_qvel_for_reward = current_joint_vel_seq.copy()
            
            self.reward_plotter.update(rewards)
            
        except Exception as e:
            if self.counter % 100 == 0:
                print(f"[警告] 奖励计算失败: {e}")
    
    def _render_scene(self, viewer):
        """渲染场景（Ghost、外力、其他环境）"""
        # 清空用户场景
        viewer.user_scn.ngeom = 0
        
        # 渲染主环境的 Ghost
        if self.ghost_rendering_enabled and self.show_main_ghost.value:
            try:
                idx = self.current_env_idx.value
                self.ghost_renderers[idx].render_ghost(viewer.user_scn)
            except Exception as e:
                if self.counter % 100 == 0:
                    print(f"[警告] 主环境 Ghost 渲染失败: {e}")
        
        # 渲染其他环境的 Ghost
        if self.ghost_rendering_enabled and self.show_other_ghosts.value and self.show_other_envs.value:
            for i in range(self.num_envs):
                if i == self.current_env_idx.value:
                    continue
                try:
                    self.ghost_renderers[i].render_ghost(viewer.user_scn)
                except Exception as e:
                    if self.counter % 100 == 0:
                        print(f"[警告] 环境 {i} Ghost 渲染失败: {e}")
        
        # 渲染外力箭头
        force_info = self.force_applicator.get_force_info(data_list=self.env_manager.data_list)
        if force_info is not None:
            force_vectors = force_info['force_vectors']
            force_scale = force_info.get('force_scale', 0.02)
            
            # 渲染主环境的外力
            if len(force_vectors) > self.current_env_idx.value:
                try:
                    self.force_visualizer.render_force_arrow(
                        viewer.user_scn,
                        self.model,
                        self.env_manager.data_list[self.current_env_idx.value],
                        force_info['body_id'],
                        force_vectors[self.current_env_idx.value],
                        force_scale=force_scale
                    )
                except Exception as e:
                    if self.counter % 100 == 0:
                        print(f"[警告] 主环境外力渲染失败: {e}")
            
            # 渲染其他环境的外力
            if self.show_other_envs.value:
                for env_i in range(self.num_envs):
                    if env_i == self.current_env_idx.value or env_i >= len(force_vectors):
                        continue
                    try:
                        self.force_visualizer.render_force_arrow(
                            viewer.user_scn,
                            self.model,
                            self.env_manager.data_list[env_i],
                            force_info['body_id'],
                            force_vectors[env_i],
                            force_scale=force_scale
                        )
                    except Exception as e:
                        if self.counter % 100 == 0:
                            print(f"[警告] 环境 {env_i} 外力渲染失败: {e}")
        
        # 渲染其他环境的机器人
        if self.show_other_envs.value:
            vopt = mujoco.MjvOption()
            pert = mujoco.MjvPerturb()
            catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
            
            for i in range(self.num_envs):
                if i == self.current_env_idx.value:
                    continue
                try:
                    mujoco.mjv_addGeoms(
                        self.model,
                        self.env_manager.data_list[i],
                        vopt,
                        pert,
                        catmask,
                        viewer.user_scn
                    )
                except Exception as e:
                    if self.counter % 100 == 0:
                        print(f"[警告] 环境 {i} 机器人渲染失败: {e}")
        
        # 渲染奖励图表
        if self.show_reward_plot.value:
            try:
                figures_data = self.reward_plotter.get_figures_for_viewer(viewer.viewport)
                viewer.set_figures(figures_data)
            except Exception as e:
                if self.counter % 100 == 0:
                    print(f"[警告] 奖励图表渲染失败: {e}")
        else:
            try:
                viewer.set_figures([])
            except:
                pass
    
    def _create_key_callback(self):
        """创建键盘回调函数"""
        def key_callback(key):
            """键盘快捷键回调"""
            # 检测 Ctrl 键
            if key == 341 or key == 345:
                self.ctrl_pressed.value = 1
                return
            
            # 上箭头 - 切换到上一个环境
            if key == 265:
                self.current_env_idx.value = (self.current_env_idx.value - 1) % self.num_envs
                print(f"[切换环境] 当前主环境: {self.current_env_idx.value}")
            
            # 下箭头 - 切换到下一个环境
            elif key == 264:
                self.current_env_idx.value = (self.current_env_idx.value + 1) % self.num_envs
                print(f"[切换环境] 当前主环境: {self.current_env_idx.value}")
            
            # M 键 - 切换其他环境显示
            elif key == 77 or key == 109:
                if self.ctrl_pressed.value:
                    # Ctrl+M - 切换其他环境的 ghost
                    self.show_other_ghosts.value = 1 - self.show_other_ghosts.value
                    status = "显示" if self.show_other_ghosts.value else "隐藏"
                    print(f"[Ghost 可视化] {status}其他环境的参考轨迹")
                    self.ctrl_pressed.value = 0
                else:
                    # M - 切换其他环境的机器人
                    self.show_other_envs.value = 1 - self.show_other_envs.value
                    status = "显示" if self.show_other_envs.value else "隐藏"
                    print(f"[多环境渲染] {status}其他环境")
            
            # Ctrl+G - 切换主环境的 ghost
            elif (key == 71 or key == 103) and self.ctrl_pressed.value:
                self.show_main_ghost.value = 1 - self.show_main_ghost.value
                status = "显示" if self.show_main_ghost.value else "隐藏"
                print(f"[Ghost 可视化] {status}主环境的参考轨迹")
                self.ctrl_pressed.value = 0
            
            # Ctrl+R - 切换奖励曲线
            elif (key == 82 or key == 114) and self.ctrl_pressed.value:
                self.show_reward_plot.value = 1 - self.show_reward_plot.value
                status = "显示" if self.show_reward_plot.value else "隐藏"
                print(f"[奖励可视化] {status}奖励曲线窗口")
                self.ctrl_pressed.value = 0
            
            # Ctrl+F - 触发外力
            elif (key == 70 or key == 102) and self.ctrl_pressed.value:
                self.force_applicator.trigger(
                    duration=5.0,
                    force_magnitude=20.0,
                    force_direction=np.array([0.0, 1.0, 0.0]),
                    data_list=self.env_manager.data_list
                )
                self.ctrl_pressed.value = 0
        
        return key_callback
    
    def run(self):
        """运行仿真主循环"""
        print("[协调器] 启动仿真...")
        print("[协调器] 快捷键:")
        print("  ↑/↓: 切换主环境")
        print("  M: 切换其他环境显示")
        print("  Ctrl+G: 切换主环境 Ghost")
        print("  Ctrl+M: 切换其他环境 Ghost")
        print("  Ctrl+R: 切换奖励曲线")
        print("  Ctrl+F: 施加/停止外力")
        
        key_callback = self._create_key_callback()
        
        with mujoco.viewer.launch_passive(
            self.model, 
            self.main_data, 
            key_callback=key_callback
        ) as viewer:
            # 仿真主循环 - 持续运行直到用户关闭窗口
            while viewer.is_running():
                step_start = time.time()
                
                # 同步主数据到当前选中的环境
                idx = self.current_env_idx.value
                self.env_manager.sync_to_main_data(self.main_data, idx)
                
                # 计算所有环境的控制并执行步进
                for i in range(self.num_envs):
                    tau = self._compute_control(i)
                    self.env_manager.data_list[i].ctrl[:] = tau
                    mujoco.mj_step(self.model, self.env_manager.data_list[i])
                
                self.counter += 1
                
                # 控制频率更新
                if self.counter % self.control_decimation == 0:
                    # 计算奖励
                    self._compute_rewards()
                    
                    # 更新所有环境的 Ghost
                    if self.ghost_rendering_enabled:
                        for i in range(self.num_envs):
                            self._update_ghost(i)
                    
                    # 递增时间步
                    for i in range(self.num_envs):
                        self.timestep_list[i] += 1
                    
                    # 渲染场景
                    self._render_scene(viewer)
                
                # 同步 viewer
                viewer.sync()
                
                # 同步外力
                if np.any(self.main_data.xfrc_applied != 0):
                    self.env_manager.data_list[self.current_env_idx.value].xfrc_applied[:] = \
                        self.main_data.xfrc_applied[:]
                else:
                    self.main_data.xfrc_applied[:] = 0
                    self.env_manager.data_list[self.current_env_idx.value].xfrc_applied[:] = 0
                
                # 更新外力施加器
                self.force_applicator.update(self.env_manager.data_list)
                
                # 控制仿真速率
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("[协调器] 仿真结束")
