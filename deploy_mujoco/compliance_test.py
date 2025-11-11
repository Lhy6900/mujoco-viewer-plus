# 逐行解释

import os
import time  # 导入time模块，用于计时
import mujoco.viewer  # 导入mujoco的viewer模块，用于可视化仿真
import mujoco  # 导入mujoco主模块
import numpy as np  # 导入numpy，用于数值计算
from scipy.spatial.transform import Rotation as R  # 导入scipy的Rotation
# from isaacgym.torch_utils import *  # 导入isaacgym的torch工具函数
import torch  # 导入PyTorch
import yaml  # 导入yaml，用于读取配置文件
from module.actor_critic import ActorCritic  # 导入自定义的ActorCritic类
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import numpy as np  # 再次导入numpy（重复）
from collections import defaultdict  # 导入defaultdict，用于字典的默认值
from multiprocessing import Process, Value  # 导入多进程相关类
# 日志记录类
class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)  # 用于存储状态日志
        self.rew_log = defaultdict(list)    # 用于存储奖励日志
        self.dt = dt                        # 时间步长
        self.num_episodes = 0               # 记录episode数量
        self.plot_process = None            # 用于绘图的进程

    def log_state(self, key, value):
        self.state_log[key].append(value)   # 记录单个状态

    def log_states(self, dict):
        for key, value in dict.items():     # 批量记录状态
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                # print("value.shape:",value.shape)
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes
        print("self.num_episodes:",self.num_episodes)
        print("============================================")

    def reset(self):
        self.state_log.clear()  # 清空状态日志
        self.rew_log.clear()    # 清空奖励日志

    def plot_states(self):
        self.plot_process = Process(target=self._plot)  # 新建进程绘图
        self.plot_process.start()

    def _plot(self):
        log= self.state_log
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))  # 生成时间序列
            break

        dof_pos = np.array(log['dof_pos'])  # 关节位置
        actions_dof = np.array(log['actions_dof'])  # 关节命令
        fig_dof_pos = plt.figure()
        axs = fig_dof_pos.subplots(2, 2)
        a = axs[0, 0]
        a.plot(time, dof_pos[:,12], label='dof_pos_0')
        a.plot(time, actions_dof[:,12], label='cmd_0')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[0, 1]
        a.plot(time, dof_pos[:,13], label='dof_pos_1')
        a.plot(time, actions_dof[:,13], label='cmd_1')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 0]
        a.plot(time, dof_pos[:,14], label='dof_pos_2')
        a.plot(time, actions_dof[:,14], label='cmd_2')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 1]
        a.plot(time, dof_pos[:,15], label='dof_pos_3')
        a.plot(time, actions_dof[:,15], label='cmd_3')
        a.set(xlabel='time [s]')
        a.legend() 
        
        dof_vel = np.array(log['dof_vel'])  # 关节速度
        fig_dof_vel = plt.figure()
        axs = fig_dof_vel.subplots(2, 2)
        a = axs[0, 0]
        a.plot(time, dof_vel[:,12], label='dof_vel_0')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[0, 1]
        a.plot(time, dof_vel[:,13], label='dof_vel_1')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 0]
        a.plot(time, dof_vel[:,14], label='dof_vel_2')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 1]
        a.plot(time, dof_vel[:,15], label='dof_vel_3')
        a.set(xlabel='time [s]')
        a.legend() 
        
        dof_trq = np.array(log['dof_trq'])  # 关节力矩
        fig_dof_trq = plt.figure()
        axs = fig_dof_trq.subplots(2, 2)
        a = axs[0, 0]
        a.plot(time, dof_trq[:,12], label='dof_trq_0')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[0, 1]
        a.plot(time, dof_trq[:,13], label='dof_trq_1')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 0]
        a.plot(time, dof_trq[:,14], label='dof_trq_2')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 1]
        a.plot(time, dof_trq[:,15], label='dof_trq_3')
        a.set(xlabel='time [s]')
        a.legend() 
        
        # 下面注释掉的代码是用于绘制其他关节力矩的
        # fig_dof_trq2 = plt.figure()
        # axs = fig_dof_trq2.subplots(2, 1)
        # a = axs[0]
        # a.plot(time, dof_trq[:,4], label='dof_trq_4')
        # a.set(xlabel='time [s]')
        # a.legend()
        # a = axs[1]
        # a.plot(time, dof_trq[:,5], label='dof_trq_5')
        # a.set(xlabel='time [s]')
        # a.legend()

        plt.show()  # 显示所有图

    def print_rewards(self):
        print("Average rewards per episode:")  # 打印每个episode的平均奖励
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def save_logs(self, save_path):
        """保存日志数据到文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **self.state_log)
        print(f"日志已保存到: {save_path}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()  # 析构时关闭绘图进程
            

# 计算重力方向
def get_gravity_orientation(quaternion):
    """
    计算重力方向在机体坐标系中的表示
    quaternion: (x, y, z, w) 格式的四元数
    返回: 重力向量在机体坐标系中的表示
    """
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    gravity_orientation = np.zeros(3)

    # 计算重力方向 (在机体坐标系中表示的[0,0,-1])
    gravity_orientation[0] = 2 * (x*z + w*y)
    gravity_orientation[1] = 2 * (y*z - w*x)
    gravity_orientation[2] = 1 - 2 * (x*x + y*y)

    return gravity_orientation

def project_gravity_to_world():
    """
    计算重力向量在机体坐标系中的表示，使用scipy的Rotation
    返回: 重力向量在机体坐标系中的表示
    """
    # MuJoCo中四元数的顺序是[w, x, y, z]，而scipy.Rotation需要[x, y, z, w]
    quat = np.array([d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[3]])  # 转换为[x, y, z, w]格式
    
    # 使用scipy的Rotation计算旋转矩阵
    rot_mat = R.from_quat(quat).as_matrix()
    
    # 计算重力向量在机体坐标系中的表示
    base_proj_gravity = np.matmul(rot_mat.T, np.array([0, 0, -1.0]))
    
    return base_proj_gravity

# PD控制器
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_conjugate(q):
    # q: (..., 4) 四元数 (x, y, z, w)
    x, y, z, w = q.unbind(-1)
    return torch.stack([-x, -y, -z, w], dim=-1)

def quat_mul(q, r):
    # q, r: (..., 4) 四元数 (x, y, z, w)
    x1, y1, z1, w1 = q.unbind(-1)
    x2, y2, z2, w2 = r.unbind(-1)
    
    # 正确的四元数乘法公式 (x,y,z,w) 格式
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return torch.stack([x, y, z, w], dim=-1)

def quat_rotate_inverse(q, v):
    """
    使用四元数q的逆旋转来旋转向量v
    q: (..., 4) 四元数 (x, y, z, w)
    v: (..., 3) 向量
    返回: v 在 q 的逆旋转下的结果
    """
    # 计算四元数的共轭（对于单位四元数，共轭等于逆）
    q_conj = quat_conjugate(q)
    
    # 将向量v转换为纯四元数 (v, 0)
    zeros = torch.zeros(v.shape[:-1] + (1,), dtype=v.dtype, device=v.device)
    v_as_quat = torch.cat([v, zeros], dim=-1)
    
    # 执行旋转: q^-1 * v * q
    result = quat_mul(quat_mul(q_conj, v_as_quat), q)
    
    # 返回向量部分
    return result[..., :3]

# 这个函数只是接受target_points_global和target_points_local，其实是在用data数据计算actual_global和actual_local
def log_keypoint(target_keypoints_global, target_keypoints_local, data):
    # sta_dict['target_keypoints_global'].append(target_keypoints_global)
    # sta_dict['target_keypoints_local'].append(target_keypoints_local)
    actual_root_pos=data.qpos[0:3]  
    print('compare pos:', data.qpos[0:3], data.xpos[1])                                                              #这玩意对不对，是不是
    actual_root_quat_wxyz=data.qpos[3:7]
    actual_root_quat_xyzw = np.array([actual_root_quat_wxyz[1], 
                                      actual_root_quat_wxyz[2], 
                                      actual_root_quat_wxyz[3], 
                                      actual_root_quat_wxyz[0]])
    actual_keypoints_global=np.zeros([29,7])
    actual_keypoints_local = np.zeros([29, 7])
    for i in range(29):
        actual_keypoints_global[i,:3]=data.xpos[i+2]
        actual_keypoints_global[i,3:6]=data.xquat[i+2,1:4]  #从left_h开始的
        actual_keypoints_global[i,6:7]=data.xquat[i+2,0:1]
        # 开始局部转换
        actual_keypoints_local[i, :3] = quat_rotate_inverse(
            torch.from_numpy(actual_root_quat_xyzw).view(-1, 4),
            torch.from_numpy(actual_keypoints_global[i, :3] - actual_root_pos).view(-1, 3)
        ).numpy().flatten()                                             #在body坐标系下的位置
        actual_quat_xyzw = actual_keypoints_global[i, 3:7]
        actual_keypoints_local[i, 3:7] = quat_mul(
            torch.from_numpy(actual_quat_xyzw).view(-1, 4),
            quat_conjugate(torch.from_numpy(actual_root_quat_xyzw).view(-1, 4))
        ).numpy().flatten()
    # sta_dict['actual_keypoints_global'].append(actual_keypoints_global)
    # sta_dict['actual_keypoints_local'].append(actual_keypoints_local)
    return {
            'target_keypoints_global': target_keypoints_global,
            'target_keypoints_local': target_keypoints_local,
            'actual_keypoints_global': actual_keypoints_global,
            'actual_keypoints_local': actual_keypoints_local,
    }




# 主程序入口
if __name__ == "__main__":
    # 从命令行获取配置文件名
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    # 读取配置文件
    with open(f"./configs/{config_file}", "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]
        traj_path = config["traj_path"]
        
        # 添加日志保存路径配置
        log_save_path = config.get("log_save_path", "./logs/39800_origin_simulation_log.npz")
    
        # 外部力参数
        spring_kp = 50
        apply_external_force = config.get("apply_external_force", False)
        force_body_name = config.get("force_body_name", "torso")
        force_start_time = config.get("force_start_time", 1.0)
        force_duration = config.get("force_duration", 3.0)
        force_magnitude = config.get("force_magnitude", 100.0)
        force_direction_world = np.array(config.get("force_direction_world", [1.0, 0.0, 0.0]), dtype=np.float32)
        # 归一化方向向量
        if np.linalg.norm(force_direction_world) > 1e-6:
            force_direction_world = force_direction_world / np.linalg.norm(force_direction_world)
        else:
            force_direction_world = np.array([1.0, 0.0, 0.0], dtype=np.float32) # 避免零向量
        

        simulation_duration = 10000  # 仿真总时长
        simulation_dt = 0.002        # 仿真步长
        control_decimation = 10      # 控制步长

        # PD控制器参数
        # kps = np.array([200, 150, 150, 200, 20, 20,\
        #                 200, 150, 150, 200, 20, 20,\
        #                 200, 200, 200,\
        #                 20, 20, 20, 20, 20, 5, 5,\
        #                 20, 20, 20, 20, 20, 5, 5], dtype=np.float32)
        # kds = np.array([5, 5, 5, 5, 2, 2,\
        #                 5, 5, 5, 5, 2, 2,\
        #                 5, 5, 5,\
        #                 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2,\
        #                 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2], dtype=np.float32)

        kps = np.array([200, 150, 150, 200, 100, 100,\
                        200, 150, 150, 200, 100, 100,\
                        200, 200, 200,\
                        20, 20, 20, 20, 20, 5, 5,\
                        20, 20, 20, 20, 20, 5, 5], dtype=np.float32)
        kds = np.array([5, 5, 5, 5, 5, 5,\
                        5, 5, 5, 5, 5, 5,\
                        5, 5, 5,\
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2,\
                        0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2], dtype=np.float32)

        # 默认关节角度
        default_angles = np.array([ -0.2,  0.0,  0.0,  0.42, -0.23, 0.0,\
                                    -0.2,  0.0,  0.0,  0.42, -0.23, 0.0,\
                                    0.0, 0.0, 0.0,\
                                    0.0, 0.2, 0.15, 1.2, 0.0, 0.0, 0.0,\
                                    0.0, -0.2, -0.15, 1.2, 0.0, 0.0, 0.0,], dtype=np.float32)

        # 观测和动作缩放参数
        # ang_vel_scale = 0.25
        # dof_pos_scale = 1.0
        # dof_vel_scale = 0.05
        ang_vel_scale = 1.0
        dof_pos_scale = 2.0
        dof_vel_scale = 0.2

        
        clip_action = 100.0
        clip_observations = 100.0
        
        action_scale = 0.25
        cmd_scale = 1.0

        num_actions = 29
        num_obs = 332 
        # num_critic = 695  #original model
        num_critic = 3117  #push task

        num_obs_history_length = 5
        
        cmd = np.array([1.0, 0, 0.0], dtype=np.float32)
        
    # 读取轨迹数据
    traj_counter = 0
    traj_data_xyzw_keypoints = np.genfromtxt(traj_path, delimiter=',')   
    # 获取轨迹总长度
    traj_total_length = len(traj_data_xyzw_keypoints)
    print(f"轨迹总长度: {traj_total_length}")

    last_target_base_pos = traj_data_xyzw_keypoints[traj_counter,:3]
    last_target_base_quat_xyzw = traj_data_xyzw_keypoints[traj_counter,3:7]
    last_target_keypoints = traj_data_xyzw_keypoints[traj_counter, 36:36+29*7].reshape(29, 7)
    for i in range(29):
            last_target_keypoints[i, :3] = quat_rotate_inverse(
                            torch.from_numpy(last_target_base_quat_xyzw).view(-1, 4),
                            torch.from_numpy(last_target_keypoints[i, :3] - last_target_base_pos).view(-1, 3)
                        ).numpy().flatten()
                        
            last_target_keypoints[i, 3:7] = quat_mul(
                            torch.from_numpy(last_target_keypoints[i, 3:7]).view(-1, 4),
                            quat_conjugate(torch.from_numpy(last_target_base_quat_xyzw).view(-1, 4))
                        ).numpy().flatten()

    # 定义上下文变量
    action = np.zeros(num_actions, dtype=np.float32) 
    target_dof_pos_action = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_history = np.zeros(num_obs*num_obs_history_length, dtype=np.float32)
    counter = 0
    
    # 加载机器人模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 动态选择设备 (CPU或CUDA)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")

    # 加载策略网络
    model_dict = torch.load(policy_path, map_location=device)
    policy = ActorCritic(num_vae=36, num_obs_step=num_obs, 
                        num_critic_obs=num_critic, num_history=num_obs_history_length,\
                        num_actions=num_actions,\
                        actor_hidden_dims=[512, 256, 128],critic_hidden_dims=[1024, 512, 256, 128],\
                        ).to(device)
    policy.load_state_dict(model_dict['model_state_dict'])
    policy.eval()
    policy_inference = policy.act_inference
    
    # 获取施加外力的身体ID
    force_body_id = -1
    if apply_external_force:
        try:
            force_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, force_body_name)
            print(f"将外力施加到身体: {force_body_name} (ID: {force_body_id})")
        except KeyError:
            print(f"错误: 未找到名为 '{force_body_name}' 的身体。将禁用外部力。")
            apply_external_force = False

    # 在类的初始化部分添加引力中心的属性
    logger = Logger(0.02)  # 日志记录器，dt为0.02
    gravity_center = np.zeros(3) # 引力中心位置

    # 启动mujoco可视化窗口
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 仿真主循环，仿真时间未超过simulation_duration，并且轨迹未结束
        start = time.time()
        running = True
        while viewer.is_running() and time.time() - start < simulation_duration and running:
            step_start = time.time()
            # 计算PD控制器输出
            tau = pd_control(target_dof_pos_action, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau  # 设置控制输入

            # --- 外部力施加逻辑 ---用force_start_time控制外力开始的时间，这里只开始一次，所以只需要记一个gravity_center
            if apply_external_force and force_body_id != -1:
                current_time = d.time
                relative_time = current_time - force_start_time
                if relative_time >= 0 and relative_time < force_duration:
                # 初始化引力中心
                    if np.all(gravity_center == 0) and relative_time < 0.002 and relative_time >= 0:
                        # 获取施加外力的身体位置
                        body_position = d.xpos[force_body_id]
                        # 初始化引力中心，假设在身体附近[-0.2, 0.2]的范围内
                        gravity_center = body_position + np.random.uniform(-0.3, 0.3, size=3)
                        print("gravity_center:",gravity_center)

                    # 计算身体与引力中心的距离
                    body_position = d.xpos[force_body_id]
                    distance_vector = body_position - gravity_center
                    distance = np.linalg.norm(distance_vector)

                    # 施加与距离成比例的外力
                    if distance > 0:  # 确保距离不为零
                        force_magnitude = spring_kp * distance  # spring_kp 是弹簧系数, 根据距离决定力的norm
                        applied_force_vector = force_magnitude * (distance_vector / distance)  # 施加的外力方向与距离向量相同
                        d.xfrc_applied[force_body_id, :3] = applied_force_vector
                    else:
                        d.xfrc_applied[force_body_id, :] = 0.0  # 如果距离为零，施加零外力
                elif relative_time >= force_duration:
                    gravity_center = np.zeros(3) 
                    d.xfrc_applied[force_body_id, :] = 0.0
            # --- 外部力施加逻辑结束 ---

            mujoco.mj_step(m, d)  # 进行一步仿真
            counter += 1

            # 每200步绘制一次状态图
            # if counter / control_decimation ==200:
            #     logger.plot_states()
            
            if counter % control_decimation == 0:  # 每隔control_decimation步更新一次
                # 构造观测向量
                qj = d.qpos[7:].copy()
                dqj = d.qvel[6:].copy()
                quat = d.qpos[3:7].copy()
                omega = d.qvel[3:6].copy()
                if traj_counter !=0:
                    keypoint_dict = log_keypoint(target_keypoints_global, target_keypoints_local, d)
                    logger.log_states(keypoint_dict)
                traj_counter +=1
                
                # 检查是否到达轨迹末尾
                if traj_counter >= traj_total_length - 10:
                    print("轨迹数据已结束，保存日志并退出...")
                    # 保存日志
                    logger.save_logs(log_save_path)
                    running = False
                    break
                
                target_base_pos = traj_data_xyzw_keypoints[traj_counter,:3]
                target_base_quat_xyzw = traj_data_xyzw_keypoints[traj_counter,3:7]
                target_dof_pos = traj_data_xyzw_keypoints[traj_counter,7:36]
                # print("traj_counter:",traj_counter)
                # print("traj_data_xyzw_keypoints[traj_counter,7:36]:",traj_data_xyzw_keypoints[traj_counter,7:36])
                # print("traj_data_xyzw_keypoints[traj_counter+1,7:36]:",traj_data_xyzw_keypoints[traj_counter+1,7:36])
                # print("traj_data_xyzw_keypoints[traj_counter+2,7:36]:",traj_data_xyzw_keypoints[traj_counter+2,7:36])


        

                # 获取关键点数据
                target_keypoints = traj_data_xyzw_keypoints[traj_counter, 36:36+29*7].reshape(29, 7)  #29个body，形状上没错。这个应该是世界系下的
                target_keypoints_global = target_keypoints.copy()
                keypoints_delta = np.zeros_like(target_keypoints)
                if traj_counter > 0:
                    # 计算目标位姿的增量
                    target_base_pos_delta = quat_rotate_inverse(torch.from_numpy(target_base_quat_xyzw).view(-1,4),\
                                (torch.from_numpy(target_base_pos)-torch.from_numpy(last_target_base_pos)).view(-1,3)).numpy()
                    target_base_quat_delta = quat_mul(torch.from_numpy(target_base_quat_xyzw).view(-1,4),\
                                quat_conjugate(torch.from_numpy(last_target_base_quat_xyzw).view(-1,4))).numpy()
                    
                    for i in range(29):
                        # 当前帧关键点转换到机体坐标系
                        target_keypoints[i, :3] = quat_rotate_inverse(
                            torch.from_numpy(target_base_quat_xyzw).view(-1, 4),
                            torch.from_numpy(target_keypoints[i, :3] - target_base_pos).view(-1, 3)
                        ).numpy().flatten()
                        
                        target_keypoints[i, 3:7] = quat_mul(
                            torch.from_numpy(target_keypoints[i, 3:7]).view(-1, 4),
                            quat_conjugate(torch.from_numpy(target_base_quat_xyzw).view(-1, 4))
                        ).numpy().flatten()
                        

                        keypoints_delta[i, :3] = target_keypoints[i, :3] - last_target_keypoints[i, :3]
                        
                        # 姿态delta（四元数）
                        keypoints_delta[i, 3:7] = quat_mul(
                            torch.from_numpy(target_keypoints[i, 3:7]).view(-1, 4),
                            quat_conjugate(torch.from_numpy(last_target_keypoints[i, 3:7]).view(-1, 4))
                        ).numpy().flatten()
                    target_keypoints_local = target_keypoints.copy()





                # 填充观测向量
                obs[:3] = omega * ang_vel_scale
                obs[3:6] = project_gravity_to_world()
                # print("obs[3:6]:",obs[3:6])
                # print("gravity_before:",get_gravity_orientation(quat))
                obs[6:35] = (qj - default_angles) * dof_pos_scale
                obs[35:64] = dqj * dof_vel_scale
                obs[64:93] = action
                obs[93:96] = target_base_pos_delta * 100.0
                obs[96:100] = target_base_quat_delta
                obs[100:129] = (target_dof_pos - default_angles) * dof_pos_scale
                
                # 添加关键点delta到obs
                if traj_counter > 0:
                    obs[129:332] = keypoints_delta.reshape(-1)  # 29*7 = 203维
                else:
                    obs[129:332] = 0.0  # 第一步时，关键点delta为零
                
                # 打印关键变量
                print('time:',d.time)
                print("\n--- Step:", counter, "---")
                print("omega (角速度):", omega * ang_vel_scale)
                print("gravity_orientation:", get_gravity_orientation(quat))
                print("qj (关节角度):", (qj - default_angles) * dof_pos_scale, "...")  # 只打印前5个元素
                print("qj:",qj)
                print("dqj (关节速度):", dqj * dof_vel_scale, "...")
                print("action:", action, "...")
                print("target_base_pos_delta:", target_base_pos_delta * 100.0)
                print("target_base_quat_delta:", target_base_quat_delta)
                print("target_dof_pos:", (target_dof_pos - default_angles) * dof_pos_scale, "...")
                print("target_dof_pos:", target_dof_pos)
                # print('目标点增量：',keypoints_delta)

                
                obs_tensor = torch.from_numpy(obs).to(device) # 将obs张量移动到正确设备
                obs_tensor = torch.clip(obs_tensor, -clip_observations, clip_observations)
                
                # 观测历史更新
                obs_history[:] = np.roll(obs_history, num_obs)
                obs_history[:num_obs] = obs
                obs_history_tensor = torch.from_numpy(obs_history).to(device) # 将obs_history张量移动到正确设备
                
                # 策略推理
                action_tensor = policy_inference(obs_tensor, obs_history_tensor)
                action_tensor = torch.clip(action_tensor, -clip_action, clip_action)
                action = action_tensor.cpu().detach().numpy() # 在转换为numpy之前，先移动到CPU
                # 动作转为目标关节角度
                target_dof_pos_action = action * action_scale + default_angles
                print("target_dof_pos_action:",target_dof_pos_action)
                # 保存上一步的状态
                last_target_base_pos = target_base_pos.copy()
                last_target_base_quat_xyzw = target_base_quat_xyzw.copy()
                last_target_keypoints = target_keypoints.copy()

                # 下面注释掉的代码用于记录日志
                dof_pos = d.qpos[7:].copy()
                dof_vel = d.qvel[6:].copy()
                dof_trq = d.ctrl[:].copy()
                root_state = d.qpos[:7].copy()
                xfrc_applied = d.xfrc_applied[force_body_id, :3].copy()

                logger.log_states(
                    {
                        'actions_dof':target_dof_pos_action,
                        'dof_pos': dof_pos,
                        'dof_vel': dof_vel,
                        'dof_trq': dof_trq,
                        'xfrc_applied': xfrc_applied,
                        'root_state': root_state,
                        'gravity_center': gravity_center,
                        'force_body_id': force_body_id,
                    }
                )

            # 同步viewer，刷新显示
            viewer.sync()

            # 下面注释掉的代码用于精确控制仿真步长
            # time_until_next_step = m.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step)
            #     print("time_until_next_step:",time_until_next_step)
        
        # 如果是因为轨迹结束而退出循环，确保日志已保存
        if not running or traj_counter >= traj_total_length - 1:            
            logger.save_logs(log_save_path)
            print("模拟结束，日志已保存")

# 这个文件采用弹簧外力施加，并且允许记录引力中心和施加外力的索引，方便后续的可视化。