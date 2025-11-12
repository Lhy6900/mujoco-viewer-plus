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
from pathlib import Path
from typing import Dict, Optional

# dof_idx_23_to_29: [ 0, 1, 2, 3, 4, 5,
#                     6, 7, 8, 9, 10, 11,
#                     12,13,14,
#                     15,16,17,18,
#                     22,23,24,25,]
# locked_kp: 40
# locked_kd: 1

# 日志记录类
class Logger:
    def __init__(self, dt):
        state_log = defaultdict(list)  # 用于存储状态日志
        rew_log = defaultdict(list)    # 用于存储奖励日志
        dt = dt                        # 时间步长
        num_episodes = 0               # 记录episode数量
        plot_process = None            # 用于绘图的进程

    def log_state(self, key, value):
        state_log[key].append(value)   # 记录单个状态

    def log_states(self, dict):
        for key, value in dict.items():     # 批量记录状态
            log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                # print("value.shape:",value.shape)
                rew_log[key].append(value.item() * num_episodes)
        num_episodes += num_episodes
        print("num_episodes:",num_episodes)
        print("============================================")

    def reset(self):
        state_log.clear()  # 清空状态日志
        rew_log.clear()    # 清空奖励日志


    def print_rewards(self):
        print("Average rewards per episode:")  # 打印每个episode的平均奖励
        for key, values in rew_log.items():
            mean = np.sum(np.array(values)) / num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {num_episodes}")
    
    def save_logs(self, save_path):
        """保存日志数据到文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **state_log)
        print(f"日志已保存到: {save_path}")
    
    def __del__(self):
        if plot_process is not None:
            plot_process.kill()  # 析构时关闭绘图进程
            

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

def transfer_q(q_raw):
    if q_raw.shape[0] == 23:
        q = q_raw.copy()
    elif q_raw.shape[0] == 29:
        q = q_raw[[0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11,
                    12,13,14,
                    15,16,17,18,
                    22,23,24,25]]
    else:
        raise ValueError(f"Invalid q_raw shape: {q_raw.shape}")
    return q

def inv_transfer_q(target_q, default_angles):
    target_q_29 = np.zeros(default_angles.shape[0])
    if target_q.shape[0] == 23 and default_angles.shape[0] == 29:
        target_q_29[[0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11,
                    12,13,14,
                    15,16,17,18,
                    22,23,24,25]] = target_q.copy()
    return target_q_29


# 重构get_observation,也将其改为3个函数的累计,一个更新状态，一个计算obs，一个计算obs_his一个但不需要这么多self吧，传出一个合适的参量就好了
def _pbhc_get_state(data, num_obs, timer, dt, motion_len, obs_his_dict, action):
    '''Extracts physical states from the mujoco data structure
    # action base_ang_vel dof_pos dof_vel history(304) pro_gravity,ref_phase(1) =23+3+23+23+3+1+304 =76 +76*4 =76*5 =380
    这里返回一个obs，一个更新后的obshistory字典,键值大小为[num_his*num_key_obs],传到外面用于记录，在计算obs的时候对obs按顺序concact即可，
    '''
    # action也需要29-23的转换
    action_raw = action.copy()
    action = transfer_q(action_raw)
    # joint_pos, joint_vel,transfer_q是用来从29转换为23的，目前先用23的，所以是直接copy过来的。
    obs = np.zeros(num_obs)
    q_raw = data.qpos[7:] # 23 dim
    q = transfer_q(q_raw)
    dq_raw = data.qvel[6:] # 23 dim ?????
    dq = transfer_q(dq_raw)
    # base_pos, base_quat, base_vel, base_omega
    pos = data.qpos[:3]
    quat_raw = data.qpos[3:7][[1,2,3,0]] # WXYZ to XYZW
    quat = quat_raw.copy()
    vel = data.qvel[:3]
    omega_raw = data.qvel[3:6]
    omega = omega_raw.copy()
    motion_time = timer * dt 
    ref_motion_phase = motion_time / motion_len
    # 下面四行是原本的转换，我先用自己的
    # r = R.from_quat(quat)  # R.from_quat: need xyzw
    # rpy = quaternion_to_euler_array(quat) # need xyzw
    # rpy[rpy > math.pi] -= 2 * math.pi
    # gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    obs[:23] = action * 1.0
    obs[23:26] = omega * 0.25
    obs[26:49] = q * 1.0
    obs[49:72] = dq * 0.05
    obs[72:72+304] = np.concatenate((obs_his_dict['action'], obs_his_dict['base_ang_vel'], obs_his_dict['dof_pos'], 
    obs_his_dict['dof_vel'], obs_his_dict['pro_gravity'], obs_his_dict['ref_phase']))

    obs[72+304:72+304+3] = project_gravity_to_world() * 1.0
    obs[72+304+3:72+304+3+1] = ref_motion_phase * 1.0
    obs_his_dict['action'] = np.roll(obs_his_dict['action'], 23)
    obs_his_dict['action'][:23] = action
    obs_his_dict['base_ang_vel'] = np.roll(obs_his_dict['base_ang_vel'], 3)
    obs_his_dict['base_ang_vel'][:3] = omega * 0.25
    obs_his_dict['dof_pos'] = np.roll(obs_his_dict['dof_pos'], 23)
    obs_his_dict['dof_pos'][:23] = q
    obs_his_dict['dof_vel'] = np.roll(obs_his_dict['dof_vel'], 23)
    obs_his_dict['dof_vel'][:23] = dq * 0.05
    obs_his_dict['pro_gravity'] = np.roll(obs_his_dict['pro_gravity'], 3)
    obs_his_dict['pro_gravity'][:3] = project_gravity_to_world() * 1.0
    obs_his_dict['ref_phase'] = np.roll(obs_his_dict['ref_phase'], 1)
    obs_his_dict['ref_phase'][:1] = ref_motion_phase * 1.0


    # print(omega_raw)
    ##################源码中没有启用域随机化########################
    # if RAND_OFFSET:
    #     q = q_raw - _motor_offset
    # if RAND_IMU:
    #     step_noise_imu = noise_imu.x
        
    #     rpy = (quaternion_to_euler_array(quat)) + step_noise_imu[:3] * noise_imu_rpy *(np.pi/180)  
    #     quat = rpy_to_quaternion_array(rpy)
    #     quat_dist = np.arccos(np.clip(np.abs(np.dot(quat_raw, quat)), -1.0, 1.0))
        
    #     omega = omega + step_noise_imu[3:] * noise_imu_omega
        
        # print('quat_dist:', quat_dist)
        # print('rpy', rpy)
        # print('omega', omega, '\t| omega_raw', omega_raw, '\t| step_noise_imu', step_noise_imu[3:])
        # breakpoint()
        
    # if RAND_RADIAL:
    #     # omega = radial_perturbation(omega)
    #     dq = radial_perturbation(dq)
        
    #     # print(f"{omega=}, {omega_raw=}")
    #     print(f"{dq=}, {dq_raw=}")
    #########################################这部分rpy先不管，到时候看有没有用上#####################################

    return obs,obs_his_dict,ref_motion_phase



# def _make_motionlib(self, cfg_policies: List[URCIPolicyObs]): # 这个函数好像没有必要搞， 先放这里不引用了
    '''
    Extracts physical states from the mujoco data structure , 
    这里是原文mujoco用于提取轨迹的，原文设定了多轨迹加载的机制，此处先不做此安排,直接return一个motion_lib
    '''
    # m_cfg = DictConfig({
    #     'motion_file': obs_cfg.motion_file,
    #     'asset': self.cfg.robot.motion.asset,
    #     'extend_config': self.cfg.robot.motion.extend_config,
    # })
    # if os.path.isfile(motion_file):
    #     with open(motion_file, 'rb') as f:
    #         motion_data = joblib.load(f)
    #     the_motion_data = motion_data[next(iter(motion_data))]
    #     motion_len = len(the_motion_data['dof']) / the_motion_data['fps']
    # # 需要的都在pbhc_config.yaml中
    # motion_lib = MotionLibRobot(m_cfg, num_envs=1, device='cpu')
    # motion_lib.load_motions(random_sample=False)[0]
    # return motion_lib
    # For all motion tracking policy, load the motion lib file
    # self.motion_libs: List[Optional[MotionLibRobot]] = []  #这里先注释掉，因为原文是用于加载多个轨迹的，此处先不加载，注意self.motion_libs是list类型
    
    # for cfg_policy in cfg_policies:
    #     obs_cfg, policy_fn = cfg_policy
    #     if isinstance(obs_cfg, DictConfig):  #如果obs是字典类型 就加载参数，后面两种情况不考虑了直接
    #         m_cfg = DictConfig({
    #             'motion_file': obs_cfg.motion_file,
    #             'asset': self.cfg.robot.motion.asset,
    #             'extend_config': self.cfg.robot.motion.extend_config,
    #         })
            
    #         motion_lib = MotionLibRobot(m_cfg, num_envs=1, device='cpu')
    #         motion_lib.load_motions(random_sample=False)[0]
    #         self.motion_libs.append(motion_lib)
    #     elif isinstance(obs_cfg, Callable):
    #         # self.motion_len = -1
    #         # pass
    #         self.motion_libs.append(None)
    #     else:
    #         raise ValueError(f"Invalid obs_cfg: {obs_cfg}")

def load_policy(checkpoint:Path, num_obs: int):
    '''
    这里加载策略网络,onnx类型，先复制过来，还没有适配好
    '''
    assert checkpoint.suffix == '.onnx', f"File {checkpoint} is not a .onnx file."

    session = ort.InferenceSession(checkpoint, providers=['CPUExecutionProvider'])  # 使用CPU

    actor_dim = num_obs
    action_dim = 23

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    example_input = np.random.randn(1, actor_dim).astype(np.float32)
    try_inferr = session.run([output_name], {input_name: example_input})
    assert try_inferr[0].shape == (1, action_dim), f"Action shape {try_inferr[0].shape} does not match expected shape (1, {action_dim})."
    def policy_fn(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # assert obs.shape == (1, actor_dim), f"Observation shape {obs.shape} does not match expected shape (1, {actor_dim})."
        result = session.run([output_name], obs_dict)
        # obs = obs_dict[input_name]
        # result = session.run([output_name], {input_name: obs_dict})
        # result = session.run([output_name], {input_name: obs})
        return result[0]
            
    return policy_fn

import onnxruntime as ort


# 主程序入口
if __name__ == "__main__":
    
    # 添加初始化， 例如仿真设置， 日志保存路径配置
    log_save_path = "./logs/pbhc_log.npz"
    model_path = "D:/RL/deploy/model/PBHC_ONNX/model_49000.onnx"
    # xml_path = "D:/RL/deploy/resources/robots/g1_description/g1_23dof_lock_wrist.xml"
    xml_path = 'D:/RL/deploy/resources/robots/g1_description/g1_29dof_zy.xml'
    model_path = Path(model_path)
    policy_fn = load_policy(model_path, num_obs=380)
    simulation_duration = 10000  # 仿真总时长
    simulation_dt = 0.002        # 仿真步长
    control_decimation = 10      # 控制步长
    # 23PBHC cfg information
    action_scale = 0.25
    clip_torques = True
    action_clip_value = 100.0
    #kpkd for 23 PBHC
    kps = np.array([100, 100, 100, 150, 40, 40,\
                    100, 100, 100, 150, 40, 40,\
                    400, 400, 400,\
                    100, 100, 50, 50, 40, 40, 40,
                    100, 100, 50, 50, 40, 40, 40], dtype=np.float32)
    kds = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0, \
                    2.0, 2.0, 2.0, 4.0, 2.0, 2.0, \
                    5.0, 5.0, 5.0,\
                    2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0,
                    2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    # original kpkd
    # kps = np.array([200, 150, 150, 200, 100, 100,\
    #                 200, 150, 150, 200, 100, 100,\
    #                 200, 200, 200,\
    #                 20, 20, 20, 20, 20, 5, 5,\
    #                 20, 20, 20, 20, 20, 5, 5], dtype=np.float32)
    # kds = np.array([5, 5, 5, 5, 5, 5,\
    #                 5, 5, 5, 5, 5, 5,\
    #                 5, 5, 5,\
    #                 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2,\
    #                 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2], dtype=np.float32)
    # 23自由度
    # default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
    #                           -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
    #                           0.0, 0.0, 0.0, 
    #                           0.2, 0.2, 0.0, 0.9, 
    #                           0.2, -0.2, 0.0, 0.9], dtype=np.float32)

    # 29自由度
    default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 
                                0.0, 0.0, 0.0, 
                                0.2, 0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 
                                0.2, -0.2, 0.0, 0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    
    tau_limit = np.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
                          88.0, 139.0, 88.0, 139.0, 50.0, 50.0, 
                          88.0, 50.0, 50.0, 
                          25.0, 25.0, 25.0, 25.0, 50.0, 50.0, 50.0,
                          25.0, 25.0, 25.0, 25.0, 50.0, 50.0, 50.0], dtype=np.float32)
 
    clip_action = 100.0
    clip_observations = 100.0
    action_scale = 0.25
    num_actions = 23
    num_obs = 380 
    num_critic = 763  #push task
    num_obs_history_length = 4

    # 定义上下文变量
    action = np.zeros(num_actions, dtype=np.float32) 
    # 在创建 obs 时确保使用 float32 类型
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_his_dict = {}
    obs_his_dict['action'] = np.zeros(23*num_obs_history_length, dtype=np.float32)
    obs_his_dict['base_ang_vel'] = np.zeros(3*num_obs_history_length, dtype=np.float32)
    obs_his_dict['dof_pos'] = np.zeros(23*num_obs_history_length, dtype=np.float32)
    obs_his_dict['dof_vel'] = np.zeros(23*num_obs_history_length, dtype=np.float32)
    obs_his_dict['pro_gravity'] = np.zeros(3*num_obs_history_length, dtype=np.float32)
    obs_his_dict['ref_phase'] = np.zeros(1*num_obs_history_length, dtype=np.float32)
    timer = 0
    ref_motion_phase = 0.0
    target_q = np.zeros(23, dtype=np.float32)
    
    # 加载机器人模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 动态选择设备 (CPU或CUDA)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")
    counter = 0

    d.qpos[0:3] = np.array([0, 0, 0.8])
    print("d.qpos[0:3]:",d.qpos[0:3])
    print("d.qpos[3:7]:",d.qpos[3:7])
    d.qpos[7:] = default_angles
    # mujoco.mj_step(m, d)
    # obs, obs_his_dict, ref_motion_phase = _pbhc_get_state(d, timer=timer, num_obs=380, dt=0.02, motion_len=7.0, obs_his_dict=obs_his_dict, action=action)
    logger = Logger(0.02)  # 日志记录器，dt为0.02

    # 加载策略网络
    # model_dict = torch.load(policy_path, map_location=device)
    # policy = ActorCritic(num_vae=36, num_obs_step=num_obs, 
    #                     num_critic_obs=num_critic, num_history=num_obs_history_length,\
    #                     num_actions=num_actions,\
    #                     actor_hidden_dims=[512, 256, 128],critic_hidden_dims=[1024, 512, 256, 128],\
    #                     ).to(device)
    # policy.load_state_dict(model_dict['model_state_dict'])
    # policy.eval()
    # policy_inference = policy.act_inference
    
    # 启动mujoco可视化窗口
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 仿真主循环，仿真时间未超过simulation_duration，并且轨迹未结束
        start = time.time()
        running = True
        while viewer.is_running():       # and time.time() - start < simulation_duration and running:
            step_start = time.time()
            # 每200步绘制一次状态图
            # if counter / control_decimation ==200:
            #     logger.plot_states()
            
            if counter % control_decimation == 0:  # 每隔control_decimation步更新一次
                # 构造观测向量
                timer += 1
                # 检查是否到达轨迹末尾
                if ref_motion_phase > 1.05:
                    print("轨迹数据已结束，保存日志并退出...")
                    # 保存日志
                    logger.save_logs(log_save_path)
                    running = False
                    break

                obs, obs_his_dict, ref_motion_phase = _pbhc_get_state(d, timer=timer, num_obs=380, dt=0.02, motion_len=7.0, obs_his_dict=obs_his_dict, action=action)
                # obs_tensor = torch.from_numpy(obs).to(device) # 将obs张量移动到正确设备
                # obs_tensor = torch.clip(obs_tensor, -clip_observations, clip_observations)
                print("obs:",obs)
                
                # 策略推理
                action = policy_fn({'actor_obs': obs.astype(np.float32).reshape(1, -1)})[0]  # 输出是23自由度的
                action = inv_transfer_q(action, default_angles)
                # break
                target_q = np.clip(action, -action_clip_value, action_clip_value) * action_scale + default_angles
                # 动作转为目标关节角度

            # 同步viewer，刷新显示
                        # 计算PD控制器输出
            tau = pd_control(target_q, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            # tau = np.clip(tau, -tau_limit, tau_limit) 
            d.ctrl[:] = tau  # 设置控制输入
            mujoco.mj_step(m, d)  # 进行一步仿真
            counter += 1
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