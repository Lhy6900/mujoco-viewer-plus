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

# 日志记录类
class Logger:
    def __init__(self, dt):
        state_log = defaultdict(list)  # 用于存储状态日志
        rew_log = defaultdict(list)    # 用于存储奖励日志
        dt = dt                        # 时间步长
        num_episodes = 0               # 记录episode数量

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

def project_gravity_to_world(data):
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

# PD控制器
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_conjugate(q):
    # q: (..., 4) 四元数 (x, y, z, w)
    # 检查是否为PyTorch张量
    if isinstance(q, torch.Tensor):
        x, y, z, w = q.unbind(-1)
        return torch.stack([-x, -y, -z, w], dim=-1)
    else:  # 假设是numpy数组
        q_np = np.array(q)
        result = q_np.copy()
        result[:3] = -result[:3]  # 取反前三个分量
        return result

def quat_mul(q, r):
    # q, r: (..., 4) 四元数 (x, y, z, w)
    # 检查是否为PyTorch张量
    if isinstance(q, torch.Tensor) and isinstance(r, torch.Tensor):
        x1, y1, z1, w1 = q.unbind(-1)
        x2, y2, z2, w2 = r.unbind(-1)
        
        # 正确的四元数乘法公式 (x,y,z,w) 格式
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        
        return torch.stack([x, y, z, w], dim=-1)
    else:  # 假设是numpy数组
        q_np = np.array(q)
        r_np = np.array(r)
        
        x1, y1, z1, w1 = q_np
        x2, y2, z2, w2 = r_np
        
        # 四元数乘法公式
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([x, y, z, w])
def quat_invmul(quat_a, quat_b):
    # quat_a, quat_b: (..., 4) 四元数 (x, y, z, w)
    # 计算相对旋转: quat_B_to_A = quat_A^* ⊗ quat_B
    rel_quat = quat_mul(quat_conjugate(quat_a), quat_b)
    return rel_quat


def get_orientation_2d_from_quat(quat):
    """
    从四元数获取2D方向信息
    quat: (..., 4) 四元数 (x, y, z, w)格式
    返回: 旋转矩阵的前两列，展平为一维数组
    """
    # 检查是否为PyTorch张量，如果是则转换为NumPy数组
    if isinstance(quat, torch.Tensor):
        quat_np = quat.detach().cpu().numpy()
    else:
        quat_np = quat
    # 使用scipy.Rotation将四元数转换为旋转矩阵
    rot = R.from_quat(quat_np)
    anchor_ori = rot.as_matrix()
    # 提取前两列（用于表示2D方向）
    print('matrix:', anchor_ori)
    anchor_ori = anchor_ori[:, :2]
    # 将方向信息展平为一维数组
    anchor_ori = anchor_ori.reshape(-1,)
    return anchor_ori


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




import onnxruntime
import onnx

joint_xml = [
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

# 主程序入口
if __name__ == "__main__":
    
    # 添加初始化， 例如仿真设置， 日志保存路径配置
    log_save_path = "./logs/BM_log.npz"
    # model_path = "D:\RL\deploy\model\BM_ONNX\policy_zuiwu_48000.onnx"
    model_path = "D:\RL\deploy\model\BM_ONNX\dalafan_prog.onnx"

    model = onnx.load(model_path)

    # motion_ref_path = "D:/RL/deploy/deploy_mujoco/bm_traj_npz/dance_zui.npz"
    motion_ref_path = "D:/RL/deploy/deploy_mujoco/bm_traj_npz/dalafan.npz"

    motionref =  np.load(motion_ref_path)
    motionrefpos = motionref["body_pos_w"]
    motionrefquat = motionref["body_quat_w"]
    motionrefinputpos = motionref["joint_pos"]
    motionrefinputvel = motionref["joint_vel"]
    i = 0
    # xml_path = 'D:/RL/deploy-beyondmimic/Beyond_mimic_sim2sim_G1/unitree_description/mjcf/g1_liao.xml'
    xml_path = 'D:/RL/deploy/resources/robots/g1_description/g1_29dof_zy.xml'

    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        if prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
            joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        if prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
            stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
            # stiffness_array = np.array([])
            
        if prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
            damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])        
        
        if prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")
    num_actions = 29
    num_obs = 151  # 154# with yaw
    action = np.zeros(num_actions, dtype=np.float32)
    # target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
                
    policy = onnxruntime.InferenceSession(model_path)
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name
    
    # 加载机器人模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    control_decimation = 10
    simulation_dt = 0.002
    m.opt.timestep = simulation_dt
    simulation_duration = 300.0
    # 动态选择设备 (CPU或CUDA)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")
    counter = 0
    # 初始化 机器人状态
    # d.qpos[0:3] = np.array([0, 0, 0.8])
    logger = Logger(0.02)  # 日志记录器，dt为0.02
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0
    motioninput = np.concatenate((motionrefinputpos[timestep,:],motionrefinputvel[timestep,:]), axis=0)
    motionposcurrent = motionrefpos[timestep,9,:]
    motionquatcurrent = motionrefquat[timestep,9,:]
    target_dof_pos = joint_pos_array.copy()
    d.qpos[7:] = target_dof_pos
    print('d.init_pos:',d.qpos[:7])
    # 添加50度旋转
    import math
    from scipy.spatial.transform import Rotation as R
    
    # 获取当前四元数 (w, x, y, z) 格式
    current_quat = d.qpos[3:7].copy()
    print('原始四元数 (w, x, y, z):', current_quat)
    
    # 将当前四元数转换为scipy需要的格式 (x, y, z, w)
    scipy_quat = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    
    # 创建一个绕Z轴旋转50度的四元数
    angle_rad = math.radians(50)  # 转换为弧度
    rotation = R.from_euler('z', angle_rad)  # 绕Z轴旋转
    rotation_quat = rotation.as_quat()  # 获取四元数 (x, y, z, w)格式
    
    # 使用scipy计算四元数乘法（先旋转50度，再应用原始旋转）
    combined_rot = R.from_quat(scipy_quat) * rotation
    combined_quat_scipy = combined_rot.as_quat()  # (x, y, z, w)格式
    
    # 转换回MuJoCo需要的格式 (w, x, y, z)
    combined_quat_mujoco = np.array([combined_quat_scipy[3], combined_quat_scipy[0], combined_quat_scipy[1], combined_quat_scipy[2]])
    
    # 更新四元数
    d.qpos[3:7] = combined_quat_mujoco
    print('旋转50度后的四元数 (w, x, y, z):', d.qpos[3:7])
    
    body_name = "torso_link"  # robot_ref_body_index=3 motion_ref_body_index=7
    # body_name = "pelvis"
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")
    # 启动mujoco可视化窗口
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 仿真主循环，仿真时间未超过simulation_duration，并且轨迹未结束
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            mujoco.mj_step(m, d)
            tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)# xml
            # break
            d.ctrl[:] = tau
            counter += 1
            if counter % control_decimation == 0:  # 每隔control_decimation步更新一次
                position = d.xpos[body_id]
                # 获取机器人躯干当前的姿态四元数
                # quaternion = d.xquat[body_id]
                quaternion = d.qpos[3:7]
                # 构造观测向量
                # 从运动数据中获取当前时间步的关节位置和速度并拼接
                motioninput = np.concatenate((motionrefinputpos[timestep,:],motionrefinputvel[timestep,:]), axis=0)
                # 获取当前时间步第9个身体部位（通常是参考点）的位置
                motionposcurrent = motionrefpos[timestep,9,:]
                # 获取当前时间步第9个身体部位的四元数
                motionquatcurrent = motionrefquat[timestep,9,:]
                # 计算机器人躯干相对于参考点的旋转四元数quat_rel和前两列旋转矩阵anchor_ori
                quaternion = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
                motionquatcurrent = np.array([motionquatcurrent[1], motionquatcurrent[2], motionquatcurrent[3], motionquatcurrent[0]])
                quat_rel = quat_invmul(quaternion, motionquatcurrent)
                anchor_ori = get_orientation_2d_from_quat(quat_rel)
                # 创建观测向量
                # motion cmd:58;ori:6(唯一的问题是这个oritation怎么计算出来，为什么还要用到pos信息)
                # 填充运动参考输入（58个值）,填充方向信息（6个值）
                obs[0:58] = motioninput
                #**************with yaw***************
                obs[58:61] = project_gravity_to_world(d)  
                obs[61:64] = d.qvel[3 : 6]
                # 获取当前关节位置（从第7个位置开始，因为前7个是根节点位置和四元数）
                qpos_xml = d.qpos[7 : 7 + num_actions]  # joint positions
                # 将XML关节顺序转换为策略期望的关节顺序
                qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                # 计算关节位置相对于默认位置的偏移量，并作为观测值
                obs[64:93] = qpos_seq - joint_pos_array_seq  # joint positions
                # 获取当前关节速度（从第6个位置开始）
                qvel_xml = d.qvel[6 : 6 + num_actions]  # joint positions
                # 将XML关节顺序转换为策略期望的关节顺序
                qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                # 将关节速度作为观测值
                obs[93:122] = qvel_seq  # joint velocities
                # 将上一步的动作作为观测值的一部分（用于实现动作平滑过渡）
                obs[122:151] = action_buffer
                #**************wo yaw***************
                # obs[58:61] = d.qvel[3 : 6]
                # # 获取当前关节位置（从第7个位置开始，因为前7个是根节点位置和四元数）
                # qpos_xml = d.qpos[7 : 7 + num_actions]  # joint positions
                # # 将XML关节顺序转换为策略期望的关节顺序
                # qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                # # 计算关节位置相对于默认位置的偏移量，并作为观测值
                # obs[61:90] = qpos_seq - joint_pos_array_seq  # joint positions
                # # 获取当前关节速度（从第6个位置开始）
                # qvel_xml = d.qvel[6 : 6 + num_actions]  # joint positions
                # # 将XML关节顺序转换为策略期望的关节顺序
                # qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                # # 将关节速度作为观测值
                # obs[90:119] = qvel_seq  # joint velocities
                # # 将上一步的动作作为观测值的一部分（用于实现动作平滑过渡）
                # obs[119:148] = action_buffer
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # 使用ONNX模型预测动作，输入观测值和时间步
                action = policy.run(['actions'], {'obs': obs_tensor.numpy(),'time_step':np.array([timestep], dtype=np.float32).reshape(1,1)})[0]
                print('obs:',obs)
                print('anchor_ori:',obs[58:58+6])
                print('timestep:',timestep)
                # break
                # 将预测动作转换为numpy数组并重塑
                action = np.asarray(action).reshape(-1)
                action_buffer = action.copy()
                # 根据动作缩放因子和默认关节位置，计算目标关节位置
                target_dof_pos = action * action_scale + joint_pos_array_seq
                print('action_scale:', action_scale)
                target_dof_pos = target_dof_pos.reshape(-1,)
                # 将策略关节顺序转换回XML关节顺序
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                # 时间步加1，准备处理下一帧数据
                timestep+=1

            # 同步viewer，刷新显示
                        # 计算PD控制器输出
            # tau = pd_control(target_dof_pos, d.qpos[7:], stiffness_array, np.zeros_like(damping_array), d.qvel[6:], damping_array)# xml
            # # tau = np.clip(tau, -tau_limit, tau_limit) 
            # d.ctrl[:] = tau  # 设置控制输入
            # mujoco.mj_step(m, d)  # 进行一步仿真
            # counter += 1
            viewer.sync()

            # 下面注释掉的代码用于精确控制仿真步长
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                print("time_until_next_step:",time_until_next_step)


# 这个文件采用弹簧外力施加，并且允许记录引力中心和施加外力的索引，方便后续的可视化。
