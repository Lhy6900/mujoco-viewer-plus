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
    model_path = "/home/ubuntu/deploy_mini/model/BM_ONNX/dalafan_woyaw.onnx"

    model = onnx.load(model_path)

    # motion_ref_path = "D:/RL/deploy/deploy_mujoco/bm_traj_npz/dance_zui.npz"
    motion_ref_path = "/home/ubuntu/deploy_mini/deploy_mujoco/bm_traj_npz/dalafan.npz"

    motionref =  np.load(motion_ref_path)
    motionrefpos = motionref["body_pos_w"]
    motionrefquat = motionref["body_quat_w"]
    motionrefinputpos = motionref["joint_pos"]
    motionrefinputvel = motionref["joint_vel"]
    i = 0
    # xml_path = 'D:/RL/deploy-beyondmimic/Beyond_mimic_sim2sim_G1/unitree_description/mjcf/g1_liao.xml'
    xml_path = '/home/ubuntu/deploy_mini/resources/robots/g1_description/g1_29dof_zy.xml'

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
    num_obs = 148  # 154# with yaw
    num_envs = 4
    action = np.zeros(num_actions, dtype=np.float32)
    # target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    # 初始化主策略和多环境策略列表
    policy = onnxruntime.InferenceSession(model_path)
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name
    policy_list = []
    input_name_list = []
    output_name_list = []
    for i in range(num_envs):
        policy_list.append(onnxruntime.InferenceSession(model_path))
        input_name_list.append(policy_list[-1].get_inputs()[0].name)
        output_name_list.append(policy_list[-1].get_outputs()[0].name)

    # 加载机器人模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    dlist= []       # 多环境数据列表
    for i in range(num_envs):
        dlist.append(mujoco.MjData(m))
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
    target_dof_pos = joint_pos_array.copy()
    d.qpos[7:] = target_dof_pos
    # 多环境初始化，包括action_buffer,timestep,motion_input还有默认位置，注意下面两个for循环不能合并，timestep列表要先完成初始化
    action_buffer_list = []
    timestep_list = []
    for i in range(num_envs):
        action_buffer_list.append( np.zeros((num_actions,), dtype=np.float32))
        timestep_list.append(0)
    motioninput = np.concatenate((motionrefinputpos[timestep,:],motionrefinputvel[timestep,:]), axis=0)
    motionposcurrent = motionrefpos[timestep,9,:]
    motionquatcurrent = motionrefquat[timestep,9,:]
    motion_input_list = []
    motionposcurrent_list = []
    motionquatcurrent_list = []
    target_dof_pos_list = []
    for i in range(num_envs):
        timestep_i = timestep_list[i]
        motioninput_i = np.concatenate((motionrefinputpos[timestep_i,:],motionrefinputvel[timestep_i,:]), axis=0)
        motionposcurrent_i = motionrefpos[timestep_i,9,:]
        motionquatcurrent_i = motionrefquat[timestep_i,9,:]
        motion_input_list.append(motioninput_i)
        motionposcurrent_list.append(motionposcurrent_i)
        motionquatcurrent_list.append(motionquatcurrent_i)
        target_dof_pos_list.append(joint_pos_array.copy())
        dlist[i].qpos[7:] = target_dof_pos_list[i]
        dlist[i].qpos[0] = 0.5 * (i + 1)  # 每个环境的x位置错开一些
        dlist[i].qpos[1] = 0.5 * (i + 1)  # 每个环境的y位置错开一些

    body_name = "torso_link"  # robot_ref_body_index=3 motion_ref_body_index=7
    # body_name = "pelvis"
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")
    
    # 当前选择的主环境索引（用于交互）
    current_env_idx = Value('i', 0)  # 使用共享内存变量
    
    # 初始化渲染所需的MjvOption和MjvPerturb对象
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
    
    # 定义快捷键切换环境的回调函数
    def key_callback(key):
        # 逗号键 - 切换到上一个环境
        if key == 44:  # ord(',')
            current_env_idx.value = (current_env_idx.value - 1) % num_envs
            print(f"[切换环境] 当前主环境: {current_env_idx.value}")
        # 句号键 - 切换到下一个环境
        elif key == 46:  # ord('.')
            current_env_idx.value = (current_env_idx.value + 1) % num_envs
            print(f"[切换环境] 当前主环境: {current_env_idx.value}")
    
    # 启动mujoco可视化窗口，传入键盘回调
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        # 仿真主循环，仿真时间未超过simulation_duration，并且轨迹未结束
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # 先执行所有环境的仿真步
            for i in range(num_envs):
                mujoco.mj_step(m, dlist[i])
                tau_i = pd_control(target_dof_pos_list[i], dlist[i].qpos[7:], stiffness_array, np.zeros_like(damping_array), dlist[i].qvel[6:], damping_array)
                dlist[i].ctrl[:] = tau_i
            
            # 将当前选中的环境数据同步到viewer的主数据d
            idx = current_env_idx.value
            d.qpos[:] = dlist[idx].qpos[:]
            d.qvel[:] = dlist[idx].qvel[:]
            d.ctrl[:] = dlist[idx].ctrl[:]
            mujoco.mj_forward(m, d)
            
            # 如果有外力扰动（通过viewer交互施加），将其应用回选中的环境
            if np.any(d.xfrc_applied != 0):
                dlist[idx].xfrc_applied[:] = d.xfrc_applied[:]
            
            # counter 共用
            counter += 1
            if counter % control_decimation == 0:  # 每隔control_decimation步更新一次
                position = d.xpos[body_id]
                quaternion = d.qpos[3:7]
                motioninput = np.concatenate((motionrefinputpos[timestep,:],motionrefinputvel[timestep,:]), axis=0)
                motionposcurrent = motionrefpos[timestep,9,:]
                motionquatcurrent = motionrefquat[timestep,9,:]
                quaternion = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
                motionquatcurrent = np.array([motionquatcurrent[1], motionquatcurrent[2], motionquatcurrent[3], motionquatcurrent[0]])
                quat_rel = quat_invmul(quaternion, motionquatcurrent)
                anchor_ori = get_orientation_2d_from_quat(quat_rel)
                obs[0:58] = motioninput
                obs[58:61] = d.qvel[3 : 6]
                qpos_xml = d.qpos[7 : 7 + num_actions]  # joint positions
                qpos_seq = np.array([qpos_xml[joint_xml.index(joint)] for joint in joint_seq])
                obs[61:90] = qpos_seq - joint_pos_array_seq  # joint positions
                qvel_xml = d.qvel[6 : 6 + num_actions]  # joint positions
                qvel_seq = np.array([qvel_xml[joint_xml.index(joint)] for joint in joint_seq])
                obs[90:119] = qvel_seq  # joint velocities
                obs[119:148] = action_buffer
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy.run(['actions'], {'obs': obs_tensor.numpy(),'time_step':np.array([timestep], dtype=np.float32).reshape(1,1)})[0]                    #
                # break
                # 将预测动作转换为numpy数组并重塑
                action = np.asarray(action).reshape(-1)
                action_buffer = action.copy()
                # 根据动作缩放因子和默认关节位置，计算目标关节位置
                target_dof_pos = action * action_scale + joint_pos_array_seq
                target_dof_pos = target_dof_pos.reshape(-1,)
                # 将策略关节顺序转换回XML关节顺序
                target_dof_pos = np.array([target_dof_pos[joint_seq.index(joint)] for joint in joint_xml])
                # 时间步加1，准备处理下一帧数据
                timestep+=1
                for i in range(num_envs):
                    timestep_i = timestep_list[i]
                    motioninput_i = np.concatenate((motionrefinputpos[timestep_i,:],motionrefinputvel[timestep_i,:]), axis=0)
                    motionposcurrent_i = motionrefpos[timestep_i,9,:]
                    motionquatcurrent_i = motionrefquat[timestep_i,9,:]
                    quaternion_i = dlist[i].qpos[3:7]
                    quaternion_i = np.array([quaternion_i[1], quaternion_i[2], quaternion_i[3], quaternion_i[0]])
                    motionquatcurrent_i = np.array([motionquatcurrent_i[1], motionquatcurrent_i[2], motionquatcurrent_i[3], motionquatcurrent_i[0]])
                    quat_rel_i = quat_invmul(quaternion_i, motionquatcurrent_i)
                    anchor_ori_i = get_orientation_2d_from_quat(quat_rel_i)
                    obs_i = np.zeros(num_obs, dtype=np.float32)
                    obs_i[0:58] = motioninput_i
                    obs_i[58:61] = dlist[i].qvel[3 : 6]
                    qpos_xml_i = dlist[i].qpos[7 : 7 + num_actions]  # joint positions
                    qpos_seq_i = np.array([qpos_xml_i[joint_xml.index(joint)] for joint in joint_seq])
                    obs_i[61:90] = qpos_seq_i - joint_pos_array_seq  # joint positions
                    qvel_xml_i = dlist[i].qvel[6 : 6 + num_actions]  # joint positions
                    qvel_seq_i = np.array([qvel_xml_i[joint_xml.index(joint)] for joint in joint_seq])
                    obs_i[90:119] = qvel_seq_i  # joint velocities
                    obs_i[119:148] = action_buffer_list[i]
                    obs_tensor_i = torch.from_numpy(obs_i).unsqueeze(0)
                    action_i = policy_list[i].run(['actions'], {'obs': obs_tensor_i.numpy(),'time_step':np.array([timestep_i], dtype=np.float32).reshape(1,1)})[0]
                    # 将预测动作转换为numpy数组并重塑
                    action_i = np.asarray(action_i).reshape(-1)
                    action_buffer_list[i] = action_i.copy()
                    target_dof_pos_i = action_i * action_scale + joint_pos_array_seq
                    target_dof_pos_i = target_dof_pos_i.reshape(-1,)
                    # 将策略关节顺序转换回XML关节顺序
                    target_dof_pos_i = np.array([target_dof_pos_i[joint_seq.index(joint)] for joint in joint_xml])
                    target_dof_pos_list[i] = target_dof_pos_i
                    # 时间步加1，准备处理下一帧数据
                    timestep_list[i] +=1

            # 同步viewer，刷新显示
            # 清空user_scn中的geoms，为渲染其他环境做准备
            viewer.user_scn.ngeom = 0
            
            # 渲染所有其他环境（除了主环境current_env_idx）
            for i in range(num_envs):
                if i == current_env_idx.value:
                    continue  # 跳过当前主环境（已经在viewer的主场景中显示）
                # 使用mjv_addGeoms将每个环境的机器人渲染到viewer中
                mujoco.mjv_addGeoms(
                    m,           # 模型
                    dlist[i],    # 该环境的数据
                    vopt,        # 可视化选项
                    pert,        # 扰动（未使用）
                    catmask,     # 类别掩码（显示动态物体）
                    viewer.user_scn  # 添加到viewer的用户场景中
                )
            
            viewer.sync()

            # 下面注释掉的代码用于精确控制仿真步长
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                print("time_until_next_step:",time_until_next_step)


# 这个文件采用弹簧外力施加，并且允许记录引力中心和施加外力的索引，方便后续的可视化。
