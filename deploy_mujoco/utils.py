"""
Beyond Mimic 工具函数
包含四元数计算、PD控制、重力计算等通用函数
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    PD控制器
    Calculates torques from position commands
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_conjugate(q):
    """
    计算四元数的共轭
    q: (..., 4) 四元数 (x, y, z, w)
    """
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
    """
    四元数乘法
    q, r: (..., 4) 四元数 (x, y, z, w)
    """
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
    """
    计算相对旋转
    quat_a, quat_b: (..., 4) 四元数 (x, y, z, w)
    计算相对旋转: quat_B_to_A = quat_A^* ⊗ quat_B
    """
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


def log_keypoint(target_keypoints_global, target_keypoints_local, data):
    """
    这个函数只是接受target_points_global和target_points_local，
    其实是在用data数据计算actual_global和actual_local
    """
    actual_root_pos = data.qpos[0:3]  
    print('compare pos:', data.qpos[0:3], data.xpos[1])
    actual_root_quat_wxyz = data.qpos[3:7]
    actual_root_quat_xyzw = np.array([actual_root_quat_wxyz[1], 
                                      actual_root_quat_wxyz[2], 
                                      actual_root_quat_wxyz[3], 
                                      actual_root_quat_wxyz[0]])
    actual_keypoints_global = np.zeros([29, 7])
    actual_keypoints_local = np.zeros([29, 7])
    
    for i in range(29):
        actual_keypoints_global[i, :3] = data.xpos[i+2]
        actual_keypoints_global[i, 3:6] = data.xquat[i+2, 1:4]  # 从left_h开始的
        actual_keypoints_global[i, 6:7] = data.xquat[i+2, 0:1]
        
        # 开始局部转换
        actual_keypoints_local[i, :3] = quat_rotate_inverse(
            torch.from_numpy(actual_root_quat_xyzw).view(-1, 4),
            torch.from_numpy(actual_keypoints_global[i, :3] - actual_root_pos).view(-1, 3)
        ).numpy().flatten()  # 在body坐标系下的位置
        
        actual_quat_xyzw = actual_keypoints_global[i, 3:7]
        actual_keypoints_local[i, 3:7] = quat_mul(
            torch.from_numpy(actual_quat_xyzw).view(-1, 4),
            quat_conjugate(torch.from_numpy(actual_root_quat_xyzw).view(-1, 4))
        ).numpy().flatten()
    
    return {
        'target_keypoints_global': target_keypoints_global,
        'target_keypoints_local': target_keypoints_local,
        'actual_keypoints_global': actual_keypoints_global,
        'actual_keypoints_local': actual_keypoints_local,
    }
