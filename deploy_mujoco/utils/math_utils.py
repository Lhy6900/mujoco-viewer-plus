"""
数学工具函数
四元数计算、旋转、距离计算等
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def quat_conjugate(q):
    """
    计算四元数的共轭
    q: (..., 4) 四元数 (x, y, z, w)
    """
    if isinstance(q, torch.Tensor):
        x, y, z, w = q.unbind(-1)
        return torch.stack([-x, -y, -z, w], dim=-1)
    else:
        q_np = np.array(q)
        result = q_np.copy()
        result[:3] = -result[:3]
        return result


def quat_mul(q, r):
    """
    四元数乘法
    q, r: (..., 4) 四元数 (x, y, z, w)
    """
    if isinstance(q, torch.Tensor) and isinstance(r, torch.Tensor):
        x1, y1, z1, w1 = q.unbind(-1)
        x2, y2, z2, w2 = r.unbind(-1)
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        
        return torch.stack([x, y, z, w], dim=-1)
    else:
        q_np = np.array(q)
        r_np = np.array(r)
        
        x1, y1, z1, w1 = q_np
        x2, y2, z2, w2 = r_np
        
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
    if isinstance(quat, torch.Tensor):
        quat_np = quat.detach().cpu().numpy()
    else:
        quat_np = quat
    
    rot = R.from_quat(quat_np)
    anchor_ori = rot.as_matrix()
    anchor_ori = anchor_ori[:, :2]
    anchor_ori = anchor_ori.reshape(-1,)
    return anchor_ori


def quat_rotate_inverse(q, v):
    """
    使用四元数q的逆旋转来旋转向量v
    q: (..., 4) 四元数 (x, y, z, w)
    v: (..., 3) 向量
    返回: v 在 q 的逆旋转下的结果
    """
    q_conj = quat_conjugate(q)
    
    zeros = torch.zeros(v.shape[:-1] + (1,), dtype=v.dtype, device=v.device)
    v_as_quat = torch.cat([v, zeros], dim=-1)
    
    result = quat_mul(quat_mul(q_conj, v_as_quat), q)
    
    return result[..., :3]


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
    gravity_orientation[0] = 2 * (x*z + w*y)
    gravity_orientation[1] = 2 * (y*z - w*x)
    gravity_orientation[2] = 1 - 2 * (x*x + y*y)
    return gravity_orientation


def project_gravity_to_world(data):
    """
    计算重力向量在机体坐标系中的表示
    data: mujoco.MjData对象
    返回: 重力向量在机体坐标系中的表示
    """
    quat = np.array([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])
    rot_mat = R.from_quat(quat).as_matrix()
    base_proj_gravity = np.matmul(rot_mat.T, np.array([0, 0, -1.0]))
    return base_proj_gravity


def compute_distance(pos1, pos2):
    """
    计算两点之间的欧式距离
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
