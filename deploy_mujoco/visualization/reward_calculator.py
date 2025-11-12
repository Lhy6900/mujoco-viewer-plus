"""
Reward Calculator for BeyondMimic Multi-Environment Simulation
计算跟踪误差、平滑度等奖励项
参考 deploy_robot-main-v1.0/deploy_robot/sim/visualization/policy_viz_utils.py
"""

import numpy as np
from typing import Dict, Optional, List


def compute_rewards(
    timestep: int,
    current_env_idx: int,
    dlist: List,
    target_dof_pos_list: List[np.ndarray],
    motionrefinputpos: np.ndarray,
    motionrefinputvel: np.ndarray,
    joint_xml: List[str],
    joint_seq: List[str],
    last_qvel: Optional[np.ndarray] = None,
    last_joint_pos: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算当前帧的奖励项
    
    Args:
        timestep: 当前时间步
        current_env_idx: 当前主环境索引
        dlist: 所有环境的 MjData 列表
        target_dof_pos_list: 所有环境的目标关节位置列表
        motionrefinputpos: 参考运动的关节位置 [T, num_joints]
        motionrefinputvel: 参考运动的关节速度 [T, num_joints]
        joint_xml: XML 中的关节顺序
        joint_seq: 策略输出的关节顺序
        last_qvel: 上一帧的关节速度（用于计算平滑度）
        last_joint_pos: 上一帧的关节位置（备用）
    
    Returns:
        奖励字典 {term_name: value}
        所有奖励值为负数（惩罚），越接近0越好
    """
    rewards = {
        "smoothness": 0.0,              # 运动平滑度（速度变化）
        "pos_tracking_global": 0.0,      # 全局位置跟踪误差（L2范数）
        "pos_tracking_local": 0.0,       # 局部位置跟踪误差（平均绝对值）
        "quat_tracking_global": 0.0,     # 基座姿态跟踪误差
        "quat_tracking_local": 0.0,      # 关节角度跟踪误差
    }
    
    try:
        # 获取当前环境的数据
        data = dlist[current_env_idx]
        
        # 当前关节位置和速度（从 XML 顺序）
        current_joint_pos_xml = data.qpos[7:7+len(joint_xml)]  # 跳过前7个自由度（base）
        current_joint_vel_xml = data.qvel[6:6+len(joint_xml)]  # 跳过前6个自由度（base）
        
        # 转换到策略顺序
        current_joint_pos_seq = np.array([
            current_joint_pos_xml[joint_xml.index(joint)] 
            for joint in joint_seq
        ])
        current_joint_vel_seq = np.array([
            current_joint_vel_xml[joint_xml.index(joint)] 
            for joint in joint_seq
        ])
        
        # ========== 1. 平滑度奖励 ==========
        # 惩罚关节速度的剧烈变化（加速度）
        if last_qvel is not None:
            # 计算速度变化（加速度的近似）
            joint_vel_change = np.linalg.norm(current_joint_vel_seq - last_qvel)
            # 负值表示惩罚，变化越大惩罚越大
            rewards["smoothness"] = -joint_vel_change
        else:
            rewards["smoothness"] = 0.0
        
        # ========== 2. 位置跟踪奖励（全局）==========
        # 当前目标位置 vs 实际位置（XML 顺序）
        target_joint_pos_xml = target_dof_pos_list[current_env_idx]
        
        # 转换目标位置到策略顺序
        target_joint_pos_seq = np.array([
            target_joint_pos_xml[joint_xml.index(joint)] 
            for joint in joint_seq
        ])
        
        # 全局跟踪误差：L2 范数
        pos_error_global = np.linalg.norm(current_joint_pos_seq - target_joint_pos_seq)
        rewards["pos_tracking_global"] = -pos_error_global
        
        # ========== 3. 位置跟踪奖励（局部）==========
        # 局部跟踪误差：平均绝对误差（对单个关节更敏感）
        pos_error_local = np.mean(np.abs(current_joint_pos_seq - target_joint_pos_seq))
        rewards["pos_tracking_local"] = -pos_error_local
        
        # ========== 4. 姿态跟踪奖励（全局）==========
        # 基座姿态跟踪：当前四元数 vs 参考四元数
        if timestep < len(motionrefinputpos):
            # 当前基座姿态（四元数 [w, x, y, z]）
            current_base_quat = data.qpos[3:7]  # MuJoCo 格式：[w, x, y, z]
            
            # 参考姿态（如果有的话，这里简化为速度的模作为代理）
            # 注：你的代码中没有直接的基座姿态参考，这里用关节速度误差代替
            ref_joint_vel = motionrefinputvel[timestep, :]
            vel_error = np.linalg.norm(current_joint_vel_seq - ref_joint_vel)
            rewards["quat_tracking_global"] = -vel_error * 0.1  # 缩放系数
        else:
            rewards["quat_tracking_global"] = 0.0
        
        # ========== 5. 姿态跟踪奖励（局部）==========
        # 参考轨迹的关节位置跟踪
        if timestep < len(motionrefinputpos):
            ref_joint_pos = motionrefinputpos[timestep, :]
            # 局部姿态误差：与参考轨迹的关节位置差异
            ref_pos_error = np.mean(np.abs(current_joint_pos_seq - ref_joint_pos))
            rewards["quat_tracking_local"] = -ref_pos_error
        else:
            rewards["quat_tracking_local"] = 0.0
            
    except Exception as e:
        print(f"[警告] compute_rewards 计算失败: {e}")
        # 返回零值
        for key in rewards:
            rewards[key] = 0.0
    
    return rewards


def compute_rewards_simple(
    current_joint_pos: np.ndarray,
    target_joint_pos: np.ndarray,
    current_joint_vel: np.ndarray,
    last_joint_vel: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    简化版奖励计算（用于快速测试）
    
    Args:
        current_joint_pos: 当前关节位置
        target_joint_pos: 目标关节位置
        current_joint_vel: 当前关节速度
        last_joint_vel: 上一帧关节速度
    
    Returns:
        奖励字典
    """
    rewards = {
        "smoothness": 0.0,
        "pos_tracking_global": 0.0,
        "pos_tracking_local": 0.0,
        "quat_tracking_global": 0.0,
        "quat_tracking_local": 0.0,
    }
    
    # 平滑度
    if last_joint_vel is not None:
        vel_change = np.linalg.norm(current_joint_vel - last_joint_vel)
        rewards["smoothness"] = -vel_change
    
    # 位置跟踪
    pos_error = np.linalg.norm(current_joint_pos - target_joint_pos)
    rewards["pos_tracking_global"] = -pos_error
    rewards["pos_tracking_local"] = -np.mean(np.abs(current_joint_pos - target_joint_pos))
    
    # 简化的姿态奖励（使用速度的平方作为代理）
    vel_magnitude = np.linalg.norm(current_joint_vel)
    rewards["quat_tracking_global"] = -vel_magnitude * 0.1
    rewards["quat_tracking_local"] = -np.std(current_joint_vel) * 0.1
    
    return rewards


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Reward Calculator 测试")
    print("=" * 60)
    
    # 模拟数据
    num_joints = 29
    joint_xml = [f"joint_{i}" for i in range(num_joints)]
    joint_seq = joint_xml.copy()  # 假设顺序相同
    
    # 当前状态
    current_joint_pos = np.random.randn(num_joints) * 0.1
    target_joint_pos = np.random.randn(num_joints) * 0.1
    current_joint_vel = np.random.randn(num_joints) * 0.5
    last_joint_vel = np.random.randn(num_joints) * 0.5
    
    print("\n测试简化版奖励计算...")
    rewards = compute_rewards_simple(
        current_joint_pos,
        target_joint_pos,
        current_joint_vel,
        last_joint_vel
    )
    
    print("\n计算结果:")
    for name, value in rewards.items():
        print(f"  {name:25s}: {value:+.6f}")
    
    # 检查奖励范围
    print("\n奖励统计:")
    values = list(rewards.values())
    print(f"  最小值: {min(values):+.6f}")
    print(f"  最大值: {max(values):+.6f}")
    print(f"  平均值: {np.mean(values):+.6f}")
    
    # 测试多次更新
    print("\n测试连续更新（模拟10步）...")
    last_vel = None
    for step in range(10):
        pos = np.random.randn(num_joints) * 0.1
        target = np.random.randn(num_joints) * 0.1
        vel = np.random.randn(num_joints) * 0.5
        
        r = compute_rewards_simple(pos, target, vel, last_vel)
        last_vel = vel
        
        if step % 3 == 0:
            print(f"  Step {step}: smoothness={r['smoothness']:+.4f}, "
                  f"pos_global={r['pos_tracking_global']:+.4f}")
    
    print("\n✅ Reward Calculator 测试完成！")
    print("\n说明:")
    print("  - 所有奖励值为负数（惩罚），越接近0越好")
    print("  - smoothness: 速度变化越小越好（平滑运动）")
    print("  - pos_tracking: 位置误差越小越好（精确跟踪）")
    print("  - quat_tracking: 姿态/速度误差越小越好")
