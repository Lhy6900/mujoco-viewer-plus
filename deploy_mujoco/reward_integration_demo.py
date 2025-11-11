"""
演示如何在 bmwoyaw_multienv.py 中集成 reward_calculator
这个文件展示关键代码片段，不是完整的运行文件
"""

# ==================== 在文件开头导入 ====================
from reward_calculator import compute_rewards

# ==================== 在主循环前初始化 ====================
# 用于保存上一帧的关节速度（计算平滑度）
last_qvel_for_reward = None

# ==================== 在 control_decimation 块中计算奖励 ====================
# 在现有的策略计算代码之后，添加：

if counter % control_decimation == 0:
    # ... 现有的策略推理代码 ...
    
    # ===== 新增：计算奖励 =====
    idx = current_env_idx.value  # 获取当前主环境索引
    
    # 调用奖励计算函数
    rewards = compute_rewards(
        timestep=timestep,
        current_env_idx=idx,
        dlist=dlist,
        target_dof_pos_list=target_dof_pos_list,
        motionrefinputpos=motionrefinputpos,
        motionrefinputvel=motionrefinputvel,
        joint_xml=joint_xml,
        joint_seq=joint_seq,
        last_qvel=last_qvel_for_reward
    )
    
    # 保存当前速度供下一帧使用
    current_joint_vel_seq = np.array([
        dlist[idx].qvel[6 + joint_xml.index(joint)] 
        for joint in joint_seq
    ])
    last_qvel_for_reward = current_joint_vel_seq.copy()
    
    # 打印奖励值（调试用）
    if timestep % 50 == 0:  # 每50步打印一次
        print(f"\n[Rewards @ step {timestep}]")
        for name, value in rewards.items():
            print(f"  {name:25s}: {value:+.6f}")
    
    # ===== 后续步骤 3 会在这里添加 reward_plotter.update(rewards) =====

# ==================== 典型输出示例 ====================
"""
预期的控制台输出：

[Rewards @ step 0]
  smoothness               : +0.000000
  pos_tracking_global      : -0.023456
  pos_tracking_local       : -0.003421
  quat_tracking_global     : -0.045123
  quat_tracking_local      : -0.002891

[Rewards @ step 50]
  smoothness               : -0.012345
  pos_tracking_global      : -0.018234
  pos_tracking_local       : -0.002567
  quat_tracking_global     : -0.034567
  quat_tracking_local      : -0.001234
"""

print("=" * 60)
print("这是一个演示文件，展示如何集成 reward_calculator")
print("请参考注释将代码片段添加到 bmwoyaw_multienv.py")
print("=" * 60)
