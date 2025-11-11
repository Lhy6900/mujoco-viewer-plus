import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

joint_names = [
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

def calculate_mpjpe(actual, target):
    """计算平均每个关节位置误差 (MPJPE)"""
    return np.mean(np.sqrt(np.sum((actual - target)**2, axis=2)), axis=1)

def calculate_per_joint_error(actual, target):
    """计算每个关节的平均位置误差"""
    return np.mean(np.sqrt(np.sum((actual - target)**2, axis=2)), axis=0)

def quat_dot(q1, q2):
    """计算两个四元数的点积"""
    return np.sum(q1 * q2, axis=-1)

def calculate_rotation_error(actual_quats, target_quats):
    """
    计算四元数旋转误差 (角度，弧度)
    
    参数:
    actual_quats: 实际四元数数组，形状为 (N, 4)
    target_quats: 目标四元数数组，形状为 (N, 4)
    
    返回:
    每个时间步的旋转误差数组 (N,)
    """
    # 确保四元数是单位四元数
    actual_quats = actual_quats / np.linalg.norm(actual_quats, axis=-1, keepdims=True)
    target_quats = target_quats / np.linalg.norm(target_quats, axis=-1, keepdims=True)

    # 计算点积的绝对值
    dot_products = np.abs(quat_dot(actual_quats, target_quats))
    
    # 避免浮点误差导致的arccos输入超出[-1, 1]范围
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # 计算角度距离 (2 * arccos(|q1 . q2|))
    # 结果是弧度
    angular_distances = 2 * np.arccos(dot_products)
    
    return angular_distances


def analyze_keypointquat(data, plot, save_dir):
    """
    分析关键点四元数误差 (全局和根相对旋转误差)
    
    参数:
    data: 包含关键点数据的npz文件加载对象
    plot: 是否绘制图表
    save_dir: 保存图表的目录
    """
    print("\n--- 开始分析关键点旋转误差 ---")
    
    if 'target_keypoints_global' not in data.keys() or \
       'actual_keypoints_global' not in data.keys() or \
       'target_keypoints_local' not in data.keys() or \
       'actual_keypoints_local' not in data.keys():
        print("错误: 日志文件中缺少关键点数据。跳过旋转误差分析。")
        return

    # 提取关键点四元数
    # item[:, 3:] 提取的是 (x, y, z, w)
    target_keypoints_global_quat = np.array([item[:, 3:] for item in data['target_keypoints_global']])
    actual_keypoints_global_quat = np.array([item[:, 3:] for item in data['actual_keypoints_global']])
    
    target_keypoints_local_quat = np.array([item[:, 3:] for item in data['target_keypoints_local']])
    actual_keypoints_local_quat = np.array([item[:, 3:] for item in data['actual_keypoints_local']])
    
    num_frames = target_keypoints_global_quat.shape[0]
    num_joints = target_keypoints_global_quat.shape[1] # 29 joints

    # 初始化误差数组 (帧数, 关节数)
    global_rotation_errors = np.zeros((num_frames, num_joints))
    local_rotation_errors = np.zeros((num_frames, num_joints))
    
    # 计算每个关节在每个时间步的旋转误差
    for i in range(num_joints):
        global_rotation_errors[:, i] = calculate_rotation_error(
            actual_keypoints_global_quat[:, i, :], 
            target_keypoints_global_quat[:, i, :]
        )
        local_rotation_errors[:, i] = calculate_rotation_error(
            actual_keypoints_local_quat[:, i, :], 
            target_keypoints_local_quat[:, i, :]
        )

    # 计算平均旋转误差 (跨所有关节和所有帧)
    mean_global_rotation_error = np.mean(global_rotation_errors) # 弧度
    mean_local_rotation_error = np.mean(local_rotation_errors)   # 弧度
    
    # 计算每个关节的平均旋转误差 (跨所有帧)
    per_joint_global_rotation_error = np.mean(global_rotation_errors, axis=0) # 弧度
    per_joint_local_rotation_error = np.mean(local_rotation_errors, axis=0)   # 弧度

    # 转换为角度以便打印
    mean_global_rotation_error_deg = np.degrees(mean_global_rotation_error)
    mean_local_rotation_error_deg = np.degrees(mean_local_rotation_error)
    per_joint_global_rotation_error_deg = np.degrees(per_joint_global_rotation_error)
    per_joint_local_rotation_error_deg = np.degrees(per_joint_local_rotation_error)

    print("\n===== 全局关节旋转误差 (Global Rotation Error) =====")
    print(f"平均值: {mean_global_rotation_error_deg:.4f} 度")
    print(f"标准差: {np.degrees(np.std(global_rotation_errors)):.4f} 度")
    print(f"最大值: {np.degrees(np.max(global_rotation_errors)):.4f} 度")
    print(f"最小值: {np.degrees(np.min(global_rotation_errors)):.4f} 度")

    print("\n===== 根相对关节旋转误差 (Root-relative Rotation Error) =====")
    print(f"平均值: {mean_local_rotation_error_deg:.4f} 度")
    print(f"标准差: {np.degrees(np.std(local_rotation_errors)):.4f} 度")
    print(f"最大值: {np.degrees(np.max(local_rotation_errors)):.4f} 度")
    print(f"最小值: {np.degrees(np.min(local_rotation_errors)):.4f} 度")

    # 输出每个关节的旋转误差
    print(f"\n===== 每个关节的平均旋转误差 (共{num_joints}个关节) =====")
    print("关节ID  |  全局误差(度)  |  根相对误差(度)")
    print("-" * 50)
    for i in range(num_joints):
        print(f"{joint_names[i]:<15} | {per_joint_global_rotation_error_deg[i]:.4f} | {per_joint_local_rotation_error_deg[i]:.4f}")

    # 绘制误差图表
    if plot:
        # 绘制全局和局部旋转误差随时间变化的曲线
        plt.figure(figsize=(12, 6))
        plt.plot(np.mean(global_rotation_errors, axis=1), label='Global Rotation Error (Avg over Joints)')
        plt.plot(np.mean(local_rotation_errors, axis=1), label='Root-relative Rotation Error (Avg over Joints)')
        plt.xlabel('Frame')
        plt.ylabel('Error (rad)')
        plt.title('Average Rotation Error over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 绘制每个关节的平均旋转误差条形图
        plt.figure(figsize=(18, 10)) # 调整 figsize 以适应更多关节名称
        x = np.arange(num_joints)
        width = 0.35
        
        plt.bar(x - width/2, per_joint_global_rotation_error_deg, width, label='Global Error')
        plt.bar(x + width/2, per_joint_local_rotation_error_deg, width, label='Root-relative Error')
        
        plt.xlabel('Joint')
        plt.ylabel('Average Error (degrees)')
        plt.title('Per-joint Average Rotation Error')
        plt.xticks(x, joint_names, rotation=90, fontsize=8) # 旋转标签并减小字体
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout() # 自动调整布局，防止标签重叠
        plt.show()


def analyze_keypointpos(data,plot, save_dir):
    frames = len(data['target_keypoints_global'])
    print(f"总帧数: {frames}")
      
    # 提取全局关节位置
    target_keypoints_global = np.array([item[:, :3] for item in data['target_keypoints_global']])
    actual_keypoints_global = np.array([item[:, :3] for item in data['actual_keypoints_global']])
    
    # 提取局部(根相对)关节位置
    target_keypoints_local = np.array([item[:, :3] for item in data['target_keypoints_local']])
    actual_keypoints_local = np.array([item[:, :3] for item in data['actual_keypoints_local']])
    
    # 计算全局MPJPE
    global_mpjpe = calculate_mpjpe(actual_keypoints_global, target_keypoints_global)
    
    # 计算根相对MPJPE
    local_mpjpe = calculate_mpjpe(actual_keypoints_local, target_keypoints_local)
    
    # 计算每个关节的平均误差
    per_joint_global_error = calculate_per_joint_error(actual_keypoints_global, target_keypoints_global)
    per_joint_local_error = calculate_per_joint_error(actual_keypoints_local, target_keypoints_local)
    
    # 输出统计结果
    print("\n===== 全局关节位置误差 (Global MPJPE) =====")
    print(f"平均值: {np.mean(global_mpjpe):.4f} m")
    print(f"标准差: {np.std(global_mpjpe):.4f} m")
    print(f"最大值: {np.max(global_mpjpe):.4f} m")
    print(f"最小值: {np.min(global_mpjpe):.4f} m")
    
    print("\n===== 根相对关节位置误差 (Root-relative MPJPE) =====")
    print(f"平均值: {np.mean(local_mpjpe):.4f} m")
    print(f"标准差: {np.std(local_mpjpe):.4f} m")
    print(f"最大值: {np.max(local_mpjpe):.4f} m")
    print(f"最小值: {np.min(local_mpjpe):.4f} m")
    
    # 输出每个关节的误差
    # num_joints = len(per_joint_global_error)
    # print(f"\n===== 每个关节的平均误差 (共{num_joints}个关节) =====")
    # print("关节ID  |  全局误差(m)  |  根相对误差(m)")
    # print("-" * 40)
    # for i in range(num_joints):
    #     print(f"{i:7d}  |  {per_joint_global_error[i]:.4f}  |  {per_joint_local_error[i]:.4f}")
    
    # 绘制误差图表
    if plot:
        # 绘制全局和局部MPJPE随时间变化的曲线
        plt.figure(figsize=(12, 6))
        plt.plot(global_mpjpe, label='Global MPJPE')
        plt.plot(local_mpjpe, label='Root-relative MPJPE')
        plt.xlabel('Frame')
        plt.ylabel('Error (m)')
        plt.title('MPJPE over time')
        plt.legend()
        plt.grid(True)
        plt.show()
        # plt.savefig(os.path.join(save_dir, 'mpjpe_over_time.png'))
        
        # 绘制每个关节的平均误差条形图
        num_joints = len(per_joint_global_error)
        plt.figure(figsize=(14, 8))
        x = np.arange(num_joints)
        width = 0.35
        plt.bar(x - width/2, per_joint_global_error, width, label='Global Error')
        plt.bar(x + width/2, per_joint_local_error, width, label='Root-relative Error')
        plt.xlabel('Joint ID')
        plt.ylabel('Average Error (m)')
        plt.title('Per-joint Average Error')
        plt.xticks(x)
        plt.legend()
        plt.grid(True, axis='y')
        plt.show()
        # plt.savefig(os.path.join(save_dir, 'per_joint_error.png'))
        
        print(f"\n图表已保存到 {save_dir} 目录")

def analyze_dof(data,data2, plot, save_dir):
    """分析并绘制DOF位置和速度图"""
    if 'dof_pos' not in data.keys() or 'dof_vel' not in data.keys():
        print("日志文件中未找到'dof_pos'或'dof_vel'。跳过DOF分析。")
        return

    print("\n--- 开始分析DOF数据 ---")
    dof_pos = np.array(data['dof_pos'])
    dof_vel = np.array(data['dof_vel'])
    dof_pos2 = np.array(data2['dof_pos'])
    dof_vel2 = np.array(data2['dof_vel'])

    if plot:
        # 1. 绘制DOF位置图 (每个关节一个子图)
        print("正在绘制29个关节的DOF位置图...")
        fig, axes = plt.subplots(6, 5, figsize=(20, 24))
        fig.suptitle('DOF Positions (rad)', fontsize=18)
        axes = axes.flatten()
        
        for i in range(len(joint_names)):
            ax = axes[i]
            ax.plot(dof_pos[:, i])
            ax.plot(dof_pos2[:, i])
            ax.set_title(joint_names[i], fontsize=10)
            ax.grid(True)
            ax.tick_params(axis='x', labelbottom=False) # 隐藏x轴坐标数字
        
        # 隐藏多余的子图
        for i in range(len(joint_names), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

        # 2. 绘制DOF速度图 (所有关节在一张图上)
        print("正在绘制29个关节的DOF速度图...")
        fig, axes = plt.subplots(6, 5, figsize=(20, 24))
        fig.suptitle('DOF Velocity (rad/s)', fontsize=18)
        axes = axes.flatten()
        
        for i in range(len(joint_names)):
            ax = axes[i]
            ax.plot(dof_vel[:, i])
            ax.plot(dof_vel2[:, i])
            ax.set_title(joint_names[i], fontsize=10)
            ax.grid(True)
            ax.tick_params(axis='x', labelbottom=False) # 隐藏x轴坐标数字
        
        # 隐藏多余的子图
        for i in range(len(joint_names), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


def analyze_compliance(data, data2, plot, save_dir):
    """
    分析机器人对外力的配合情况，并绘制相关图表。
    第一张图：外力xfrc_applied的大小随时间步的变化。
    第二张图：躯干在x方向位移和z方向位置的轨迹图。
    
    参数:
    data: 包含日志数据的npz文件加载对象
    plot: 是否绘制图表
    save_dir: 保存图表的目录 (目前未使用，但保留参数以保持一致性)
    """
    print("\n--- 开始分析机器人对外力的配合情况 ---")

    if 'xfrc_applied' not in data.keys() or 'root_state' not in data.keys():
        print("错误: 日志文件中缺少 'xfrc_applied' 或 'root_state' 数据。跳过配合分析。")
        return

    xfrc_applied_log = np.array(data['xfrc_applied'])
    root_state_log = np.array(data['root_state'])
    root_state_log2 = np.array(data2['root_state'])
    
    # 提取外力大小 (xfrc_applied是3维向量，取其模长)
    # 注意：xfrc_applied可能不是每次控制步都非零，这取决于外部力施加逻辑
    xfrc_magnitude = np.linalg.norm(xfrc_applied_log, axis=1)
    
    # 提取躯干的x和z位置
    # root_state的格式是 [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    root_pos_x = root_state_log[:, 0]
    root_pos_z = root_state_log[:, 2]
    root_pos_x2 = root_state_log2[:, 0]
    root_pos_z2 = root_state_log2[:, 2]


    
    num_frames = len(xfrc_magnitude)
    time_steps = np.arange(num_frames)

    if plot:
        print("正在绘制外力配合图...")
        fig, axes = plt.subplots(2, 1, figsize=(10, 10)) # 创建两个子图，垂直排列
        fig.suptitle('机器人外力配合分析', fontsize=16)

        # 第一个子图：外力大小随时间步的变化
        ax1 = axes[0]
        ax1.plot(time_steps, xfrc_magnitude, label='External Force Magnitude', color='blue')
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('外力大小 (N)')
        ax1.set_title('外力大小随时间步的变化')
        ax1.grid(True)
        ax1.legend()

        # 第二个子图：躯干X-Z平面运动轨迹
        ax2 = axes[1]
        ax2.plot(time_steps[:400], root_pos_x[:400], label='Torso XZ Trajectory', color='red')
        ax2.plot(time_steps[:400], root_pos_x2[:400], label='Torso XZ Trajectory', color='green')

        # ax2.set_ylim('躯干X方向位移 (m)') # 修正：此行是错误的用法，用于设置Y轴范围而非标签
        ax2.set_xlabel('时间步') # 修正：横坐标现在是时间步
        ax2.set_ylabel('躯干X方向位移 (m)') # 修正：纵坐标是躯干X方向位移
        ax2.set_title('躯干X方向位移随时间步的变化')
        ax2.grid(True)
        ax2.legend()
        # ax2.set_aspect('equal', adjustable='box') # 重新添加：保持X和Z轴比例一致

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局，避免重叠
        plt.show()
        
        print("外力配合图表绘制完成。")

def analyze_joint_compliance(data):
    """
    分析关节柔顺性并绘制三维散点图。

    参数:
    gravity_center: 引力中心位置序列 (N, 3)
    target_body_xpos: 目标轨迹位置序列 (N, 3)
    actual_body_xpos: 实际轨迹位置序列 (N, 3)
    xfrc_applied: 外力施加序列 (N, 3)
    
    """
    xfrc_applied = np.array(data['xfrc_applied'])
    gravity_center = np.array(data['gravity_center'])
    force_body_id = np.array(data['force_body_id'])
    target_keypoints_global = np.array([item[:, :3] for item in data['target_keypoints_global']])
    actual_keypoints_global = np.array([item[:, :3] for item in data['actual_keypoints_global']])
    
    print(target_keypoints_global.shape)
    print(actual_keypoints_global.shape)
    target_body_xpos = target_keypoints_global[:, 28, :3]
    actual_body_xpos = actual_keypoints_global[:, 28, :3]
    print(target_body_xpos.shape)
    print(actual_body_xpos.shape)
    print('force_body_id:',force_body_id)
    # 找到有效的时间步序列 S
    valid_indices = np.where((gravity_center is not None) & (np.linalg.norm(xfrc_applied, axis=1) > 1e-6))[0]
    print('length of valid_indices:',len(valid_indices))
    if len(valid_indices) == 0:
        print("没有找到有效的时间步序列。")
        return
    print('valid_indices:',valid_indices)
    # 提取有效的目标和实际位置
    target_positions = target_body_xpos[valid_indices]
    actual_positions = actual_body_xpos[valid_indices]
    gravity_positions = gravity_center[valid_indices]

    # 创建三维散点图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # # 绘制目标轨迹位置（蓝色）
    # ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
    #            color='blue', alpha=0.5, label='Target Trajectory')

    # # 绘制实际轨迹位置（绿色）
    # ax.scatter(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 
    #            color='green', alpha=0.5, label='Actual Trajectory')

    # # 绘制引力中心位置（红色）
    ax.scatter(gravity_positions[:, 0], gravity_positions[:, 1], gravity_positions[:, 2], 
               color='red', label='Gravity Center')

    # 设置透明度
    for i in range(len(valid_indices)):
        alpha = 1 - (i / len(valid_indices))  # 越早的时间步透明度越高
        ax.scatter(target_positions[i, 0], target_positions[i, 1], target_positions[i, 2], 
                   color='blue', alpha=alpha)
        ax.scatter(actual_positions[i, 0], actual_positions[i, 1], actual_positions[i, 2], 
                   color='green', alpha=alpha)

    # 设置图例和标签
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Joint Compliance Analysis')
    ax.legend()

    # 显示图形
    plt.show()

def main():
    log_file = "D:/RL/deploy/logs/0916-testupper/model_56250_cz_origin_pos=0.5_keypoint=0.1_dt=0.001_numenv=10000.npz"
    # log_file = "D:/RL/deploy/logs/7-24-22400-xfrc-1.5s.npz"

    log_file2 = "D:/RL/deploy/logs/0914-testmujoco/NEWSII_model_36400_dt=0.005_origin+LAFAN+nokeypoint.npz"
    log_file3 = "D:/RL/deploy/logs/0916-testupper/F30_model_11000.npz"
    log_file4 = "D:/RL/deploy/logs/0916-testupper/Fm200_model_10900.npz"
    save_dir = "./results"
    plot = True
    
    # 创建保存结果的目录
    if plot:
        os.makedirs(save_dir, exist_ok=True)
    
    # 加载NPZ文件
    print(f"加载日志文件: {log_file}")
    if not os.path.exists(log_file):
        print(f"错误: 找不到日志文件 {log_file}")
        return
        
    data = np.load(log_file, allow_pickle=True)
    data2 = np.load(log_file2, allow_pickle=True)
    data3 = np.load(log_file3, allow_pickle=True)
    data4 = np.load(log_file4, allow_pickle=True)
    
    # 检查文件中的键
    print("NPZ文件中的键:")
    for key in data.keys():
        print(f"  - {key}")
    
    # 提取关键数据
    frames = len(data['target_keypoints_global'])
    print(f"总帧数: {frames}")
    analyze_keypointpos(data,plot,save_dir)
    # analyze_joint_compliance(data)
    # analyze_keypointquat(data,plot,save_dir)
    # analyze_compliance(data, data2,plot, save_dir) # 添加对analyze_compliance的调用
    # analyze_dof(data,data2,plot,save_dir)

if __name__ == "__main__":
    main()