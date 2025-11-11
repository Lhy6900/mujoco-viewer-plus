import time  # 导入time模块，用于计时
import mujoco.viewer  # 导入mujoco的viewer模块，用于可视化仿真
import mujoco  # 导入mujoco主模块
import numpy as np  # 导入numpy，用于数值计算
import torch  # 导入PyTorch
import argparse
import onnxruntime
import onnx
from multiprocessing import Value  # 导入多进程相关类

# 导入配置、工具函数和日志类
from bmconfig import *
from utils import (
    pd_control, quat_conjugate, quat_mul, quat_invmul,
    get_orientation_2d_from_quat, quat_rotate_inverse
)
from bm_logger import BMLogger
from reward_calculator import compute_rewards
from reward_plotter import RewardPlotter
from ghost_renderer import GhostRenderer

# 主程序入口
if __name__ == "__main__":
    # 从配置文件导入常量
    joint_xml = JOINT_XML
    model_path = MODEL_PATH
    motion_ref_path = MOTION_REF_PATH
    xml_path = XML_PATH
    body_name = BODY_NAME
    
    # 加载 ONNX 模型
    model = onnx.load(model_path)

    # 加载运动参考数据
    motionref = np.load(motion_ref_path)
    motionrefpos = motionref["body_pos_w"]
    motionrefquat = motionref["body_quat_w"]
    motionrefinputpos = motionref["joint_pos"]
    motionrefinputvel = motionref["joint_vel"]

    # 从模型元数据中读取配置
    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = prop.value.split(",")
        if prop.key == "default_joint_pos":   
            joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
            joint_pos_array = np.array([joint_pos_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        if prop.key == "joint_stiffness":
            stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
            stiffness_array = np.array([stiffness_array_seq[joint_seq.index(joint)] for joint in joint_xml])
        if prop.key == "joint_damping":
            damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
            damping_array = np.array([damping_array_seq[joint_seq.index(joint)] for joint in joint_xml])        
        if prop.key == "action_scale":
            action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(f"{prop.key}: {prop.value}")
    
    # 从配置文件导入常量
    num_actions = NUM_ACTIONS
    num_obs = NUM_OBS
    
    # 通过命令行参数设置并发环境数量
    parser = argparse.ArgumentParser(description='Run multi-env BM w/ MuJoCo')
    parser.add_argument('--num_envs', type=int, default=DEFAULT_NUM_ENVS, help='number of parallel environments')
    args = parser.parse_args()
    num_envs = int(args.num_envs)
    print(f"使用并行环境数量 num_envs={num_envs}")
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
    dlist = []  # 多环境数据列表
    for i in range(num_envs):
        dlist.append(mujoco.MjData(m))
    
    # 从配置文件导入仿真参数
    control_decimation = CONTROL_DECIMATION
    simulation_dt = SIMULATION_DT
    m.opt.timestep = simulation_dt
    simulation_duration = SIMULATION_DURATION
    
    # 动态选择设备 (CPU或CUDA)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用的设备: {device}")
    counter = 0
    
    # 计算网格布局的环境位置
    def compute_env_origins_grid(num_envs, env_spacing=3.0):
        """计算网格布局的环境原点位置（参考 mjlab）"""
        env_origins = np.zeros((num_envs, 2))
        # 计算网格行列数：尽可能接近正方形
        num_rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
        num_cols = int(np.ceil(num_envs / num_rows))
        
        # 生成网格索引
        ii, jj = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
        
        # 计算每个环境的位置（居中布局）
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing  # x
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing   # y        
        return env_origins
    
    # 计算环境位置
    env_spacing = 3.0  # 环境间距，可以根据机器人大小调整
    env_origins = compute_env_origins_grid(num_envs, env_spacing)
    print(f"环境布局：{int(np.ceil(num_envs / int(np.sqrt(num_envs))))}行 x {int(np.ceil(num_envs / int(np.ceil(num_envs / int(np.sqrt(num_envs))))))}列")
    
    # 初始化日志记录器
    logger = BMLogger(LOGGER_DT)
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    timestep = 0
    target_dof_pos = joint_pos_array.copy()
    d.qpos[7:] = target_dof_pos
    
    # 用于奖励计算的变量
    last_qvel_for_reward = None  # 保存上一帧的关节速度（用于计算平滑度）
    
    # 初始化奖励绘图器
    reward_plotter = RewardPlotter(history_length=300)
    reward_plotter.register_terms([
        "smoothness",
        "pos_tracking_global",
        "pos_tracking_local",
        "quat_tracking_global",
        "quat_tracking_local"
    ])
    print("[奖励可视化] RewardPlotter 已初始化，图表将显示在屏幕右侧")
    
    # 初始化 Ghost 渲染器
    ghost_renderer = GhostRenderer(m)
    print("[Ghost 可视化] GhostRenderer 已初始化，将显示半透明绿色参考轨迹")
    
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
        # 使用网格布局设置每个环境的初始位置
        dlist[i].qpos[0] = env_origins[i, 0]  # x 位置
        dlist[i].qpos[1] = env_origins[i, 1]  # y 位置

    # 获取body id
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found in model")
    
    # 当前选择的主环境索引（用于交互）
    current_env_idx = Value('i', 0)  # 使用共享内存变量
    show_other_envs = Value('i', 0)  # 是否显示其他环境（0=不显示，1=显示）
    show_reward_plot = Value('i', 1)  # 是否显示奖励曲线（0=不显示，1=显示，默认显示）
    show_ghost = Value('i', 1)  # 是否显示 ghost 参考轨迹（0=不显示，1=显示，默认显示）
    
    # 用于检测 Ctrl 键状态
    ctrl_pressed = Value('i', 0)  # 0=未按下，1=已按下
    
    # 初始化渲染所需的MjvOption和MjvPerturb对象
    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
    
    # 定义快捷键切换环境的回调函数
    def key_callback(key):
        # 检测 Ctrl 键按下和释放
        if key == 341 or key == 345:  # GLFW_KEY_LEFT_CONTROL 或 GLFW_KEY_RIGHT_CONTROL
            ctrl_pressed.value = 1
            return
        
        # 上箭头键 - 切换到上一个环境
        if key == 265:  # GLFW_KEY_UP
            current_env_idx.value = (current_env_idx.value - 1) % num_envs
            print(f"[切换环境] 当前主环境: {current_env_idx.value}")
        # 下箭头键 - 切换到下一个环境
        elif key == 264:  # GLFW_KEY_DOWN
            current_env_idx.value = (current_env_idx.value + 1) % num_envs
            print(f"[切换环境] 当前主环境: {current_env_idx.value}")
        # M键 - 切换是否显示其他环境
        elif key == 77 or key == 109:  # 'M' 或 'm'
            show_other_envs.value = 1 - show_other_envs.value
            status = "显示" if show_other_envs.value else "隐藏"
            print(f"[多环境渲染] {status}其他环境")
        # Ctrl+R - 切换奖励曲线显示
        elif (key == 82 or key == 114) and ctrl_pressed.value:  # 'R' 或 'r' + Ctrl
            show_reward_plot.value = 1 - show_reward_plot.value
            status = "显示" if show_reward_plot.value else "隐藏"
            print(f"[奖励可视化] {status}奖励曲线窗口")
            ctrl_pressed.value = 0  # 重置 Ctrl 状态
        # Ctrl+G - 切换 ghost 显示
        elif (key == 71 or key == 103) and ctrl_pressed.value:  # 'G' 或 'g' + Ctrl
            show_ghost.value = 1 - show_ghost.value
            status = "显示" if show_ghost.value else "隐藏"
            print(f"[Ghost 可视化] {status}参考轨迹")
            ctrl_pressed.value = 0  # 重置 Ctrl 状态
    
    # 启动mujoco可视化窗口，传入键盘回调
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        # 仿真主循环，仿真时间未超过simulation_duration，并且轨迹未结束
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # 将当前选中的环境数据同步到viewer的主数据d
            idx = current_env_idx.value
            d.qpos[:] = dlist[idx].qpos[:]
            d.qvel[:] = dlist[idx].qvel[:]
            d.ctrl[:] = dlist[idx].ctrl[:]
            # 注意：不复制 xfrc_applied，让 viewer 交互系统自己管理
            mujoco.mj_forward(m, d)
            
            # 执行所有环境的仿真步
            for i in range(num_envs):
                tau_i = pd_control(target_dof_pos_list[i], dlist[i].qpos[7:], stiffness_array, np.zeros_like(damping_array), dlist[i].qvel[6:], damping_array)
                dlist[i].ctrl[:] = tau_i
                mujoco.mj_step(m, dlist[i])
            
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
                
                # ===== 计算奖励（新增）=====
                try:
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
                    
                    # 每50步打印一次奖励值
                    if timestep % 50 == 0:
                        print(f"\n[Rewards @ step {timestep}]")
                        for name, value in rewards.items():
                            print(f"  {name:25s}: {value:+.6f}")
                    
                    # 更新奖励绘图器
                    reward_plotter.update(rewards)
                    
                except Exception as e:
                    print(f"[警告] 奖励计算失败: {e}")
                # ===== 奖励计算结束 =====
                
                # ===== 构造 Ghost Qpos（新增）=====
                try:
                    # 使用 policy.run 获取 body_pos_w 和 body_quat_w
                    ghost_outputs = policy.run(
                        ['body_pos_w', 'body_quat_w'],
                        {
                            'obs': obs_tensor.numpy(),
                            'time_step': np.array([timestep], dtype=np.float32).reshape(1,1)
                        }
                    )
                    
                    # 从返回的列表中提取数据
                    policy_body_pos_w = ghost_outputs[0]   # (1, 14, 3)
                    policy_body_quat_w = ghost_outputs[1]  # (1, 14, 4)
                    
                    # 从 policy 输出提取 root 位置和姿态
                    # 使用第一个 body（索引 0）作为 base/root
                    base_pos_policy = policy_body_pos_w[0, 0, :]  # (3,) - XYZ 位置
                    base_quat_policy = policy_body_quat_w[0, 0, :]  # (4,) - 四元数    
                    base_quat_mujoco = base_quat_policy
                    
                    # 提取参考轨迹的关节角度（joint_seq 顺序）
                    joint_pos_ref_seq = motionrefinputpos[timestep, :]  # (num_joints,)
                    
                    # 转换为 dof 顺序（joint_xml 顺序）
                    ghost_joint_pos_dof = np.array([
                        joint_pos_ref_seq[joint_seq.index(joint)] 
                        for joint in joint_xml
                    ])
                    
                    # 构造完整的 ghost_qpos
                    ghost_qpos = ghost_renderer.construct_ghost_qpos(
                        base_pos=base_pos_policy,
                        base_quat=base_quat_mujoco,
                        joint_pos_dof_order=ghost_joint_pos_dof,
                        current_qpos=d.qpos
                    )
                    
                    # 设置 ghost 姿态
                    ghost_renderer.set_ghost_qpos(ghost_qpos)
                    
                except Exception as e:
                    if timestep % 100 == 0:  # 每100步打印一次错误
                        print(f"[警告] Ghost qpos 构造失败: {e}")
                # ===== Ghost Qpos 构造结束 =====
                
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
                    # ===== 渲染奖励图表（新增）=====
                # 只有在 show_reward_plot 为 True 时才渲染奖励曲线
                if show_reward_plot.value:
                    try:                
                        # 获取图表布局
                        figures_data = reward_plotter.get_figures_for_viewer(viewer.viewport)
                        viewer.set_figures(figures_data)
                                
                    except Exception as e:
                        # 静默失败，避免影响主仿真
                        if timestep % 100 == 0:  # 每100步打印一次错误
                            print(f"[警告] 奖励图表渲染失败: {e}")
                else:
                    # 隐藏奖励曲线时清空图表
                    try:
                        viewer.set_figures([])
                    except:
                        pass
                # ===== 奖励图表渲染结束 =====

            # 同步viewer，刷新显示
            # 清空user_scn中的geoms，为渲染其他环境做准备
            viewer.user_scn.ngeom = 0
            
            # ===== 渲染 Ghost（新增）=====
            if show_ghost.value:
                try:
                    ghost_renderer.render_ghost(viewer.user_scn)
                except Exception as e:
                    if timestep % 100 == 0:  # 每100步打印一次错误
                        print(f"[警告] Ghost 渲染失败: {e}")
            # ===== Ghost 渲染结束 =====
            
            # 只有在 show_other_envs 为 True 时才渲染其他环境
            if show_other_envs.value:
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
            
            # 在 viewer.sync() 之后，检查用户是否通过交互施加了外力
            # viewer.sync() 会处理鼠标拖拽等交互，并更新 d.xfrc_applied
            if np.any(d.xfrc_applied != 0):
                # 将外力复制到当前选中的环境
                dlist[current_env_idx.value].xfrc_applied[:] = d.xfrc_applied[:]
            else:
                d.xfrc_applied[:] = 0
                dlist[current_env_idx.value].xfrc_applied[:] = 0

            # 下面注释掉的代码用于精确控制仿真步长
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                print("time_until_next_step:",time_until_next_step)


# 这个文件采用弹簧外力施加，并且允许记录引力中心和施加外力的索引，方便后续的可视化。
