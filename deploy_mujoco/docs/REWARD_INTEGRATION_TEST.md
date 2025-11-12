# 奖励计算集成测试指南

## ✅ 已完成的修改

### 1. 导入模块
在 `bmwoyaw_multienv.py` 开头添加了：
```python
from reward_calculator import compute_rewards
```

### 2. 初始化变量
在主循环前添加了：
```python
last_qvel_for_reward = None  # 保存上一帧的关节速度
```

### 3. 计算奖励
在 `if counter % control_decimation == 0:` 块中的策略计算之后，添加了：
```python
# 计算奖励
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

# 保存速度供下一帧使用
current_joint_vel_seq = np.array([...])
last_qvel_for_reward = current_joint_vel_seq.copy()

# 每50步打印一次
if timestep % 50 == 0:
    print(f"\n[Rewards @ step {timestep}]")
    for name, value in rewards.items():
        print(f"  {name:25s}: {value:+.6f}")
```

---

## 🧪 测试方法

### 方法 1：使用测试脚本（推荐）
```bash
cd /home/ubuntu/deploy_mini/deploy_mujoco
./test_rewards.sh
```

### 方法 2：直接运行
```bash
cd /home/ubuntu/deploy_mini/deploy_mujoco
source ~/anaconda3/bin/activate mujoco
python bmwoyaw_multienv.py --num_envs=2
```

---

## 📊 预期输出

### 启动信息
```
使用并行环境数量 num_envs=2
环境布局：2行 x 1列
当前使用的设备: cpu
[RewardCalculator] 初始化完成
```

### 每50步的奖励打印
```
[Rewards @ step 0]
  smoothness               : +0.000000  # 首帧无速度历史，为0
  pos_tracking_global      : -0.023456  # 位置误差（L2范数）
  pos_tracking_local       : -0.003421  # 位置误差（平均）
  quat_tracking_global     : -0.045123  # 速度误差
  quat_tracking_local      : -0.002891  # 参考轨迹误差

[Rewards @ step 50]
  smoothness               : -0.012345  # 速度变化
  pos_tracking_global      : -0.018234
  pos_tracking_local       : -0.002567
  quat_tracking_global     : -0.034567
  quat_tracking_local      : -0.001234

[Rewards @ step 100]
  smoothness               : -0.008765
  pos_tracking_global      : -0.015432
  pos_tracking_local       : -0.001987
  quat_tracking_global     : -0.028765
  quat_tracking_local      : -0.000876
```

---

## 🔍 如何解读奖励值

### 数值范围
- 所有奖励值都是**负数**（惩罚）
- 越接近 **0** 表示表现越好
- 典型范围：
  - `smoothness`: -0.5 到 0 （首帧为0）
  - `pos_tracking_global`: -0.5 到 -0.001
  - `pos_tracking_local`: -0.1 到 -0.0001
  - `quat_tracking_global`: -0.2 到 -0.001
  - `quat_tracking_local`: -0.05 到 -0.0001

### 好的表现
```
smoothness               : -0.001234  # 运动平滑
pos_tracking_global      : -0.005678  # 跟踪准确
pos_tracking_local       : -0.000891  # 局部误差小
```

### 不好的表现
```
smoothness               : -0.456789  # 抖动严重
pos_tracking_global      : -0.234567  # 跟踪误差大
pos_tracking_local       : -0.045678  # 局部偏差大
```

---

## 🐛 常见问题

### Q1: 没有看到奖励打印
**可能原因**：
- timestep 还没到 50 的倍数
- 程序启动时间太短

**解决方法**：
- 等待至少 1-2 秒（50步 @ 50Hz ≈ 1秒）
- 或修改打印频率：`if timestep % 10 == 0:`

### Q2: 奖励值全是 0.000000
**可能原因**：
- 计算函数出错（应该有警告信息）
- 数据未正确初始化

**解决方法**：
- 查看是否有 `[警告] 奖励计算失败` 的打印
- 检查控制台是否有其他错误信息

### Q3: smoothness 一直是 0
**原因**：这是正常的！
- 第一帧（timestep=0）时 `last_qvel_for_reward` 为 None
- 所以 smoothness 返回 0
- 从第二次打印（timestep=50）开始应该有正常值

### Q4: 奖励值异常大（如 -10.0 以下）
**可能原因**：
- 机器人姿态不稳定
- 参考轨迹和实际动作差异大

**这是正常的**：
- 初始几帧可能误差较大
- 随着仿真进行应该逐渐减小

---

## ✅ 验证成功的标志

如果看到以下现象，说明集成成功：

1. ✅ 程序正常启动，没有 import 错误
2. ✅ 每50步打印一次奖励值
3. ✅ 5个奖励项都有数值（不是全为0）
4. ✅ smoothness 从 step 50 开始有非零值
5. ✅ 奖励值范围合理（-1.0 到 0.0 之间为佳）

---

## 🎯 下一步

验证成功后，我们将进行：

**步骤 3**：添加 Ctrl+R 键盘控制
- 修改 `key_callback` 函数
- 添加 `show_reward` 全局变量
- 测试键盘切换功能

**步骤 4**：集成 RewardPlotter
- 导入 `RewardPlotter` 类
- 初始化绘图器
- 将 `rewards` 传递给 `reward_plotter.update()`

**步骤 5**：渲染奖励图表到 viewer
- 在渲染循环中添加图表显示
- 测试最终效果

---

## 📝 测试记录模板

测试时请记录：

```
测试时间：2025-11-11
环境数量：2
运行时长：约 10 秒

✅ 程序启动正常
✅ 看到奖励打印
✅ 数值范围：smoothness: -0.01, pos_global: -0.02, ...
⚠️ 问题：（如有问题在此记录）
```

按 Ctrl+C 退出测试即可。
