# Deploy Robot - 多环境机器人仿真框架 🤖

> 一个基于 MuJoCo 的高性能多环境机器人仿真框架，支持并行仿真、DDS 通信、可视化渲染等功能

## 目录

- [快速开始](#快速开始)
- [命令行参数详解](#命令行参数详解)
- [项目结构](#项目结构)
- [通信架构](#通信架构)
- [可视化系统](#可视化系统)
- [使用示例](#使用示例)
- [常见问题](#常见问题)

---

## 快速开始 🚀

### 最简单的运行方式

```bash
# 单环境仿真（默认配置）
python scripts/run_multirobot_sim.py

# 多环境仿真（2个并行环境）
python scripts/run_multirobot_sim.py --num_envs=2

# 使用自定义模型
python scripts/run_multirobot_sim.py --model=model/your_model.onnx --num_envs=4
```

### 运行效果

启动后你会看到：
- 🎮 MuJoCo 可视化窗口（可以用鼠标拖拽机器人施加外力）
- 👻 半透明的 Ghost 机器人（显示参考轨迹）
- 📊 实时奖励曲线图（显示跟踪误差等指标）
- 🤹 多个环境同时运行（如果 `num_envs > 1`）

---

## 命令行参数详解 📋

### `run_multirobot_sim.py` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `model/dalafan_prog.onnx` | ONNX 模型文件路径<br>支持任何兼容的 ONNX 策略模型 |
| `--num_envs` | int | `2` | 并行仿真环境数量<br>• `1`: 单环境模式<br>• `2+`: 多环境并行仿真 |
| `--viewer` | str | `plus` | 可视化类型<br>• `native`: MuJoCo 原生 viewer<br>• `plus`: 增强版 viewer（推荐）|

### 使用示例

```bash
# 基础用法
python scripts/run_multirobot_sim.py

# 多环境高性能仿真（4个环境并行）
python scripts/run_multirobot_sim.py --num_envs=4

# 使用原生 viewer（性能更好但功能较少）
python scripts/run_multirobot_sim.py --viewer=native

# 完整配置
python scripts/run_multirobot_sim.py \
    --model=model/my_policy.onnx \
    --num_envs=8 \
    --viewer=plus
```

### 环境变量

```bash
# 设置日志级别
export LOGLEVEL=DEBUG    # 详细调试信息
export LOGLEVEL=INFO     # 标准信息（默认）
export LOGLEVEL=WARNING  # 仅警告和错误

# 然后运行
python scripts/run_multirobot_sim.py
```

---

## 项目结构 📁

### 顶层目录

```
deploy_robot-main-v1.0/
├── scripts/              # 🎬 运行脚本
│   ├── run_multirobot_sim.py   # 多环境仿真（主要入口）
│   ├── run_robot_sim.py        # 单机器人仿真
│   └── run_robot.py            # 真机运行
├── deploy_robot/         # 📦 核心代码包
├── model/               # 🧠 ONNX 模型文件
├── logs/                # 📝 运行日志和数据
├── docs/                # 📚 文档
└── unitree_sdk2_python/ # 🔧 Unitree SDK
```

### `deploy_robot/` 核心模块

```
deploy_robot/
├── controllers/          # 🎮 控制器
│   └── beyondmimic.py   # BeyondMimic 策略控制器
├── policies/            # 🧠 策略模型
│   └── onnx_policy.py   # ONNX 策略推理
├── g1/                  # 🤖 G1 机器人接口
│   └── g1.py           # G1 硬件通信和控制
├── sim/                 # 🎮 仿真系统（重点！）
│   ├── mujoco_simulator.py       # 单环境仿真器
│   ├── mujoco_multisimulator.py  # 多环境仿真器
│   ├── mujoco_env.py            # 环境封装（单环境）
│   ├── mujoco_multienv.py       # 环境封装（多环境）
│   ├── dds/                     # DDS 通信模块
│   ├── viewer_plus/             # 增强可视化
│   └── visualization/           # 可视化工具库
├── utils/               # 🛠️ 工具函数
└── assets/              # 📦 资源文件（模型、场景等）
```

---

## Sim 模块详解 🎮

### Simulator vs Env：啥区别？

#### 🎯 **Simulator**（仿真器）
> 负责物理引擎、状态更新、数据通信

**核心文件**：
- `mujoco_simulator.py` - 单环境物理仿真
- `mujoco_multisimulator.py` - 多环境并行仿真

**主要功能**：
```python
# Simulator 做什么？
simulator.step()           # 执行物理步进
simulator.set_dof_targets()  # 设置关节目标
simulator.get_observations()  # 获取观测数据
simulator.add_ghost()      # 添加 ghost 可视化
```

**特点**：
- ✅ 直接操作 MuJoCo 物理引擎
- ✅ 管理 DDS 通信（与真机接口兼容）
- ✅ 处理多环境的状态同步
- ✅ 负责 viewer 渲染

#### 🎁 **Env**（环境）
> 包装 Simulator，提供更高层的接口（类似 Gym）

**核心文件**：
- `mujoco_env.py` - 单环境封装
- `mujoco_multienv.py` - 多环境封装

**主要功能**：
```python
# Env 做什么？
env.start()    # 启动仿真线程
env.pause()    # 暂停物理仿真
env.resume()   # 恢复仿真
env.stop()     # 停止并清理
```

**特点**：
- ✅ 线程管理（独立线程运行仿真）
- ✅ 生命周期管理（start/stop）
- ✅ 更简洁的 API
- ✅ 适合集成到训练/测试流程

**类比理解**：
```
Simulator = 汽车引擎（底层机械）
Env = 汽车整体（包含引擎 + 驾驶接口）

你开车只需要：
- env.start()     # 启动车辆
- env.resume()    # 踩油门
- env.pause()     # 刹车
- env.stop()      # 熄火

而不需要直接操作引擎的活塞、曲轴...
```

---

## 通信架构 📡

### 核心概念：为什么需要 DDS？

在真机上运行时，策略代码和机器人硬件是**分离的进程**：
- 🖥️ 策略进程：运行神经网络推理
- 🤖 机器人进程：控制电机、读取传感器

**DDS（Data Distribution Service）** 让它们能高效通信！

### 仿真模式：让仿真环境"假装"是真机

```
┌─────────────────────────────────────────────┐
│           仿真环境（模拟真机）                 │
│  ┌──────────────┐      ┌──────────────┐    │
│  │ G1RobotDDS   │ DDS  │  G1 Robot    │    │
│  │ (模拟硬件)    │◄────►│  (策略接口)   │    │
│  └──────────────┘      └──────────────┘    │
│         ↕                                   │
│   Shared Memory                             │
│         ↕                                   │
│  ┌──────────────┐                          │
│  │MuJoCo Simulator│                         │
│  │  (物理引擎)    │                          │
│  └──────────────┘                          │
└─────────────────────────────────────────────┘
```

### 共享内存通信

**为什么用共享内存？** 🚀
- ⚡ **极快**：零拷贝，纳秒级延迟
- 💪 **高效**：适合高频数据（1kHz 控制循环）
- 🔄 **双向**：策略 ↔ 仿真器

**数据流**：

```python
# 策略 → 仿真器（控制命令）
策略计算关节目标 → 写入共享内存 → DDS 发布
    → G1RobotDDS 接收 → 应用到 MuJoCo

# 仿真器 → 策略（状态反馈）
MuJoCo 状态 → G1RobotDDS 读取 → DDS 发布
    → G1 Robot 接收 → 更新策略输入

数据流程图
┌─────────────────────────────────────────────────────────────────────┐
│                    MuJoCo 仿真环境 (MultiMujocoEnv)                  │
│                                                                     │
│  MuJoCo Simulator                                                   │
│  计算物理、更新关节状态                                                │
│         │                                                           │
│         ↓                                                           │
│  ┌─────────────────────┐                                           │
│  │ write_robot_state() │ ← 每个时间步调用                            │
│  └──────────┬──────────┘                                           │
│             │ 写入                                                  │
│             ↓                                                       │
│  ┌──────────────────────────────────────┐                          │
│  │  input_shm (共享内存)                 │                          │
│  │  名称: isaac_robot_state_g1_robot_0  │                          │
│  │  内容: {                              │                          │
│  │    "joint_positions": [...],         │                          │
│  │    "joint_velocities": [...],        │                          │
│  │    "joint_torques": [...],           │                          │
│  │    "imu_data": [...]                 │                          │
│  │  }                                   │                          │
│  └──────────┬───────────────────────────┘                          │
│             │ 读取                                                  │
│             ↓                                                       │
│  ┌──────────────────────┐                                          │
│  │  dds_publisher()     │ ← 定时循环调用（100Hz）                    │
│  │  读取 input_shm      │                                           │
│  │  转换为 LowState_    │                                           │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│             ↓ DDS 发布                                              │
├─────────────┼───────────────────────────────────────────────────────┤
│   DDS Topic: "rt/lowstate" (domain_id=1)                           │
├─────────────┼───────────────────────────────────────────────────────┤
│             ↓ DDS 订阅                                              │
│  ┌──────────────────────┐                                          │
│  │  G1 Robot (控制端)    │                                          │
│  │  lowstate_subscriber │                                          │
│  │  update_state()      │ ← robot.update_state() 读取              │
│  └──────────────────────┘                                          │
│             ↓                                                       │
│  ┌──────────────────────┐                                          │
│  │  run_policy()        │                                          │
│  │  计算控制命令         │                                           │
│  │  robot.control()     │                                          │
│  └──────────┬───────────┘                                          │
│             │ DDS 发布                                              │
│             ↓                                                       │
├─────────────┼───────────────────────────────────────────────────────┤
│   DDS Topic: "rt/lowcmd" (domain_id=1)                             │
├─────────────┼───────────────────────────────────────────────────────┤
│             ↓ DDS 订阅                                              │
│  ┌──────────────────────┐                                          │
│  │  dds_subscriber()    │ ← 收到 LowCmd_ 消息                       │
│  │  解析命令并写入       │                                           │
│  └──────────┬───────────┘                                          │
│             │ 写入                                                  │
│             ↓                                                       │
│  ┌──────────────────────────────────────┐                          │
│  │  output_shm (共享内存)                │                          │
│  │  名称: dds_robot_cmd_g1_robot_0      │                          │
│  │  内容: {                              │                          │
│  │    "mode_pr": 1,                     │                          │
│  │    "mode_machine": 5,                │                          │
│  │    "motor_cmd": {                    │                          │
│  │      "positions": [...],             │                          │
│  │      "kp": [...], "kd": [...]        │                          │
│  │    }                                 │                          │
│  │  }                                   │                          │
│  └──────────┬───────────────────────────┘                          │
│             │ 读取                                                  │
│             ↓                                                       │
│  ┌──────────────────────┐                                          │
│  │ get_robot_command()  │ ← env 循环中调用                          │
│  │ 读取 output_shm      │                                           │
│  └──────────┬───────────┘                                          │
│             │                                                       │
│             ↓                                                       │
│  MuJoCo Simulator                                                   │
│  应用控制命令到关节                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### DDS 模块结构

```
deploy_robot/sim/dds/
├── dds_base.py          # 基类：定义 DDS 对象接口
├── dds_master.py        # 管理器：统一管理所有 DDS 节点
├── g1_robot_dds.py      # G1 机器人的 DDS 实现
├── sharedmemorymanager.py  # 共享内存封装
└── dds_create.py        # DDS 节点工厂
```

**关键特性**：
- 🔐 **Domain ID 隔离**：多环境通过不同 domain ID 隔离通信
- 🔄 **自动重连**：网络中断后自动恢复
- 📊 **频率控制**：可配置发布频率（避免浪费 CPU）

---

## 可视化系统 🎨

### ViewerPlus：不只是看看！

#### 功能清单

| 功能 | 快捷键 | 说明 |
|------|--------|------|
| 👻 Ghost 渲染 | `g` 或 `Ctrl+G` | 显示参考轨迹（半透明绿色） |
| 📊 奖励曲线 | `r` 或 `Ctrl+R` | 实时显示跟踪误差等指标 |
| 🤹 多环境显示 | `m` | 同时显示所有环境的机器人 |
| ⬅️➡️ 切换环境 | `←` `→` | 切换主环境（多环境模式）|
| 🎯 施加外力 | 鼠标拖拽 | 交互式测试（按住 Ctrl 拖拽）|

#### Ghost 是什么？

Ghost 是一个**半透明的参考机器人**，显示策略认为机器人应该在的位置：

```
实际机器人（不透明）→ 你看到的真实状态
Ghost（半透明）    → 策略的"理想轨迹"

如果重合度高 = 跟踪很好 ✅
如果偏差大   = 需要调整策略 ⚠️
```

### 可视化工具库（新架构！）

位置：`deploy_robot/sim/visualization/`

#### 一行代码添加 Ghost

**旧方式**（30+ 行代码）：
```python
# 提取策略输出
policy_joint_pos = np.asarray(policy.output_tensors["joint_pos"])
policy_joint_pos = policy_joint_pos.squeeze(0)
# ... 20+ 行代码 ...
simulator.add_ghost(ghost_qpos)
```

**新方式**（1 行搞定）：
```python
from deploy_robot.sim.visualization import update_ghost_from_policy
update_ghost_from_policy(simulator, policy, robot)
```

#### 计算跟踪奖励

```python
from deploy_robot.sim.visualization import compute_tracking_rewards

# 自动计算多个跟踪指标
rewards = compute_tracking_rewards(policy, robot)
# 返回：
# {
#     "smoothness": -0.05,           # 运动平滑度
#     "pos_tracking_global": -0.02,   # 全局位置误差
#     "pos_tracking_local": -0.01,    # 局部位置误差
#     "quat_tracking_global": 0.0,    # 姿态误差
#     "quat_tracking_local": 0.0,     # 局部姿态误差
# }

# 显示在图表上
simulator.update_reward_plot(rewards)
```

#### 模块化设计

```python
# 所有可视化元素都是独立的、可切换的
from deploy_robot.sim.visualization import (
    GhostRenderer,       # Ghost 渲染器
    ForceRenderer,       # 力可视化（TODO）
    ContactRenderer,     # 接触点可视化（TODO）
)

# 轻松添加自定义可视化
class MyRenderer(BaseVisualElement):
    def update(self, data):
        # 更新要显示的数据
        pass
    
    def render(self, viewer_scene, viewer_option):
        # 渲染到 MuJoCo 场景
        pass
```

**详细文档**：
- 📖 使用指南：`deploy_robot/sim/visualization/README.md`
- 🚀 快速开始：`deploy_robot/sim/visualization/QUICKSTART.md`

---

## 使用示例 💡

### 示例 1：基础单环境仿真

```python
from deploy_robot.controllers.beyondmimic import make_policy, make_robot, run_policy
from deploy_robot.sim.mujoco_env import MujocoEnv
from deploy_robot.sim.config.g1 import G1MujocoConfig

# 1. 加载策略
policy = make_policy("model/your_model.onnx")

# 2. 创建机器人接口
robot = make_robot({
    "joint_names": policy.joint_names,
    "joint_stiffness": policy.joint_stiffness,
    "joint_damping": policy.joint_damping,
    "default_joint_pos": policy.default_joint_positions,
}, env='sim')

# 3. 启动仿真环境
env = MujocoEnv(G1MujocoConfig)
env.start()

# 4. 初始化机器人
robot.start_communication()
robot.set_default_posture()

# 5. 控制循环
time_step = 0
while True:
    run_policy(policy, robot, time_step, logger)
    time_step += 1
    time.sleep(0.02)  # 50Hz 控制
```

### 示例 2：多环境并行仿真

```python
from deploy_robot.sim.mujoco_multienv import MultiMujocoEnv

# 配置多环境
G1MujocoConfig.num_envs = 4

# 创建多个策略和机器人
policylist = [make_policy(model_path) for _ in range(4)]
robotlist = [make_robot({
    "joint_names": policy.joint_names,
    # ... 其他参数 ...
    "dds_domain_id": 1 + i,  # 每个环境不同的 domain ID
}, env='sim') for i, policy in enumerate(policylist)]

# 启动多环境
env = MultiMujocoEnv(G1MujocoConfig)
env.start()

# 并行控制所有环境
while True:
    for i in range(4):
        run_policy(policylist[i], robotlist[i], time_step, logger)
    time_step += 1
```

### 示例 3：添加自定义可视化

```python
from deploy_robot.sim.visualization import update_ghost_from_policy

# 在控制循环中
while True:
    run_policy(policy, robot, time_step, logger)
    
    # 自动添加 ghost 可视化
    update_ghost_from_policy(env.simulator, policy, robot)
    
    # 计算和显示奖励
    from deploy_robot.sim.visualization import compute_tracking_rewards
    rewards = compute_tracking_rewards(policy, robot)
    env.simulator.update_reward_plot(rewards)
```

### 示例 4：环境切换（多环境）

```python
# 运行时切换观察的环境
import keyboard

while True:
    # 检查键盘输入
    if keyboard.is_pressed('1'):
        env.simulator.switch_active_env(0)
    elif keyboard.is_pressed('2'):
        env.simulator.switch_active_env(1)
    # ... 或者使用 viewer 的箭头键自动切换
    
    # 控制循环
    run_policy(...)
```

---

## 常见问题 ❓

### Q1: 运行报错 `AttributeError: 'NoneType' object has no attribute 'CreateChannel'`

**原因**：DDS 初始化失败，通常是多次调用 `ChannelFactoryInitialize` 导致。

**解决**：
- ✅ 已在 v1.0 版本修复（使用单例模式）
- 确保使用最新代码

### Q2: 外力施加后机器人没反应？

**原因**：外力被过早清除。

**解决**：
- ✅ 已修复（外力在整个 decimation 周期内有效）
- 在 viewer 中按住 Ctrl 拖拽机器人即可施加外力

### Q3: Ghost 不显示？

**解决方案**：
1. 按 `g` 键切换 ghost 显示
2. 确保使用了 `--viewer=plus` 参数
3. 检查策略是否输出了 `body_pos_w` 和 `body_quat_w`

### Q4: 多环境性能不佳？

**优化建议**：
```bash
# 1. 减少环境数量
--num_envs=2  # 而不是 8

# 2. 使用 native viewer（更快）
--viewer=native

# 3. 关闭可视化（headless 模式）
# 修改配置：G1MujocoConfig.headless = True
```

### Q5: 想在 Isaac Sim 中使用？

这个框架主要针对 MuJoCo，但架构是通用的：
- `sim/` 目录可以添加 `isaac_simulator.py`
- DDS 通信层完全兼容
- 只需实现相同的 Simulator 接口即可

### Q6: 如何记录实验数据？

```python
from deploy_robot.utils.writer import SimpleWriter

logger = SimpleWriter()

# 在控制循环中记录
logger.log('joint_pos', robot.joint_pos)
logger.log('joint_vel', robot.joint_vel)
logger.log('reward', reward_value)

# 保存到文件
import numpy as np
nowtime = time.strftime("%Y-%m-%d-%H-%M-%S")
filename = f"logs/experiment_{nowtime}.npz"
np.savez(filename, **logger)
```

### Q7: 支持其他机器人吗？

当前主要支持 **Unitree G1**，但可以扩展：

1. 在 `deploy_robot/` 下创建新机器人文件夹（如 `go2/`）
2. 实现相同的接口（参考 `g1/g1.py`）
3. 创建对应的 DDS 节点（参考 `sim/dds/g1_robot_dds.py`）

---

## 高级主题 🎓

### 性能调优

```python
# mujoco_simulator.py 中的配置
config = MultiMujocoSimulatorConfig(
    xml_path="path/to/scene.xml",
    decimation=4,        # 每个控制步运行 4 次物理步
                         # ↓ 减少 = 更快但精度低
                         # ↑ 增加 = 更慢但更精确
    
    dt=0.005,           # 物理步时间间隔
                        # 默认 5ms（200Hz 物理更新）
)
```

### 自定义奖励项

```python
# 1. 定义自己的奖励计算函数
def my_reward_function(policy, robot):
    # 例如：惩罚高速运动
    vel_penalty = -np.linalg.norm(robot.joint_vel)
    
    # 例如：奖励保持平衡
    balance_reward = 1.0 if abs(robot.base_rpy[0]) < 0.1 else 0.0
    
    return {
        "velocity_penalty": vel_penalty,
        "balance": balance_reward,
    }

# 2. 在控制循环中使用
rewards = my_reward_function(policy, robot)
env.simulator.update_reward_plot(rewards)
```

### 扩展 DDS 通信

```python
# 1. 创建自定义 DDS 对象
from deploy_robot.sim.dds.dds_base import DDSObject

class MyDDS(DDSObject):
    def dds_publisher(self):
        # 实现发布逻辑
        pass
    
    def dds_subscriber(self, msg, datatype):
        # 实现订阅逻辑
        pass

# 2. 注册到 DDSManager
from deploy_robot.sim.dds.dds_master import DDSManager
manager = DDSManager()
my_dds = MyDDS()
manager.register_object("my_node", my_dds)
```

---

## 贡献指南 🤝

欢迎贡献代码！

### 开发流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 代码风格

- 使用 Python 3.10+
- 遵循 PEP 8
- 添加类型注解
- 写清楚的文档字符串

---

## 致谢 🙏

- **MuJoCo**：强大的物理引擎
- **Unitree Robotics**：提供 SDK 和硬件支持
- **mjlab**：部分可视化灵感来源

---

## 许可证 📄

Apache License 2.0

---

## 联系方式 📧

- 问题反馈：提交 GitHub Issue
- 功能建议：欢迎 PR

---

**🎉 Happy Simulating! 让机器人动起来！**

