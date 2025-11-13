# MuJoCo 多环境仿真框架

基于 MuJoCo 的模块化多环境仿真框架，支持模仿学习、运动控制等多种策略类型。

## 📁 目录结构

```
deploy_mujoco/
├── bmwoyaw_multienv.py          # 【主程序】多环境仿真入口（~80 行精简协调器）
├── bmconfig.py                   # 【配置文件】所有参数配置（用户主要修改）
│
├── core/                         # 【核心模块】仿真核心功能
│   ├── coordinator.py           # 仿真协调器（统一管理所有组件，~550 行）
│   ├── environment.py           # 环境管理器（多环境管理、状态同步）
│   ├── policy.py                # 策略加载和推理（ONNX 策略执行）
│   ├── observation.py           # 观测构造器（策略模式，支持多种观测类型）
│   └── controller.py            # PD控制器（关节空间控制）
│
├── visualization/                # 【可视化模块】渲染和奖励
│   ├── ghost.py                 # Ghost 渲染器（参考运动可视化）
│   ├── force_applicator.py      # 外力施加器（恒定力/弹簧力）
│   ├── force_visualizer.py      # 外力可视化器（紫色箭头渲染）
│   ├── reward_calculator.py     # 奖励计算器（模块化奖励项）
│   ├── reward_visualizer.py     # 奖励图表（实时曲线绘制）
│   └── README.md                # 可视化模块说明（MuJoCo 渲染方法）
│
├── utils/                        # 【工具模块】通用工具
│   ├── math_utils.py            # 数学函数（四元数等）
│   ├── logger.py                # 日志记录
│   └── data_loader.py           # 数据加载
│
├── docs/                         # 【文档】
│   ├── README.md                # 本文档
│   ├── force_system.md          # 外力系统文档
│   ├── ghost_rendering.md       # Ghost 渲染文档
│   └── reward_system.md         # 奖励系统文档
│
├── legacy/                       # 【旧代码】历史版本（已废弃）
├── module/                       # 【旧模块】actor_critic 等（已废弃）
├── tests/                        # 【测试】测试脚本
└── configs/                      # 【配置文件】机器人配置（YAML）
```

## 🚀 快速开始

### 1. 运行仿真

```bash
cd /home/ubuntu/deploy_mini/deploy_mujoco
python bmwoyaw_multienv.py
```

### 2. 修改配置

编辑 `bmconfig.py` 文件：

```python
# 切换策略类型
POLICY_CONFIG = {
    'type': 'imitation',      # 改为 'locomotion' 或 'custom'
    'enable_ghost': True,     # 非模仿学习改为 False
}

# 调整环境数量
NUM_ENVS = 8                  # 改为任意数量

# 修改奖励权重
REWARD_CONFIG = {
    'weights': {
        'pos_tracking_global': 0.8,  # 调整权重
        'quat_tracking_global': 0.2,
    }
}

# 调整外力参数
FORCE_CONFIG = {
    'mode': {
        'select': [0, 2, 3],  # 指定施加外力的环境
    },
    'spring': {
        'k': 100.0,           # 增加弹簧系数
    },
}
```

## 🎯 核心功能

### 1. 多环境仿真

- **自动网格布局**：环境自动排列成网格
- **独立仿真**：每个环境独立步进
- **环境切换**：按 `↑` / `↓` 键切换主视角环境

### 2. Ghost 渲染

- **参考轨迹显示**：半透明绿色 Ghost 显示参考运动
- **智能启用**：仅当有参考运动数据时启用
- **灵活控制**：
  - `Ctrl+G`: 切换主环境 Ghost
  - `Ctrl+M`: 切换其他环境 Ghost

### 3. 外力系统

- **三维度控制**：
  - **stop**: `'fixtime'` (固定时间) | `'keeping'` (持续切换)
  - **style**: `'constant'` (恒定力) | `'spring'` (弹簧力)
  - **select**: `None` | `0` | `1` | `[0,2,3]` (选择环境)
  
- **交互控制**：
  - `Ctrl+F`: 触发/停止外力
  - 紫色箭头可视化

- **自动缩放**：弹簧模式下，箭头终点自动指向引力中心

详细文档：[force_system.md](force_system.md)

### 4. 奖励系统

- **模块化计算**：支持多种奖励项
- **实时可视化**：右侧显示奖励曲线
- **灵活配置**：可启用/禁用任意奖励项

详细文档：[reward_system.md](reward_system.md)

## 🔧 用户自定义指南

### 修改观测（Observation）

编辑 `core/observation.py`：

```python
def _build_custom(self, data):
    """【用户自定义区域】自定义观测"""
    obs = np.zeros(self.obs_dim, dtype=np.float32)
    
    # 添加你的观测逻辑
    obs[0:3] = data.qpos[0:3]      # 位置
    obs[3:6] = data.qvel[0:3]      # 速度
    # ... 更多观测
    
    return obs
```

### 修改奖励（Reward）

编辑 `visualization/reward_calculator.py`：

```python
def _compute_custom_reward(self, data, motion_data):
    """【用户自定义区域】自定义奖励"""
    # 添加你的奖励计算逻辑
    energy = np.sum(np.abs(data.ctrl))
    return -0.01 * energy  # 能量惩罚
```

然后在 `bmconfig.py` 中启用：

```python
REWARD_CONFIG = {
    'enabled': [
        'custom_reward',  # 添加到启用列表
    ],
    'weights': {
        'custom_reward': 0.1,  # 设置权重
    },
}
```

### 修改控制策略

编辑 `core/policy.py` 或创建新的策略类。

## 📊 模块说明

### 架构设计

框架采用 **模块化 + 协调器模式**：

```
bmwoyaw_multienv.py (入口，~80 行)
    ↓ 创建并启动
SimulationCoordinator (协调器，~550 行)
    ↓ 管理所有组件
    ├── EnvironmentManager (环境管理)
    ├── PolicyRunner (策略推理)
    ├── ObservationBuilder (观测构造，策略模式)
    ├── PDController (PD 控制)
    ├── GhostRenderer (Ghost 渲染)
    ├── ForceApplicator (外力施加)
    ├── ForceVisualizer (外力可视化)
    ├── RewardCalculator (奖励计算)
    └── RewardPlotter (奖励可视化)
```

**设计理念**：
- **单一职责**：每个模块负责一个独立功能
- **策略模式**：观测构造支持多种类型 (`imitation`, `locomotion`, `custom`)
- **依赖注入**：协调器通过配置字典初始化所有组件
- **无限循环**：仿真持续运行直到用户关闭窗口

### core.SimulationCoordinator

**仿真协调器** - 统一管理所有组件（~550 行）：

- `__init__(config)`: 从配置初始化所有模块
- `_build_observation(env_idx)`: 通过 ObservationBuilder 构造观测
- `_compute_control(env_idx)`: 策略推理 + PD 控制
- `_update_ghost(env_idx)`: 更新 Ghost 姿态
- `_compute_rewards(env_idx)`: 计算奖励并可视化
- `run()`: 主仿真循环（无限循环，直到窗口关闭）

### core.EnvironmentManager

**环境管理器** - 管理多个 MuJoCo 环境：

- `reset_env(env_idx)`: 重置指定环境
- `step_all(ctrl_list)`: 所有环境步进
- `sync_to_main_data()`: 同步主环境数据到 viewer
- `initialize_positions(joint_pos)`: 初始化所有环境关节位置
- `get_state(env_idx)`: 获取环境状态

### core.PolicyRunner

**策略执行器** - 加载和运行 ONNX 策略：

- `compute_action(obs, timestep, env_idx)`: 计算单环境动作
- `compute_actions_batch(obs_list, timestep)`: 批量推理（多环境）
- `compute_ghost_outputs(obs, timestep, env_idx)`: Ghost 渲染输出
- 支持 `'imitation'`, `'locomotion'`, `'custom'` 三种策略类型

### core.ObservationBuilder

**观测构造器** - 策略模式构造观测向量：

- `build(data, motion_data, action_buffer, timestep)`: 构造观测（策略模式路由）
- `_build_imitation(...)`: 模仿学习观测（58 + 3 + 29 + 29 + 29 = 148 维）
- `_build_locomotion(...)`: 运动控制观测
- `_build_custom(...)`: 自定义观测
- `set_joint_mapping(...)`: 设置关节映射（策略序列 → XML 序列）

**策略模式**：通过 `build()` 统一接口，根据 `obs_type` 自动路由到对应的私有方法。

### visualization.RewardCalculator

**奖励计算器** - 模块化奖励计算：

- `compute(data, motion_data)`: 计算所有启用的奖励项
- 返回字典：`{'total': float, 'pos_tracking_global': float, ...}`
- 支持自定义奖励项（编辑 `_compute_custom_reward()`）

### visualization.GhostRenderer

**Ghost 渲染器** - 参考运动可视化：

- `set_ghost_qpos(qpos)`: 设置 Ghost 姿态
- `render_ghost(viewer_scene)`: 渲染半透明绿色 Ghost
- 通过 MuJoCo 的 `mjv_initGeom()` 和 `mjv_addGeoms()` 实现

### visualization.ForceApplicator

**外力施加器** - 施加恒定力或弹簧力：

- `trigger(data_list, duration, force_magnitude, force_direction)`: 触发外力
- `update(data_list)`: 更新外力状态（每帧调用）
- `stop()`: 停止外力
- 支持 `'fixtime'`（固定时间）/ `'keeping'`（持续切换）
- 支持 `'constant'`（恒定力）/ `'spring'`（弹簧力）

## 🎮 快捷键

| 快捷键 | 功能 |
|--------|------|
| `↑` / `↓` | 切换主视角环境（上/下箭头键） |
| `M` | 显示/隐藏其他环境 |
| `Ctrl+G` | 切换主环境 Ghost |
| `Ctrl+M` | 切换其他环境 Ghost |
| `Ctrl+F` | 触发/停止外力 |
| `Ctrl+R` | 显示/隐藏奖励曲线窗口 |
| `Space` | 暂停/继续仿真 |
| `Esc` | 退出 |

## 📖 进阶文档

- [外力系统详细说明](force_system.md)
- [Ghost 渲染详细说明](ghost_rendering.md)
- [奖励系统详细说明](reward_system.md)

## 🐛 调试

### 启用详细日志

```python
# 在 bmwoyaw_multienv.py 开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 检查环境状态

```python
# 在主循环中添加
state = env_manager.get_state(0)
print(f"Env 0 qpos: {state['qpos']}")
```

### Ghost 不显示？

1. 检查是否成功加载参考运动数据
2. 确认 `POLICY_CONFIG['enable_ghost'] = True`
3. 按 `Ctrl+G` 确保 Ghost 已启用

### 外力不生效？

1. 检查 `FORCE_CONFIG['mode']['select']` 是否正确
2. 确认按下了 `Ctrl+F`
3. 查看控制台输出

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可

MIT License
