# MuJoCo 多环境仿真框架

基于 MuJoCo 的模块化多环境仿真框架，支持模仿学习、运动控制等多种策略类型。

## 📁 目录结构

```
deploy_mujoco/
├── bmwoyaw_multienv.py          # 【主程序】多环境仿真入口
├── bmconfig.py                   # 【配置文件】所有参数配置（用户主要修改）
│
├── core/                         # 【核心模块】仿真核心功能
│   ├── environment.py           # 环境管理器
│   ├── policy.py                # 策略加载和推理
│   ├── observation.py           # 观测构造器
│   └── controller.py            # PD控制器
│
├── visualization/                # 【可视化模块】渲染和奖励
│   ├── ghost.py                 # Ghost 渲染器
│   ├── force_applicator.py      # 外力施加器
│   ├── force_visualizer.py      # 外力可视化器
│   ├── reward_calculator.py     # 奖励计算器
│   └── reward_visualizer.py     # 奖励图表
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
- **环境切换**：按 `Tab` 键切换主视角环境

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

### core.EnvironmentManager

管理多个 MuJoCo 环境：

- `reset_env(env_idx)`: 重置指定环境
- `step_all(ctrl_list)`: 所有环境步进
- `get_state(env_idx)`: 获取环境状态

### core.PolicyRunner

加载和运行 ONNX 策略：

- `compute_action(obs, timestep)`: 计算动作
- `compute_ghost_outputs(obs, timestep)`: Ghost 渲染输出

### core.ObservationBuilder

构造观测向量：

- `build(data, motion_data, action_buffer)`: 构造观测
- 支持 `'imitation'`, `'locomotion'`, `'custom'` 三种类型

### visualization.RewardCalculator

计算奖励：

- `compute(data, motion_data)`: 计算所有奖励
- 返回字典：`{'total': float, 'pos_tracking': float, ...}`

### visualization.GhostRenderer

渲染 Ghost：

- `set_ghost_qpos(qpos)`: 设置 Ghost 姿态
- `render_ghost(viewer_scene)`: 渲染到场景

### visualization.ForceApplicator

施加外力：

- `trigger(data_list)`: 触发外力
- `update(data_list)`: 更新外力
- `stop()`: 停止外力

## 🎮 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Tab` | 切换主视角环境 |
| `M` | 显示/隐藏其他环境 |
| `Ctrl+G` | 切换主环境 Ghost |
| `Ctrl+M` | 切换其他环境 Ghost |
| `Ctrl+F` | 触发/停止外力 |
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
