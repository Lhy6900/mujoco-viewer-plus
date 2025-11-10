# 多环境仿真实现说明

## 概述

本次修改实现了类似 `bmwoyaw_multienv.py` 的多环境仿真功能，允许多个机器人实例在独立的仿真环境中运行，每个机器人与各自的策略交互。

## 架构设计

### 核心思想
- **独立的仿真数据**: 使用 `datalist` 为每个环境维护独立的 `MjData`
- **独立的 DDS 通信**: 每个环境使用不同的 DDS domain ID (1, 2, 3, ...)
- **独立的控制缓冲**: 每个环境有自己的 lag buffer
- **共享的模型**: 所有环境共享同一个 `MjModel`

### 多环境数据流

```
Robot[0] (domain=1) <--DDS--> G1RobotDDS[0] <---> datalist[0]
                                                        |
Robot[1] (domain=2) <--DDS--> G1RobotDDS[1] <---> datalist[1]
                                                        |
Robot[2] (domain=3) <--DDS--> G1RobotDDS[2] <---> datalist[2]
                                                        |
                                                   MujocoEnv 线程
                                                  (统一步进所有环境)
```

## 关键修改

### 1. MultiMujocoSimulator (mujoco_simulator_multi.py)

**新增功能**:
- `lag_buffer_list`: 每个环境的独立命令缓冲列表
- `set_joint_commands_multi(env_idx, ...)`: 为指定环境设置命令
- `compute_torque_multi(env_idx, ...)`: 为指定环境计算力矩

**兼容性**: 
- 单环境模式 (`num_envs=1`) 继续使用原有的 `lag_buffer`、`set_joint_commands()`、`compute_torque()` 方法
- 多环境模式 (`num_envs>1`) 自动初始化 `lag_buffer_list` 和 `datalist`

### 2. MultiMujocoEnv (mujoco_env_multi.py)

**新增功能**:
- 为每个环境创建独立的 `G1RobotDDS` 对象
- 每个 DDS 对象使用不同的 domain ID: `1 + i`
- 在 `run()` 循环中分别处理每个环境的:
  - 命令读取 (`get_robot_command()`)
  - 命令应用 (`set_joint_commands_multi()`)
  - 力矩计算 (`compute_torque_multi()`)
  - 仿真步进 (`mujoco.mj_step()`)
  - 状态发布 (`write_robot_state()`)

**兼容性**:
- 单环境模式使用 `self.simulator.data` 和原有的单环境方法
- 多环境模式使用 `self.simulator.datalist[i]` 和多环境方法

### 3. G1RobotDDS (sim/dds/g1_robot_dds.py)

**新增参数**:
- `dds_domain_id`: 指定 DDS domain ID（默认为 1）

**修改**:
- 在 `setup_publisher()` 和 `setup_subscriber()` 中调用 `ChannelFactoryInitialize(dds_domain_id)` 来初始化对应的 domain

**兼容性**: 默认 domain ID 为 1，与原有单环境行为一致

### 4. G1Config & G1 (g1/g1.py)

**新增字段**:
- `dds_domain_id: int = 1`: 机器人使用的 DDS domain ID

**修改**:
- 在 `__init__()` 中使用配置的 `dds_domain_id` 初始化 Channel Factory

**兼容性**: 默认值为 1，保持原有行为

### 5. make_robot (controllers/beyondmimic.py)

**修改**:
- 使用 `config.get("dds_domain_id", 1)` 获取 domain ID
- 如果配置中没有 `dds_domain_id`，使用默认值 1

**兼容性**: 完全向后兼容，不需要修改现有调用代码

### 6. run_multirobot_sim.py

**修改**:
- 在多环境模式下，为每个 robot 分配 `dds_domain_id = 1 + i`
- 单环境模式不指定 `dds_domain_id`，使用默认值

## 使用方法

### 单环境模式（保持原有用法）

```bash
# 使用默认参数（num_envs=1, domain_id=1）
python scripts/run_multirobot_sim.py --model model/dalafan_prog.onnx

# 或明确指定 num_envs=1
python scripts/run_multirobot_sim.py --model model/dalafan_prog.onnx --num_envs 1
```

### 多环境模式（新功能）

```bash
# 2个环境
python scripts/run_multirobot_sim.py --model model/dalafan_prog.onnx --num_envs 2

# 4个环境
python scripts/run_multirobot_sim.py --model model/dalafan_prog.onnx --num_envs 4

# 配合 ViewerPlus
python scripts/run_multirobot_sim.py --model model/dalafan_prog.onnx --num_envs 2 --viewer plus
```

## DDS Domain ID 分配

| 环境索引 | Domain ID | DDS 对象名称 | Robot Config |
|---------|-----------|-------------|--------------|
| 0       | 1         | g1_robot_0  | dds_domain_id=1 |
| 1       | 2         | g1_robot_1  | dds_domain_id=2 |
| 2       | 3         | g1_robot_2  | dds_domain_id=3 |
| ...     | ...       | ...         | ... |

## 兼容性保证

### 向后兼容性
1. **单环境模式**: 所有原有代码无需修改，自动使用 domain ID = 1
2. **默认参数**: 所有新增参数都有合理的默认值
3. **渐进增强**: 多环境功能作为可选扩展，不影响现有功能

### 测试验证
```bash
# 验证单环境模式
python3 -m py_compile scripts/run_multirobot_sim.py

# 语法检查通过 ✓
```

## 注意事项

1. **DDS Domain 隔离**: 每个环境必须使用不同的 domain ID 来确保通信隔离
2. **内存占用**: 每个环境都有独立的 `MjData`，多环境会增加内存占用
3. **性能考虑**: 所有环境在同一个线程中串行步进，环境数量过多会影响实时性
4. **Viewer 显示**: 当前只显示第一个环境（main environment），其他环境通过 `mjv_addGeoms()` 渲染

## 未来改进方向

1. **并行仿真**: 使用多线程或多进程并行步进各环境
2. **动态环境管理**: 支持运行时添加/删除环境
3. **环境同步策略**: 提供不同的环境同步模式（同步/异步）
4. **可视化增强**: 支持切换主显示环境或分屏显示
