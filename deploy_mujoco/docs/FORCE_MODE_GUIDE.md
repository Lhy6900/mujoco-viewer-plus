# ForceApplicator 力模式使用指南

## 概述

`ForceApplicator` 支持灵活的外力施加模式，通过 `force_mode` 字典配置两个独立的维度：

1. **stop（停止模式）**：控制外力的开始和停止时机
2. **style（力的风格）**：控制外力的计算方式

## force_mode 配置

### 1. stop 模式（何时停止）

#### `'fixtime'` - 固定时间停止（默认）
- **行为**：施加固定时长的外力后自动停止
- **用途**：测试机器人在短时扰动下的响应
- **触发方式**：按 Ctrl+F 触发，5秒后自动停止

#### `'keeping'` - 持续施加
- **行为**：持续施加外力，直到再次按键停止
- **用途**：模拟持续的外部作用力（如风力、推力等）
- **触发方式**：
  - 第一次按 Ctrl+F：开始施加外力
  - 第二次按 Ctrl+F：停止施加外力

### 2. style 模式（如何施加力）

#### `'constant'` - 恒定力（默认）
- **力大小**：恒定值（默认 20N）
- **力方向**：固定方向（默认 Y 正方向）
- **适用场景**：
  - 模拟恒定推力
  - 测试机器人平衡能力
  - 简单的扰动测试

#### `'spring'` - 弹簧力
- **力大小**：`F = K × distance`
  - `K`：弹簧系数（默认 50）
  - `distance`：body 到引力中心的距离
- **力方向**：从 body 指向引力中心
- **引力中心计算**：
  ```
  spring_center = body_position + spring_center_bias
  ```
  - `spring_center_bias`：默认 `[0.2, 0.2, 0.0]`（单位：米）
- **引力中心生命周期**：
  - 按 Ctrl+F 时：计算并固定每个环境的引力中心
  - 施加过程中：引力中心保持不变
  - 停止后：清零
  - 再次施加：重新计算
- **适用场景**：
  - 模拟弹性约束
  - 测试机器人在拉扯下的恢复能力
  - 多环境下不同的扰动位置

## 四种组合模式

### 1. `{stop: 'fixtime', style: 'constant'}`
固定时间的恒定力（最简单）
```python
force_mode = {'stop': 'fixtime', 'style': 'constant'}
```
- 按 Ctrl+F → 施加 5 秒恒定力 → 自动停止

### 2. `{stop: 'fixtime', style: 'spring'}`
固定时间的弹簧力
```python
force_mode = {'stop': 'fixtime', 'style': 'spring'}
```
- 按 Ctrl+F → 记录引力中心 → 施加 5 秒弹簧力 → 自动停止并清零引力中心

### 3. `{stop: 'keeping', style: 'constant'}`
持续的恒定力
```python
force_mode = {'stop': 'keeping', 'style': 'constant'}
```
- 第一次按 Ctrl+F → 持续施加恒定力
- 第二次按 Ctrl+F → 停止

### 4. `{stop: 'keeping', style: 'spring'}`
持续的弹簧力（推荐用于测试）
```python
force_mode = {'stop': 'keeping', 'style': 'spring'}
```
- 第一次按 Ctrl+F → 记录引力中心 → 持续施加弹簧力
- 第二次按 Ctrl+F → 停止并清零引力中心

## 配置参数

### 在 bmwoyaw_multienv.py 中配置

```python
# 修改这部分代码来切换模式
force_mode = {
    'stop': 'keeping',   # 'fixtime' 或 'keeping'
    'style': 'spring'    # 'constant' 或 'spring'
}

force_applicator = ForceApplicator(
    m, 
    body_name=force_anchor_bodyname, 
    force_anchor_bodyidx=body_id,
    force_mode=force_mode
)
```

### 修改弹簧参数

如果需要自定义弹簧参数，在初始化后修改：

```python
force_applicator = ForceApplicator(...)

# 修改弹簧系数（默认 50）
force_applicator.spring_k = 100.0  # 更强的弹簧

# 修改引力中心偏置（默认 [0.2, 0.2, 0.0]）
force_applicator.spring_center_bias = np.array([0.5, 0.0, 0.0])  # X 方向 0.5m
```

### 修改恒定力参数

在 key_callback 中修改 trigger 参数：

```python
force_applicator.trigger(
    duration=10.0,  # 持续时间（fixtime 模式）
    force_magnitude=50.0,  # 力大小（constant 模式）
    force_direction=np.array([1.0, 0.0, 0.0]),  # 力方向（constant 模式）
    data_list=dlist
)
```

## 可视化

### 外力箭头颜色
- **紫色** CAPSULE + SPHERE：表示施加的外力
  - Constant 模式：所有环境相同方向和大小
  - Spring 模式：每个环境方向和大小不同

### 显示控制
- 默认：只显示主环境的外力箭头
- 按 `M` 键：显示所有环境的机器人和外力箭头

## 多环境支持

### Constant 模式
- 所有环境施加**相同**的力向量
- 力向量在触发时确定，施加过程中保持不变

### Spring 模式
- 每个环境施加**不同**的力向量
- 引力中心在触发时计算，每个环境独立
- 力向量每帧根据当前位置动态计算
- 适合测试不同环境下的机器人响应差异

## 典型使用场景

### 场景 1：测试平衡恢复
```python
force_mode = {'stop': 'fixtime', style': 'constant'}
# 短时推力，观察机器人是否能恢复平衡
```

### 场景 2：持续扰动测试
```python
force_mode = {'stop': 'keeping', 'style': 'constant'}
# 持续施加恒定力，测试机器人在持续干扰下的行走能力
```

### 场景 3：弹性约束测试
```python
force_mode = {'stop': 'keeping', 'style': 'spring'}
force_applicator.spring_k = 100.0
force_applicator.spring_center_bias = np.array([0.3, 0.0, 0.0])
# 模拟弹性绳索拉扯，测试机器人的约束下运动
```

### 场景 4：多环境扰动对比
```python
force_mode = {'stop': 'keeping', 'style': 'spring'}
# 按 M 显示所有环境
# 按 Ctrl+F 施加弹簧力
# 观察不同环境的机器人在不同引力中心下的表现
```

## 调试信息

启动时会输出配置信息：
```
[ForceApplicator] 已初始化，目标 body: pelvis (ID: 10)
  - 停止模式: keeping
  - 力的风格: spring
  - 弹簧系数 K: 50.0
  - 引力中心偏置: [0.2 0.2 0. ]
```

触发时会输出详细信息：
```
[ForceApplicator] Spring 模式：已计算 4 个环境的引力中心
  - 环境 0 引力中心: [0.2 0.2 0.9]
[ForceApplicator] 触发外力施加：
  - 停止模式: keeping
  - 力的风格: spring
  - 持续时间: 持续施加直到再次按 Ctrl+F
  - 弹簧系数: 50.0
  - 引力中心偏置: [0.2 0.2 0. ]
```

## 注意事项

1. **Spring 模式必须提供 data_list**
   - trigger 和 get_force_info 都需要 data_list 参数
   - bmwoyaw_multienv.py 中已正确传递

2. **引力中心在施加过程中固定**
   - 不会跟随机器人移动
   - 适合模拟固定点的吸引力

3. **弹簧力大小随距离变化**
   - 距离越远，力越大
   - 注意选择合适的 K 值避免力过大

4. **多环境独立计算**
   - Spring 模式下每个环境有独立的引力中心
   - 适合对比测试

## 故障排查

### 问题 1：Spring 模式没有施加力
**原因**：没有传递 data_list  
**解决**：确保 trigger 调用时传递了 `data_list=dlist`

### 问题 2：力的方向不对
**检查**：
- Constant 模式：检查 `force_direction` 参数
- Spring 模式：检查 `spring_center_bias` 是否符合预期

### 问题 3：力太大或太小
**调整**：
- Constant 模式：修改 `force_magnitude`
- Spring 模式：修改 `spring_k` 或调整 body 到引力中心的距离
