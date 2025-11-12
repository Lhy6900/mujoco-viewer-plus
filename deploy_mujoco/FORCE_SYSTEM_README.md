# 外力施加系统使用说明

## 功能概述

外力施加系统允许你在 MuJoCo 仿真中对所有环境的指定 body（默认为 `pelvis`）施加定时外力，并通过紫色箭头可视化。

**✅ 已修复 API 问题**: 现在使用正确的 `mjv_connector` API 进行箭头渲染。

## 快速开始

### 运行测试

```bash
cd /home/ubuntu/deploy_mini
conda activate mujoco
python deploy_mujoco/bmwoyaw_multienv.py --num_envs 2
```

按 **Ctrl+F** 触发外力，应该看到紫色箭头出现在 pelvis 上。

### 快捷键

在仿真运行时，按下 **Ctrl+F** 即可触发外力施加：
- **持续时间**: 5 秒
- **力大小**: 20 N
- **力方向**: Y 正方向（向右）
- **可视化**: 紫色箭头（RGBA: [0.8, 0.0, 0.8, 0.6]）

### 模块组成

1. **force_applicator.py** - 外力施加逻辑
   - `ForceApplicator` 类：管理外力的施加、更新和停止
   - 支持自定义力大小、方向和持续时间

2. **force_visualizer.py** - 外力可视化
   - `ForceVisualizer` 类：在 MuJoCo viewer 中渲染外力箭头
   - 紫色半透明箭头，类似 viewer 自带的红色拖拽箭头

3. **bmwoyaw_multienv.py** - 主程序集成
   - 在按键回调中处理 Ctrl+F 触发
   - 在主循环中更新外力施加器
   - 在渲染阶段可视化外力箭头

## 自定义使用

### 修改外力参数

在 `bmwoyaw_multienv.py` 的按键回调中，可以修改触发参数：

```python
# Ctrl+F - 触发外力施加
elif (key == 70 or key == 102) and ctrl_pressed.value:  # 'F' 或 'f' + Ctrl
    force_applicator.trigger(
        duration=10.0,  # 持续时间改为 10 秒
        force_magnitude=50.0,  # 力大小改为 50N
        force_direction=np.array([1.0, 0.0, 0.0])  # 力方向改为 X 正方向
    )
    ctrl_pressed.value = 0
```

### 修改目标 body

在初始化时指定不同的 body：

```python
# 初始化外力施加器（指定 body 为 "torso_link"）
force_applicator = ForceApplicator(m, body_name="torso_link")
```

### 修改箭头颜色

在初始化可视化器时指定颜色：

```python
# 初始化外力可视化器（使用红色）
force_visualizer = ForceVisualizer(arrow_color=[1.0, 0.0, 0.0, 0.8])  # RGBA
```

## API 参考

### ForceApplicator

#### 初始化
```python
force_applicator = ForceApplicator(model, body_name="pelvis")
```

#### 触发外力
```python
force_applicator.trigger(
    duration=5.0,                          # 持续时间（秒）
    force_magnitude=20.0,                  # 力大小（N）
    force_direction=np.array([0, 1, 0])    # 力方向（归一化向量）
)
```

#### 更新状态（每帧调用）
```python
is_active = force_applicator.update(dlist)  # dlist: MjData 列表
```

#### 停止外力
```python
force_applicator.stop(dlist)
```

#### 获取外力信息
```python
force_info = force_applicator.get_force_info()
# 返回: {'body_id', 'force_vector', 'is_active', 'remaining_time', ...}
# 如果未激活返回 None
```

### ForceVisualizer

#### 初始化
```python
force_visualizer = ForceVisualizer(arrow_color=[0.8, 0.0, 0.8, 0.6])  # RGBA
```

#### 渲染外力箭头（使用正确的 MuJoCo API）
```python
force_visualizer.render_force_arrow(
    viewer_scene,   # viewer.user_scn
    model,          # MjModel
    data,           # MjData
    body_id,        # body ID
    force_vector    # 力向量 (3,)
)
```

**关键实现细节:**
- 使用 `mujoco.mjv_initGeom()` 初始化 geom
- 使用 `mujoco.mjv_connector(geom, type, width, from_pos, to_pos)` 创建箭头
- 箭头类型: `mujoco.mjtGeom.mjGEOM_ARROW`
- 参考 `test_connector.py` 查看完整测试示例

## 技术细节

### 外力施加原理

使用 MuJoCo 的 `xfrc_applied` 数组在指定 body 上施加外力：
```python
data.xfrc_applied[body_id, 0:3] = force_vector  # 前3个分量是力
data.xfrc_applied[body_id, 3:6] = 0.0           # 后3个分量是力矩
```

### 计时机制

使用 Python 的 `time.time()` 记录开始时间，每帧检查是否超时：
```python
elapsed_time = time.time() - self.start_time
if elapsed_time >= self.duration:
    self.stop(data_list)
```

### 多环境支持

外力同时施加到所有环境的对应 body 上，确保训练一致性。

### 可视化实现（✅ 已修复）

**正确的 MuJoCo API 使用:**

使用 `mjv_initGeom` 初始化 + `mjv_connector` 创建箭头几何体：

```python
# 初始化 geom
mujoco.mjv_initGeom(
    geom,
    type=mujoco.mjtGeom.mjGEOM_ARROW,
    size=np.zeros(3),
    pos=np.zeros(3),
    mat=np.eye(3).flatten(),
    rgba=self.arrow_color
)

# 创建连接器（箭头）
mujoco.mjv_connector(
    geom,
    mujoco.mjtGeom.mjGEOM_ARROW,
    self.arrow_width,
    body_pos,      # 起点
    arrow_end      # 终点
)

# 设置颜色
geom.rgba[:] = self.arrow_color
```

**❌ 错误用法（旧版本）:**
```python
# mjv_makeConnector 不存在于 mujoco 模块中
mujoco.mjv_makeConnector(...)  # AttributeError!
```

**验证方法:**
运行 `test_connector.py` 验证 API 正确性：
```bash
conda activate mujoco
python test_connector.py
```

## 注意事项

1. **body 名称**: 确保指定的 body 名称在 URDF/XML 中存在
2. **力的方向**: 力方向会自动归一化，无需手动归一化
3. **可视化性能**: 箭头渲染很轻量，不会影响仿真性能
4. **多环境一致性**: 外力同时施加到所有环境，确保并行训练的一致性
5. **MuJoCo API**: 使用 `mjv_connector` 而非不存在的 `mjv_makeConnector`

## 示例场景

### 测试机器人抗扰能力
按 Ctrl+F 施加侧向推力，观察机器人的平衡恢复能力。

### 模拟外部干扰
在机器人行走或执行任务时施加外力，测试鲁棒性。

### 训练数据增强
在训练过程中随机施加外力，提升策略的泛化能力。

## 故障排除

### 箭头不显示
- 检查 `show_main_ghost` 或 `show_other_ghosts` 是否关闭了场景渲染
- 确认 `force_applicator.get_force_info()` 返回非 None
- 运行 `test_connector.py` 确认 MuJoCo API 正常工作

### 外力无效果
- 检查 body 名称是否正确
- 确认 `force_applicator.update(dlist)` 在每帧被调用
- 检查仿真步长和积分器设置

### 性能问题
- 外力箭头渲染很轻量，如仍有问题可关闭可视化
- 使用 `if timestep % N == 0` 降低箭头更新频率
