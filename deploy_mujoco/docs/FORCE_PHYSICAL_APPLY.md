# 外力物理施加说明文档

## 📋 概述

**当前实现已经完全支持外力的物理施加！**

外力不仅仅是可视化效果，而是真实地通过 `xfrc_applied` 施加到选中环境的指定 body 上，影响物理仿真。

---

## ✅ 实现验证

### 1. 物理施加代码（`force_applicator.py` 第 253-290 行）

```python
def update(self, data_list):
    # ...
    if self.force_mode['style'] == 'constant':
        force_vector = self.force_direction * self.force_magnitude
        
        # 只对选中的环境施加外力
        for env_i in self.selected_envs:
            if env_i < len(data_list):
                data = data_list[env_i]
                # ✅ 通过 xfrc_applied 施加物理外力
                data.xfrc_applied[self.body_id, 0:3] = force_vector
                data.xfrc_applied[self.body_id, 3:6] = 0.0  # 不施加力矩
    
    elif self.force_mode['style'] == 'spring':
        # 弹簧力模式：每个选中环境单独计算力
        for spring_i, env_i in enumerate(self.selected_envs):
            # ...
            # ✅ 通过 xfrc_applied 施加物理外力
            data.xfrc_applied[self.body_id, 0:3] = force_vector
            data.xfrc_applied[self.body_id, 3:6] = 0.0
```

**关键点：**
- ✅ 只对 `self.selected_envs` 列表中的环境施加物理外力
- ✅ 通过 `data.xfrc_applied[body_id, 0:3]` 直接修改 MuJoCo 的外力数组
- ✅ 外力会在 `mujoco.mj_step()` 时被物理引擎使用
- ✅ 未选中的环境不会受到任何物理外力影响

---

### 2. 渲染可视化代码（`bmwoyaw_multienv.py` 第 551-579 行）

```python
# 获取当前外力信息并渲染箭头
force_info = force_applicator.get_force_info(data_list=dlist)
if force_info is not None:
    force_vectors = force_info['force_vectors']  # 每个环境的力向量列表
    
    # ✅ 只渲染主环境的外力箭头
    if len(force_vectors) > current_env_idx.value:
        force_visualizer.render_force_arrow(
            viewer.user_scn,
            m,
            dlist[current_env_idx.value],
            force_info['body_id'],
            force_vectors[current_env_idx.value]  # ✅ 使用该环境的力向量
        )
    
    # ✅ 按 M 键后渲染其他环境的外力箭头
    if show_other_envs.value:
        for env_i in range(num_envs):
            if env_i == current_env_idx.value:
                continue
            if env_i < len(force_vectors):
                force_visualizer.render_force_arrow(
                    viewer.user_scn,
                    m,
                    dlist[env_i],
                    force_info['body_id'],
                    force_vectors[env_i]  # ✅ 使用该环境的力向量
                )
```

**关键点：**
- ✅ `force_vectors` 是一个列表，长度等于环境数量
- ✅ 只有被选中的环境（`selected_envs`）的 `force_vectors[i]` 是非零向量
- ✅ 未选中环境的 `force_vectors[i]` 是零向量 `[0, 0, 0]`，不会渲染箭头
- ✅ 渲染与物理施加完全一致：哪个环境施加了外力，哪个环境就会显示箭头

---

## 🎮 使用示例

### 示例 1：只对环境 0 和 2 施加弹簧力

```python
force_mode = {
    'stop': 'keeping',   # 持续施加直到再次按 Ctrl+F
    'style': 'spring',   # 弹簧力模式
    'select': [0, 2]     # 只对环境 0 和 2 施加外力
}
```

**效果：**
1. 按 `Ctrl+F` 触发外力
2. **物理层面**：环境 0 和 2 的 pelvis 受到弹簧拉力，向引力中心移动
3. **可视化层面**：
   - 默认只看到主环境（0）的紫色箭头
   - 按 `M` 键显示所有环境，可以看到环境 0 和 2 都有紫色箭头
   - 环境 1 和 3 没有箭头（因为未施加外力）
4. 再次按 `Ctrl+F` 停止外力

---

### 示例 2：对所有环境施加恒定力

```python
force_mode = {
    'stop': 'fixtime',    # 固定 5 秒后自动停止
    'style': 'constant',  # 恒定力模式
    'select': 1           # 所有环境
}
```

**效果：**
1. 按 `Ctrl+F` 触发外力
2. **物理层面**：所有环境的 pelvis 都受到 Y 方向 20N 的推力
3. **可视化层面**：
   - 默认只看到主环境的紫色箭头（方向相同）
   - 按 `M` 键显示所有环境，所有机器人都有相同方向的紫色箭头
4. 5 秒后自动停止，箭头消失

---

### 示例 3：不施加外力（调试模式）

```python
force_mode = {
    'stop': 'keeping',
    'style': 'constant',
    'select': None        # 不对任何环境施加外力
}
```

**效果：**
1. 按 `Ctrl+F` 后打印消息：`不对任何环境施加外力`
2. **物理层面**：所有环境都不受外力影响
3. **可视化层面**：没有任何箭头显示

---

## 🔍 验证方法

### 方法 1：观察机器人运动

1. 配置 `select: [0]`（只对环境 0 施加）
2. 按 `M` 键显示所有环境
3. 按 `Ctrl+F` 触发外力
4. **观察现象**：
   - ✅ 环境 0 的机器人会被推动/拉动（受物理外力影响）
   - ✅ 其他环境的机器人保持正常运动（不受外力影响）
   - ✅ 只有环境 0 显示紫色箭头

---

### 方法 2：检查外力数组

在 `bmwoyaw_multienv.py` 的主循环中添加调试打印：

```python
# 在 force_applicator.update(dlist) 之后
if force_applicator.is_active and timestep % 10 == 0:
    for i in range(num_envs):
        force = dlist[i].xfrc_applied[force_info['body_id'], 0:3]
        print(f"Env {i} xfrc_applied: {force}")
```

**预期输出（select: [0, 2]）：**
```
Env 0 xfrc_applied: [ 3.5  4.2  0.1]  # 非零向量
Env 1 xfrc_applied: [ 0.0  0.0  0.0]  # 零向量
Env 2 xfrc_applied: [ 2.8  3.9 -0.2]  # 非零向量
Env 3 xfrc_applied: [ 0.0  0.0  0.0]  # 零向量
```

---

### 方法 3：对比实验

**实验 A：select: [0]**
- 只有环境 0 的机器人受到外力，可能会摔倒或改变轨迹
- 其他环境正常运动

**实验 B：select: 1（所有环境）**
- 所有环境的机器人同时受到外力
- 所有机器人表现相似（constant 模式）或不同（spring 模式）

---

## 📊 数据流图

```
用户按 Ctrl+F
    ↓
force_applicator.trigger(data_list=dlist)
    ↓
解析 select 参数 → self.selected_envs = [0, 2]
计算引力中心（spring 模式）
    ↓
每帧调用 force_applicator.update(dlist)
    ↓
遍历 self.selected_envs:
    - 计算 force_vector
    - dlist[env_i].xfrc_applied[body_id, 0:3] = force_vector  ← ✅ 物理施加
    ↓
mujoco.mj_step(m, dlist[env_i])  ← ✅ 物理引擎使用 xfrc_applied
    ↓
force_applicator.get_force_info(data_list=dlist)
    ↓
返回 force_vectors 列表：
    [非零向量, 零向量, 非零向量, 零向量]
    ↓
渲染循环：
    - 遍历 force_vectors
    - 只渲染非零向量的箭头  ← ✅ 可视化一致
```

---

## ✨ 总结

### 当前实现的优势

1. **物理与渲染一致**：
   - 施加外力的环境 = 显示箭头的环境
   - 不存在"只渲染不施加"或"只施加不渲染"的不一致情况

2. **灵活的环境选择**：
   - `None`：调试模式，不影响任何环境
   - `0`：单环境测试
   - `1`：全环境同步
   - `[0, 2]`：自定义组合

3. **安全的错误处理**：
   - 列表索引越界：报错
   - 列表包含非整数：报错
   - 列表重复索引：自动去重

4. **高效的实现**：
   - 只对选中环境进行力计算
   - 只渲染非零力向量
   - 不浪费计算资源在未选中环境

---

## 🚀 下一步（可选）

如果需要进一步验证或扩展功能，可以考虑：

1. **添加实时外力监控**：
   - 在 UI 中显示每个环境的外力大小
   - 绘制外力随时间变化的曲线

2. **支持多个施力点**：
   - 同时在 pelvis 和 torso 施加外力
   - 不同 body 使用不同的 select 配置

3. **外力预设模式**：
   - 快捷键切换 constant/spring
   - 预设常用的 select 组合（如"左右对称"、"交替施加"）

4. **外力录制与回放**：
   - 记录外力施加的时间序列
   - 支持重放以测试机器人的鲁棒性

---

## 📝 当前配置（`bmwoyaw_multienv.py` 第 169 行）

```python
force_mode = {
    'stop': 'keeping',   # 持续施加直到再次按 Ctrl+F
    'style': 'spring',   # 弹簧力模式
    'select': [0, 2]     # 只对环境 0 和 2 施加外力（测试用）
}
```

**测试步骤：**
1. 运行 `python deploy_mujoco/bmwoyaw_multienv.py`
2. 按 `M` 键显示所有环境
3. 按 `Ctrl+F` 触发外力
4. 观察环境 0 和 2 的机器人受到弹簧拉力
5. 观察环境 1 和 3 的机器人不受影响
6. 观察只有环境 0 和 2 显示紫色箭头

---

**✅ 外力已经真实施加到物理仿真中，不仅仅是渲染效果！**
