# Force Mode Select 参数使用指南

## 概述

`force_mode.select` 参数用于灵活控制在多环境仿真中哪些环境施加外力。这个参数提供了从"不施加"到"全部施加"再到"精确选择"的完整控制能力。

---

## Select 参数取值

### 1. `None` - 不施加外力

```python
force_mode = {
    'stop': 'keeping',
    'style': 'constant',
    'select': None  # 不对任何环境施加外力
}
```

**行为：**
- 按 Ctrl+F 后不会对任何环境施加外力
- 控制台会显示：`force_mode['select']=None，不对任何环境施加外力`

**使用场景：**
- 临时禁用外力功能，但保持代码结构
- 调试时排除外力影响

---

### 2. `0` - 仅主环境（第0个）

```python
force_mode = {
    'stop': 'keeping',
    'style': 'constant',
    'select': 0  # 仅对第0个环境施加外力
}
```

**行为：**
- 只对 `dlist[0]`（第0个环境）施加外力
- 其他环境不受影响
- 控制台显示：`将对以下环境施加外力: [0]`

**使用场景：**
- 单环境测试
- 专注观察一个环境的响应
- 与其他环境对比（有外力 vs 无外力）

---

### 3. `1` - 所有环境（默认）

```python
force_mode = {
    'stop': 'keeping',
    'style': 'constant',
    'select': 1  # 对所有环境施加外力
}
```

**行为：**
- 对所有 `dlist` 中的环境施加外力
- 如果有 4 个环境，控制台显示：`将对以下环境施加外力: [0, 1, 2, 3]`

**使用场景：**
- 测试策略在外力扰动下的鲁棒性
- 统一施加外力验证策略一致性
- **这是默认模式**（如果未指定 select 参数）

---

### 4. `[0, 2, 3]` - 自定义环境列表

```python
force_mode = {
    'stop': 'keeping',
    'style': 'spring',
    'select': [0, 2]  # 只对环境0和2施加外力
}
```

**行为：**
- 只对列表中指定索引的环境施加外力
- 环境1和3不受影响
- 控制台显示：`将对以下环境施加外力: [0, 2]`

**使用场景：**
- **精确控制**：特定环境施加外力
- **对比实验**：部分环境有扰动，部分没有
- **分组测试**：奇数环境 vs 偶数环境

---

## 列表参数的高级特性

### 自动去重

```python
force_mode = {
    'select': [0, 2, 2, 0, 3]  # 有重复
}
```

**自动处理：**
- 系统会自动去重并排序：`[0, 2, 3]`
- 控制台警告：`force_mode['select'] 列表中有重复索引，已自动去重: [0, 2, 2, 0, 3] -> [0, 2, 3]`

### 错误检测

#### 错误1：非整数元素

```python
force_mode = {
    'select': [0, 'a', 2]  # 包含字符串
}
```

**错误信息：**
```
[错误] 解析 force_mode['select'] 失败: force_mode['select'] 列表必须全是整数，当前值: [0, 'a', 2]
```

#### 错误2：索引超出范围

```python
# 假设只有 4 个环境（索引 0-3）
force_mode = {
    'select': [0, 5, 2]  # 5 超出范围
}
```

**错误信息：**
```
[错误] 解析 force_mode['select'] 失败: force_mode['select'] 列表中索引 5 超出范围 [0, 4)
```

---

## 与 Stop/Style 的组合使用

### 组合示例

#### 1. Fixtime + Constant + 单环境
```python
force_mode = {
    'stop': 'fixtime',    # 5秒后自动停止
    'style': 'constant',  # 恒定20N Y方向力
    'select': 0           # 仅环境0
}
```
**效果：** 环境0受到5秒的恒定推力，然后自动停止

#### 2. Keeping + Spring + 自定义列表
```python
force_mode = {
    'stop': 'keeping',    # 持续直到再次按键
    'style': 'spring',    # 弹簧引力
    'select': [1, 3]      # 仅环境1和3
}
```
**效果：** 环境1和3受到弹簧力牵引，按 Ctrl+F 切换开/关

#### 3. Fixtime + Spring + 所有环境
```python
force_mode = {
    'stop': 'fixtime',    # 5秒后自动停止
    'style': 'spring',    # 弹簧引力
    'select': 1           # 所有环境
}
```
**效果：** 所有环境同时受到弹簧力牵引5秒

---

## 可视化行为

### 力箭头渲染规则

**当前实现：**
1. **主环境**（current_env_idx）：
   - 总是渲染力箭头（如果该环境在 selected_envs 中）
   - 即使 `show_other_envs=False`
   
2. **其他环境**：
   - 只有当 `show_other_envs=True` (按M键) 时才渲染
   - 只渲染 selected_envs 中的环境

**示例：**

假设配置：
```python
force_mode = {'select': [0, 2]}  # 只对环境0和2施加外力
num_envs = 4                      # 总共4个环境
current_env_idx = 0               # 当前主环境是0
```

- **默认状态（show_other_envs=False）**：
  - ✅ 环境0：显示力箭头（主环境，在 selected_envs 中）
  - ❌ 环境1：不显示（不在 selected_envs 中）
  - ❌ 环境2：不显示（虽然有外力，但不是主环境且 M 未按）
  - ❌ 环境3：不显示（不在 selected_envs 中）

- **按M键后（show_other_envs=True）**：
  - ✅ 环境0：显示力箭头（主环境）
  - ❌ 环境1：不显示（不在 selected_envs 中）
  - ✅ 环境2：显示力箭头（在 selected_envs 中）
  - ❌ 环境3：不显示（不在 selected_envs 中）

---

## 实际使用流程

### 步骤1：配置 force_mode

在 `bmwoyaw_multienv.py` 中设置：

```python
force_mode = {
    'stop': 'keeping',
    'style': 'spring',
    'select': [0, 2]  # 自定义选择
}
```

### 步骤2：运行仿真

```bash
python deploy_mujoco/bmwoyaw_multienv.py --num_envs 4
```

### 步骤3：触发外力

1. 按 **Ctrl+F** - 触发外力
2. 观察控制台输出：
   ```
   [ForceApplicator] 将对以下环境施加外力: [0, 2]
   [ForceApplicator] Spring 模式：已计算 2 个环境的引力中心
   ```

### 步骤4：查看可视化

1. 默认只看到主环境（0）的力箭头
2. 按 **M** 键显示所有机器人
3. 再按 **M** 键显示所有外力箭头（环境0和2）
4. 观察环境1和3没有力箭头（未选中）

### 步骤5：停止外力

- **Keeping 模式**：再次按 **Ctrl+F**
- **Fixtime 模式**：等待 5 秒自动停止

---

## 典型应用场景

### 场景1：对比实验

**目标：** 测试外力对策略的影响

```python
force_mode = {
    'stop': 'keeping',
    'style': 'constant',
    'select': [0, 1]  # 前两个环境有外力
}
# 环境2、3作为对照组
```

**观察：**
- 环境0、1：受到外力扰动，观察恢复能力
- 环境2、3：正常运行，作为基准

### 场景2：梯度测试

**目标：** 测试不同强度的扰动

```python
# 配置1：弱扰动
force_mode = {'select': [0], ...}  # 单环境，调整 spring_k=30

# 配置2：强扰动
force_mode = {'select': [1], ...}  # 单环境，调整 spring_k=100
```

### 场景3：多方向扰动

**目标：** 测试不同方向的外力

```python
# 使用 constant 模式，修改代码中的 force_direction
# 环境0: Y方向
# 环境1: X方向（需要在代码中为不同环境设置不同方向）
```

### 场景4：轮流测试

**动态切换配置：**

```python
# 第一轮：测试环境0和1
force_mode = {'select': [0, 1]}

# 第二轮：测试环境2和3
force_mode = {'select': [2, 3]}
```

---

## 调试技巧

### 1. 验证环境选择

观察控制台输出：
```
[ForceApplicator] 将对以下环境施加外力: [0, 2]
```

### 2. 检查力向量

在 `bmwoyaw_multienv.py` 中添加调试代码：

```python
force_info = force_applicator.get_force_info(data_list=dlist)
if force_info:
    print(f"Selected envs: {force_info['selected_envs']}")
    for i, fv in enumerate(force_info['force_vectors']):
        if np.linalg.norm(fv) > 0:
            print(f"  Env {i}: force = {fv}")
```

### 3. 可视化验证

- 按 **M** 键显示所有环境
- 确认只有 selected_envs 中的环境有紫色力箭头
- 使用 ↑/↓ 键切换主环境，观察不同环境的响应

---

## 常见问题

### Q1: 设置 `select=None` 后按 Ctrl+F 没反应？

**A:** 正常行为。`select=None` 表示不对任何环境施加外力。如果需要施加外力，改为 `select=0` 或 `select=1`。

### Q2: 列表中重复的索引会重复施加外力吗？

**A:** 不会。系统会自动去重，每个环境最多施加一次外力。

### Q3: 为什么按 M 键后看不到所有环境的力箭头？

**A:** 检查：
1. 这些环境是否在 `selected_envs` 中？
2. 外力是否已激活？（按 Ctrl+F 触发）
3. 是否在正确的时间范围内？（fixtime 模式会自动停止）

### Q4: 能否动态修改 select 参数？

**A:** 当前实现中，select 参数在每次 `trigger()` 时重新解析。可以：
1. 修改 `force_applicator.force_mode['select']`
2. 再次按 Ctrl+F 触发（会重新计算 selected_envs）

---

## 进阶用法：代码级定制

### 为不同环境设置不同力参数

如果需要更复杂的控制（例如不同环境不同力大小），可以修改 `force_applicator.py`：

```python
# 在 update() 方法中
for env_i in self.selected_envs:
    # 根据环境索引调整参数
    if env_i == 0:
        force_mag = 20.0  # 环境0弱力
    elif env_i == 1:
        force_mag = 40.0  # 环境1强力
    else:
        force_mag = 30.0  # 其他环境中等力
    
    # 应用自定义力
    force_vector = self.force_direction * force_mag
    data_list[env_i].xfrc_applied[self.body_id, 0:3] = force_vector
```

---

## 总结

| Select 参数 | 行为 | 使用场景 |
|------------|------|----------|
| `None` | 不施加外力 | 临时禁用、调试 |
| `0` | 仅第0个环境 | 单环境测试、主环境专注 |
| `1` | 所有环境 | 统一测试、鲁棒性验证（默认） |
| `[0,2,3]` | 指定环境列表 | 精确控制、对比实验、分组测试 |

**关键优势：**
- ✅ 灵活性：从无到全，精确可控
- ✅ 安全性：自动验证索引范围，防止越界
- ✅ 健壮性：自动去重，避免重复施加
- ✅ 可组合：与 stop/style 参数完美配合

**最佳实践：**
1. 测试阶段：使用 `select=0` 单环境调试
2. 验证阶段：使用列表进行对比实验
3. 生产阶段：根据需求选择 `0`（专注）或 `1`（全面）
