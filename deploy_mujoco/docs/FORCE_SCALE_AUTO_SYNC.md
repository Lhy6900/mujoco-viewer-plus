# Force Scale 自动同步机制

## 问题背景

在 **Spring 模式**下，为了让渲染的外力箭头终点准确指向真实的引力中心，必须满足以下数学关系：

```
force_scale = 1 / spring_k
```

**原因分析：**
- 物理层面：弹簧力 `F = k × distance`（距离越远力越大）
- 渲染层面：箭头长度 `arrow_length = F × force_scale`
- 箭头终点：`force_end = body_pos + direction × arrow_length`

要让 `force_end = spring_center`，必须满足：
```
arrow_length = distance
=> F × force_scale = distance
=> (k × distance) × force_scale = distance
=> k × force_scale = 1
=> force_scale = 1 / k
```

**之前的痛点：**
- 修改 `spring_k` 后，需要手动同步修改 `force_scale`
- 两个参数分散在不同类中（ForceApplicator 和 ForceVisualizer）
- 容易忘记同步，导致渲染不准确

---

## 解决方案：自动同步 force_scale（方案1）

### 核心思想
**用户只需修改 `spring_k`，`force_scale` 自动计算**

### 实现机制

#### 1. ForceApplicator.get_force_info() 返回 force_scale
```python
result = {
    ...
    'force_scale': 1.0 / self.spring_k if self.force_mode['style'] == 'spring' else 0.02,
}
```

- **Spring 模式**：`force_scale = 1 / spring_k`（自动计算）
- **Constant 模式**：`force_scale = 0.02`（固定默认值）

#### 2. ForceVisualizer.render_force_arrow() 使用动态 force_scale
```python
def render_force_arrow(self, viewer_scene, model, data, body_id, force_vector, force_scale=None):
    # 使用传入的 force_scale，如果没有传入则使用默认值
    current_force_scale = force_scale if force_scale is not None else self.force_scale
    arrow_length = force_magnitude * current_force_scale
    ...
```

#### 3. bmwoyaw_multienv.py 传递 force_scale
```python
force_scale = force_info.get('force_scale', 0.02)
force_visualizer.render_force_arrow(
    viewer.user_scn, m, data, body_id, force_vector,
    force_scale=force_scale  # 传递动态 force_scale
)
```

---

## 使用方法

### 调整弹簧系数 K
只需在 `force_applicator.py` 中修改 `spring_k`，渲染会自动正确：

```python
# 在 ForceApplicator.__init__() 中：
self.spring_k = 100.0  # 从 50 改为 100

# force_scale 会自动变为 1/100 = 0.01（无需手动修改）
```

### 测试验证

#### 测试场景1：K = 50（默认）
```python
self.spring_k = 50.0
# 自动计算：force_scale = 1/50 = 0.02
# 结果：箭头终点准确指向引力中心 ✓
```

#### 测试场景2：K = 100（强弹簧）
```python
self.spring_k = 100.0
# 自动计算：force_scale = 1/100 = 0.01
# 结果：箭头更短（因为同样距离产生的力更大），但终点仍准确 ✓
```

#### 测试场景3：K = 25（弱弹簧）
```python
self.spring_k = 25.0
# 自动计算：force_scale = 1/25 = 0.04
# 结果：箭头更长（因为同样距离产生的力更小），但终点仍准确 ✓
```

---

## 优势

1. **用户体验好**：只需修改物理参数 `spring_k`，渲染自动正确
2. **逻辑清晰**：物理参数（k）决定渲染参数（scale），单向依赖
3. **保持统一**：Constant 和 Spring 模式都用相同的渲染逻辑
4. **易于扩展**：未来加入其他 style 时，渲染逻辑仍然适用
5. **符合单一数据源原则**：`spring_k` 是唯一的"真相来源"

---

## 数学验证

### 推导过程
假设 body 当前位置为 `P`，引力中心为 `C`，距离为 `d = |C - P|`

1. **物理层面**：弹簧力大小 `F = k × d`
2. **渲染层面**：箭头长度 `L = F × force_scale = k × d × force_scale`
3. **要求箭头终点 = 引力中心**：`L = d`
4. **代入得**：`k × d × force_scale = d`
5. **化简得**：`force_scale = 1 / k` ✓

### 数值示例
- `spring_k = 50`，`distance = 0.5m`
- 物理力：`F = 50 × 0.5 = 25N`
- 箭头长度：`L = 25 × (1/50) = 25 × 0.02 = 0.5m`
- 箭头终点：`P + direction × 0.5 = P + (C-P) = C` ✓（正确指向引力中心）

---

## 相关文件

- `force_applicator.py`：返回 `force_scale` 字段
- `force_visualizer.py`：接受并使用 `force_scale` 参数
- `bmwoyaw_multienv.py`：传递 `force_scale` 到渲染函数

---

## 常见问题

### Q1: Constant 模式下的 force_scale 是多少？
**A:** 固定为 `0.02`（1N = 0.02m），不随参数变化。

### Q2: 如果想让箭头更长/更短怎么办？
**A:** 
- **Spring 模式**：调整 `spring_k`（K 越大箭头越短，K 越小箭头越长）
- **Constant 模式**：可以修改 ForceVisualizer 的默认 `self.force_scale`

### Q3: 为什么不直接用 spring_center 作为箭头终点？
**A:** 因为那样会破坏 Constant 和 Spring 模式的渲染统一性，而且 Spring 模式下箭头长度失去了"力大小"的物理意义。使用 `force_scale = 1/k` 既保持了物理意义，又保证了渲染准确性。

### Q4: 如果我想手动覆盖 force_scale 怎么办？
**A:** 可以在调用 `render_force_arrow()` 时直接传入自定义的 `force_scale` 参数，它会覆盖自动计算的值。

---

**总结**：这个机制让用户只需关注物理参数（弹簧系数 K），渲染系统会自动确保可视化的准确性，大大降低了使用门槛和出错概率。
