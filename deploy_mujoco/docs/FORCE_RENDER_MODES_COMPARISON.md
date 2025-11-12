# 外力箭头渲染模式对比使用指南

## 快速开始

运行测试脚本：
```bash
./test_force_render_modes.sh
```

或直接运行：
```bash
cd /home/ubuntu/deploy_mini
conda activate mujoco
python deploy_mujoco/bmwoyaw_multienv.py --num_envs 2
```

## 快捷键说明

| 快捷键 | 功能 | 说明 |
|--------|------|------|
| **Ctrl+F** | 施加外力 | 对所有环境的 pelvis 施加 Y 方向 20N 外力，持续 5 秒 |
| **Ctrl+V** | 切换渲染模式 | 在两种箭头渲染方式之间切换 |
| Ctrl+G | 切换主环境 Ghost | 显示/隐藏主环境的参考轨迹 |
| Ctrl+M | 切换其他环境 Ghost | 显示/隐藏其他环境的参考轨迹 |
| Ctrl+R | 切换奖励曲线 | 显示/隐藏奖励曲线窗口 |
| ↑/↓ | 切换主环境 | 在不同环境之间切换显示 |
| M | 切换其他环境显示 | 显示/隐藏其他环境的机器人 |

## 两种渲染模式对比

### 模式 1：MuJoCo 原生方式（默认）

**方法名称：** `render_force_perturb`

**特点：**
- ✅ 使用 MuJoCo 内置的 `MjvPerturb` 机制
- ✅ 与 Ctrl+鼠标右键拖拽的箭头完全一致
- ✅ 自动计算箭头长度和方向
- ❌ **箭头颜色固定为浅红色** `[1.0, 0.5, 0.5, 1.0]`
- ❌ 需要重新调用 `mjv_updateScene`（性能较低）
- ⚠️ 只渲染主环境的箭头（多环境性能考虑）

**视觉效果：**
```
箭头颜色: 浅红色 (淡粉色)
箭头类型: mjGEOM_ARROW (type=100)
还包含: 红色球体标记 + 红色圆柱连接线
```

**适用场景：**
- 调试和验证 MuJoCo 内部机制
- 学习 perturbation 系统
- 需要与原生行为完全一致

### 模式 2：手动方式

**方法名称：** `render_force_arrow`

**特点：**
- ✅ **自定义紫色箭头** `[0.8, 0.0, 0.8, 0.6]`
- ✅ 性能更好（不需要重新 updateScene）
- ✅ 为所有环境渲染箭头
- ✅ 独立于 MjvPerturb 状态
- 🔵 需要手动使用 `mjv_connector` 创建箭头

**视觉效果：**
```
箭头颜色: 紫色 (半透明)
箭头类型: mjGEOM_ARROW (type=100)
纯箭头: 无额外标记物
```

**适用场景：**
- 生产环境使用（推荐）
- 多环境并行仿真
- 需要区分不同类型的力（颜色编码）

## 对比测试步骤

### 步骤 1：查看原生方式（浅红色）

1. 启动程序（默认使用原生方式）
2. 按 **Ctrl+F** 施加外力
3. 观察：
   - ✓ 主环境的 pelvis 上出现**浅红色箭头**
   - ✓ 有红色球体标记和圆柱连接线
   - ✓ 箭头长度与力大小成正比
   - ✓ 只有主环境有箭头

### 步骤 2：切换到手动方式（紫色）

1. 按 **Ctrl+V** 切换模式
2. 等待当前外力结束（5 秒后）
3. 再次按 **Ctrl+F** 施加外力
4. 观察：
   - ✓ **所有环境**的 pelvis 上都出现**紫色箭头**
   - ✓ 纯箭头，无额外标记
   - ✓ 箭头半透明
   - ✓ 渲染更流畅（性能更好）

### 步骤 3：来回切换对比

1. 按 **Ctrl+V** 切换回原生方式
2. 按 **Ctrl+F** 施加外力
3. 观察浅红色箭头
4. 重复步骤 1-3 多次对比

## 预期输出

### 终端输出示例

```
[外力系统] 已初始化 ForceApplicator 和 ForceVisualizer
  ├─ 默认渲染模式: MuJoCo 原生方式 (render_force_perturb)，箭头颜色: 浅红色
  ├─ 按 Ctrl+V 切换到手动方式 (render_force_arrow)，箭头颜色: 紫色
  └─ 按 Ctrl+F 施加外力（Y 方向 20N，持续 5 秒）
...
[外力可视化] 切换到 手动方式 (render_force_arrow)，箭头颜色: 紫色
...
[外力可视化] 切换到 MuJoCo 原生方式 (render_force_perturb)，箭头颜色: 浅红色
```

## 技术细节

### 原生方式内部实现

```python
# 1. 创建 MjvPerturb 对象
pert = mujoco.MjvPerturb()
mujoco.mjv_defaultPerturb(pert)

# 2. 设置目标 body
pert.select = body_id
pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE.value

# 3. 初始化（计算 scale）
mujoco.mjv_initPerturb(model, data, scene, pert)

# 4. 计算 localpos（反推公式）
# force = -localpos * scale * localmass * 73.316
localpos = -force_vector / (scale * localmass * 73.316)
pert.localpos[:] = localpos

# 5. 应用力和渲染
mujoco.mjv_applyPerturbForce(model, data, pert)
opt.flags[mjVIS_PERTFORCE] = 1
opt.flags[mjVIS_PERTOBJ] = 1
mujoco.mjv_updateScene(model, data, opt, pert, cam, category, scene)
```

### 手动方式内部实现

```python
# 1. 计算箭头终点
force_magnitude = np.linalg.norm(force_vector)
arrow_length = force_magnitude * 0.02  # 缩放因子
arrow_end = body_pos + force_direction * arrow_length

# 2. 初始化 geom
geom = scene.geoms[scene.ngeom]
mujoco.mjv_initGeom(geom, type=mjGEOM_ARROW, ...)

# 3. 创建箭头连接器
mujoco.mjv_connector(geom, mjGEOM_ARROW, width, body_pos, arrow_end)

# 4. 设置颜色
geom.rgba[:] = [0.8, 0.0, 0.8, 0.6]  # 紫色

# 5. 增加计数
scene.ngeom += 1
```

## 性能对比

| 指标 | 原生方式 | 手动方式 |
|------|---------|---------|
| 渲染环境数 | 1（主环境） | 全部环境 |
| updateScene 调用 | 需要 | 不需要 |
| 每帧开销 | ~1-2ms | ~0.1ms |
| 适合场景数 | 1-2 个 | 4+ 个 |

## 推荐使用

### 生产环境（推荐手动方式）
✅ 多环境并行训练  
✅ 需要高帧率渲染  
✅ 需要颜色区分不同类型的力  

### 调试环境（可选原生方式）
🔍 学习 MuJoCo perturbation 机制  
🔍 验证力的计算是否正确  
🔍 对比原生行为  

## 故障排除

### 问题 1：看不到箭头

**原因：** 外力未激活

**解决：** 按 Ctrl+F 触发外力

### 问题 2：原生方式箭头太短

**原因：** 不同的 body 有不同的 scale

**解决：** 这是正常的，MuJoCo 根据 body 大小自动缩放

### 问题 3：切换模式后箭头消失

**原因：** 需要重新按 Ctrl+F 施加外力

**解决：** 等待当前外力结束后，重新按 Ctrl+F

## 相关文档

- `force_visualizer.py` - 外力可视化器实现
- `FORCE_VISUALIZER_GUIDE.md` - 详细 API 文档
- `PERTURBATION_RENDERING_DISCOVERY.md` - MuJoCo 内部机制
- `test_force_perturb_visualizer.py` - 单元测试

## 总结

- **默认模式**：原生方式（浅红色）
- **切换快捷键**：Ctrl+V
- **施加外力**：Ctrl+F
- **推荐生产使用**：手动方式（紫色，性能好）
- **推荐调试使用**：原生方式（浅红色，与 MuJoCo 一致）
