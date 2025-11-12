# MuJoCo Perturbation Force 渲染机制分析

## 核心发现

MuJoCo 自带的 viewer 在用户按 **Ctrl+鼠标右键** 施加外力时，通过 `mjv_updateScene` 自动渲染箭头，**不需要手动创建 geom**。

## 关键 API

### 1. 启用可视化选项

必须在 `MjvOption` 中启用两个标志：

```python
opt = mujoco.MjvOption()

# 启用 perturbation force 箭头渲染
opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE.value] = 1  # index 12

# 启用 perturbation object 渲染（球体+圆柱）
opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ.value] = 1    # index 13
```

### 2. MjvPerturb 状态管理

```python
# 创建并初始化 perturbation
pert = mujoco.MjvPerturb()
mujoco.mjv_defaultPerturb(pert)

# 激活 perturbation（选择 body）
pert.select = body_id  # 目标 body 的 ID
pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE.value  # 1=平移, 2=旋转

# 初始化参考位置和缩放
mujoco.mjv_initPerturb(model, data, scene, pert)

# 设置力的方向（局部坐标系偏移）
pert.localpos = np.array([dx, dy, dz])  # 偏移向量

# 应用力到 xfrc_applied
mujoco.mjv_applyPerturbForce(model, data, pert)
```

### 3. 自动渲染

```python
# mjv_updateScene 会自动添加箭头 geom 到 scene
mujoco.mjv_updateScene(model, data, opt, pert, cam, category, scene)
```

**关键**：传入 `pert` 参数，并且 `opt.flags[12]` 和 `opt.flags[13]` 启用。

## 渲染结果

### Geom 类型

启用 `mjVIS_PERTFORCE` 和 `mjVIS_PERTOBJ` 后，scene 中会自动添加：

1. **球体 (type=2, mjGEOM_SPHERE)**：标记选中的 body 位置
   - 颜色：红色 `[0.9, 0.0, 0.0, 1.0]`
   - 位置：body 的当前位置

2. **圆柱体 (type=3, mjGEOM_CYLINDER)**：连接球体到 perturbation 点
   - 颜色：红色 `[0.9, 0.0, 0.0, 1.0]`
   - 用于显示偏移量

3. **箭头 (type=100, mjGEOM_ARROW)**：显示施加的力
   - 颜色：浅红色 `[1.0, 0.5, 0.5, 1.0]`
   - 位置：body 位置
   - 方向：通过 `mat` 矩阵定义（旋转矩阵）
   - 大小：`size[2]` 是箭头长度，与力的大小成正比

### 测试结果示例

```
找到箭头 Geom 4:
  type: 100 (mjGEOM_ARROW)
  rgba: [1.  0.5 0.5 1. ]  # 浅红色
  pos: [0.  0.  0.5]        # body 位置
  size: [0.01732051 0.01732051 0.1]  # 箭头宽度和长度
  mat: [[ 2.220446e-16 -0.000000e+00 -1.000000e+00]
        [ 0.000000e+00  1.000000e+00 -0.000000e+00]
        [ 1.000000e+00  0.000000e+00  2.220446e-16]]
```

## 与我们的实现对比

### MuJoCo 自动渲染
- ✅ 自动计算箭头长度（基于 scale 和 localpos）
- ✅ 自动管理 geom 生命周期
- ✅ 集成在 `mjv_updateScene` 中
- ✅ 颜色固定为浅红色

### 我们的手动实现
- 需要手动调用 `mjv_connector` 创建箭头
- 需要手动计算箭头起点和终点
- 可以自定义颜色（紫色）
- 可以自定义箭头长度缩放

## 建议改进

### 选项 1：完全模仿 MuJoCo（推荐用于调试）

```python
# 在 bmwoyaw_multienv.py 中
opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE.value] = 1
opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ.value] = 1

# 在主循环中传入 pert
mujoco.mjv_updateScene(model, dlist[render_index], opt, pert, cam, 
                       mujoco.mjtCatBit.mjCAT_ALL, viewer.user_scn)
```

### 选项 2：保持当前自定义实现（推荐用于生产）

优点：
- 自定义颜色区分（紫色 vs 红色）
- 独立于 MjvPerturb 状态
- 更灵活的力大小映射

缺点：
- 需要手动管理 geom

## 完整工作示例

参考：`test_perturbation.py`

运行命令：
```bash
conda activate mujoco
python test_perturbation.py
```

## 总结

MuJoCo viewer 的 perturbation 渲染通过以下步骤工作：

1. 启用 `mjVIS_PERTFORCE` 和 `mjVIS_PERTOBJ` 标志
2. 设置 `MjvPerturb` 的 `active`, `select`, `localpos`
3. 调用 `mjv_applyPerturbForce` 应用力到 `xfrc_applied`
4. 调用 `mjv_updateScene` 时传入 `pert` 参数
5. Scene 自动添加 ARROW geom（type=100）

**我们的实现不需要改变**，因为我们的目标是**自定义颜色和控制**，而不是完全模仿 MuJoCo 的交互式 perturbation。
