# Ghost 渲染功能集成指南

## 问题回答总结

### 1. 透明度和颜色设置
参考 `deploy_robot-main-v1.0.0/deploy_robot/sim/viewer_plus/viewer_plus.py:32-33`：

```python
ghost_color = [0.5, 0.7, 0.5, 0.5]  # Semi-transparent green [R, G, B, A]
self._ghost_model.geom_rgba[:] = np.array(ghost_color, dtype=np.float32)
```

**配置**：
- **颜色**：`[0.5, 0.7, 0.5, 0.5]` - 半透明绿色
- **R** (Red): 0.5
- **G** (Green): 0.7
- **B** (Blue): 0.5
- **A** (Alpha): 0.5 (50% 透明度)

### 2. 渲染架构实现

**deploy_robot-main-v1.0.0 的架构**：
```python
# 初始化阶段：
self._ghost_model = copy.deepcopy(model)  # 深拷贝主模型
self._ghost_model.geom_rgba[:] = ghost_color  # 设置颜色
self._viz_data = mujoco.MjData(self._ghost_model)  # 创建 ghost 数据

# 每帧渲染：
self._viz_data.qpos[:] = ghost_qpos  # 设置姿态
mujoco.mj_forward(self._ghost_model, self._viz_data)  # 前向动力学
mujoco.mjv_addGeoms(  # 添加到用户场景
    self._ghost_model,
    self._viz_data,
    self._vopt,  # 启用透明渲染
    self._pert,
    self._catmask,
    self.viewer.user_scn
)
```

**本项目的迁移方案**：
1. 创建 `GhostRenderer` 类（已完成：`ghost_renderer.py`）
2. 在主循环初始化阶段创建 `ghost_renderer = GhostRenderer(m)`
3. 在每帧控制循环中：
   - 从 `motionref` 提取参考轨迹数据
   - 构造 `ghost_qpos`
   - 调用 `ghost_renderer.set_ghost_qpos(ghost_qpos)`
4. 在渲染阶段（`viewer.sync()` 之前）：
   - 调用 `ghost_renderer.render_ghost(viewer.user_scn)`

### 3. 轨迹中 root 位置和姿态的决定

**deploy_robot-main-v1.0.0 的做法**：
```python
# 从 policy 输出的轨迹数据中提取：
policy_body_pos_w = policy.output_tensors["body_pos_w"]  # (1, 14, 3)
policy_body_quat_w = policy.output_tensors["body_quat_w"]  # (1, 14, 4)

# 使用第一个 body（通常是 root/base）：
base_pos_policy = policy_body_pos_w[0, 0, :]  # (3,) - XYZ
base_quat_policy = policy_body_quat_w[0, 0, :]  # (4,) - quaternion

# 构造 ghost_qpos：
ghost_qpos = np.zeros_like(current_qpos)
ghost_qpos[0:3] = base_pos_policy   # root 位置
ghost_qpos[3:7] = base_quat_policy  # root 姿态
```

**本项目的实现**：
```python
# 你有 motionref 数据：
motionrefpos = motionref["body_pos_w"]  # (时间步, 14, 3)
motionrefquat = motionref["body_quat_w"]  # (时间步, 14, 4)

# 提取当前时间步的 root（索引 9 是 pelvis）：
base_pos_ref = motionrefpos[timestep, 9, :]  # (3,)
base_quat_ref = motionrefquat[timestep, 9, :]  # (4,)

# 注意：四元数格式可能需要转换
# motionref 中可能是 [x, y, z, w] 格式
# MuJoCo qpos 需要 [w, x, y, z] 格式
# 需要检查并转换：
if base_quat_ref 是 [x,y,z,w]:
    base_quat_mujoco = np.array([base_quat_ref[3], base_quat_ref[0], 
                                  base_quat_ref[1], base_quat_ref[2]])
```

### 4. DOF 位置顺序变换

**deploy_robot-main-v1.0.0 的做法**：
```python
# policy_joint_pos 是 joint_names 顺序
ghost_joint_pos_dof = robot.joint2dof(policy_joint_pos)

# joint2dof 实现（g1.py:331-340）：
def joint2dof(self, joint_array: np.ndarray) -> np.ndarray:
    """Convert joint space array to dof space array."""
    return np.asarray([
        joint_array[self.joint_names.index(q)] 
        for q in self.dof_names
    ])
```

**本项目的实现**：
```python
# 你的 motionref 数据是 joint_seq 顺序
# 需要转换为 joint_xml 顺序（即 dof 顺序）

# 从 motionref 提取关节角度（joint_seq 顺序）：
joint_pos_ref_seq = motionrefinputpos[timestep, :]  # joint_seq 顺序

# 转换为 joint_xml（dof）顺序：
ghost_joint_pos_dof = np.array([
    joint_pos_ref_seq[joint_seq.index(joint)] 
    for joint in joint_xml
])

# ⚠️ 重要：不要重复转换！
# 这个转换等价于你在 bmwoyaw_multienv.py:346 使用的：
# target_dof_pos_i = np.array([target_dof_pos_i[joint_seq.index(joint)] for joint in joint_xml])
```

---

## 集成步骤

### Step 1: 导入 GhostRenderer
在 `bmwoyaw_multienv.py` 开头添加：
```python
from ghost_renderer import GhostRenderer
```

### Step 2: 初始化 GhostRenderer
在模型加载后、主循环前添加：
```python
# 初始化 ghost 渲染器
ghost_renderer = GhostRenderer(m)
show_ghost = Value('i', 1)  # 是否显示 ghost（0=不显示，1=显示，默认显示）
```

### Step 3: 添加 Ctrl+G 快捷键
在 `key_callback` 函数中添加：
```python
# Ctrl+G - 切换 ghost 显示
elif (key == 71 or key == 103) and ctrl_pressed.value:  # 'G' 或 'g' + Ctrl
    show_ghost.value = 1 - show_ghost.value
    status = "显示" if show_ghost.value else "隐藏"
    print(f"[Ghost 可视化] {status} 参考轨迹")
    ctrl_pressed.value = 0
```

### Step 4: 在控制循环中构造 ghost_qpos
在 `if counter % control_decimation == 0:` 块中，奖励计算之后添加：
```python
# ===== 构造 Ghost Qpos（新增）=====
try:
    # 提取参考轨迹的 root 位置和姿态
    base_pos_ref = motionrefpos[timestep, 9, :]  # (3,) - pelvis 位置
    base_quat_ref = motionrefquat[timestep, 9, :]  # (4,) - pelvis 四元数
    
    # 注意：检查四元数格式，motionref 可能是 [x,y,z,w]，需要转换为 [w,x,y,z]
    # 如果已经是 [w,x,y,z] 格式，则不需要转换
    base_quat_mujoco = np.array([base_quat_ref[3], base_quat_ref[0], 
                                  base_quat_ref[1], base_quat_ref[2]])
    
    # 提取参考轨迹的关节角度（joint_seq 顺序）
    joint_pos_ref_seq = motionrefinputpos[timestep, :]  # (num_joints,)
    
    # 转换为 dof 顺序（joint_xml 顺序）
    ghost_joint_pos_dof = np.array([
        joint_pos_ref_seq[joint_seq.index(joint)] 
        for joint in joint_xml
    ])
    
    # 构造完整的 ghost_qpos
    ghost_qpos = ghost_renderer.construct_ghost_qpos(
        base_pos=base_pos_ref,
        base_quat=base_quat_mujoco,
        joint_pos_dof_order=ghost_joint_pos_dof,
        current_qpos=d.qpos
    )
    
    # 设置 ghost 姿态
    ghost_renderer.set_ghost_qpos(ghost_qpos)
    
except Exception as e:
    print(f"[警告] Ghost qpos 构造失败: {e}")
# ===== Ghost Qpos 构造结束 =====
```

### Step 5: 在渲染循环中渲染 ghost
在 `viewer.sync()` 之前，奖励图表渲染之后添加：
```python
# ===== 渲染 Ghost（新增）=====
if show_ghost.value:
    try:
        ghost_renderer.render_ghost(viewer.user_scn)
    except Exception as e:
        if timestep % 100 == 0:
            print(f"[警告] Ghost 渲染失败: {e}")
# ===== Ghost 渲染结束 =====
```

---

## 键盘控制总结

集成完成后，键盘控制为：
- **↑/↓ 箭头键**：切换主环境
- **M 键**：显示/隐藏其他环境
- **Ctrl+R**：显示/隐藏奖励曲线窗口
- **Ctrl+G**：显示/隐藏参考轨迹 ghost ✨（新增）

---

## 注意事项

1. **四元数格式检查**：
   - MuJoCo qpos 使用 `[w, x, y, z]` 格式
   - 你的 motionref 可能使用 `[x, y, z, w]` 格式（参考 bmwoyaw_multienv.py:242-243）
   - 需要根据实际数据格式进行转换

2. **关节顺序转换**：
   - 不要重复转换！`motionrefinputpos` 是 joint_seq 顺序，只需转换一次到 joint_xml

3. **渲染顺序**：
   - Ghost 应该在 `viewer.user_scn.ngeom = 0` 之后渲染
   - Ghost 应该在其他环境渲染之前或之后都可以
   - 确保在 `viewer.sync()` 之前完成所有渲染

4. **性能优化**：
   - Ghost 渲染很轻量，对性能影响小
   - 如果需要，可以降低 ghost 更新频率（例如每 N 帧更新一次）

---

## 测试建议

1. 先测试 ghost 是否显示（半透明绿色）
2. 检查 ghost 位置是否与参考轨迹匹配
3. 测试 Ctrl+G 切换功能
4. 检查多环境模式下 ghost 是否正确显示
5. 验证四元数格式是否正确（ghost 姿态应该与参考一致）
