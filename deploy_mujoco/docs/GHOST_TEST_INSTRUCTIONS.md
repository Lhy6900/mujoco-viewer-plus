# Ghost 功能测试说明

## ✅ 已完成的修改

### 1. 导入 GhostRenderer
- 已添加 `from ghost_renderer import GhostRenderer`

### 2. 初始化 Ghost 渲染器
- 创建 `ghost_renderer = GhostRenderer(m)`
- 添加 `show_ghost = Value('i', 1)` 标志（默认显示）

### 3. 键盘快捷键
- 添加 **Ctrl+G** 快捷键切换 ghost 显示/隐藏

### 4. Ghost Qpos 构造
在控制循环中（每帧）：
- 提取 `motionrefpos[timestep, 9, :]` 作为 root 位置
- 提取 `motionrefquat[timestep, 9, :]` 作为 root 姿态
- 转换四元数格式：`[x,y,z,w]` → `[w,x,y,z]`
- 转换关节顺序：`joint_seq` → `joint_xml`（dof 顺序）
- 构造完整 `ghost_qpos` 并设置到 ghost 渲染器

### 5. Ghost 渲染
在渲染循环中：
- 调用 `ghost_renderer.render_ghost(viewer.user_scn)`
- 在 `viewer.user_scn.ngeom = 0` 之后、其他环境渲染之前

---

## 🎮 键盘控制总览

- **↑/↓ 箭头键**：切换主环境
- **M 键**：显示/隐藏其他环境
- **Ctrl+R**：显示/隐藏奖励曲线窗口
- **Ctrl+G**：显示/隐藏参考轨迹 ghost ✨

---

## 🧪 测试步骤

### 1. 基础测试
```bash
cd /home/ubuntu/deploy_mini/deploy_mujoco
python bmwoyaw_multienv.py --num_envs=1
```

**预期结果**：
- ✅ 程序正常启动
- ✅ 看到控制台打印：`[Ghost 可视化] GhostRenderer 已初始化，将显示半透明绿色参考轨迹`
- ✅ 看到半透明绿色的 ghost 机器人（参考轨迹）
- ✅ Ghost 和主机器人同时移动，但姿态略有不同

### 2. 切换测试
- 按 **Ctrl+G**，ghost 应该消失
- 再按 **Ctrl+G**，ghost 应该重新出现
- 控制台打印：`[Ghost 可视化] 显示/隐藏参考轨迹`

### 3. 多环境测试
```bash
python bmwoyaw_multienv.py --num_envs=2
```

**预期结果**：
- ✅ Ghost 只在主环境（当前选中的环境）显示
- ✅ 按 ↑/↓ 切换环境时，ghost 跟随主环境
- ✅ 按 M 键显示其他环境时，ghost 仍然只在主环境显示

### 4. 联合测试
- 按 **M** 键显示所有环境
- 按 **Ctrl+R** 切换奖励曲线
- 按 **Ctrl+G** 切换 ghost
- 所有功能应该独立工作，互不干扰

---

## 🔍 验证要点

### Ghost 颜色和透明度
- **颜色**：半透明绿色 `[R=0.5, G=0.7, B=0.5, A=0.5]`
- **透明度**：应该能看透 ghost，看到背景或其他机器人

### Ghost 姿态准确性
- Ghost 的位置应该与 `motionrefpos[timestep, 9, :]` 匹配
- Ghost 的姿态应该与 `motionrefquat[timestep, 9, :]` 匹配
- Ghost 的关节角度应该与 `motionrefinputpos[timestep, :]` 匹配

### 四元数格式检查
如果 ghost 姿态看起来不对（例如倒置、旋转错误），可能是四元数格式问题：

**当前代码假设**：
```python
# motionrefquat 是 [x, y, z, w] 格式
base_quat_mujoco = np.array([
    base_quat_ref[3],  # w
    base_quat_ref[0],  # x
    base_quat_ref[1],  # y
    base_quat_ref[2]   # z
])
```

**如果 motionrefquat 已经是 [w, x, y, z] 格式**，则修改为：
```python
base_quat_mujoco = base_quat_ref  # 直接使用，不需要转换
```

---

## 🐛 常见问题排查

### 问题 1：Ghost 不显示
- 检查控制台是否有 `[警告] Ghost qpos 构造失败` 或 `[警告] Ghost 渲染失败`
- 检查 `show_ghost.value` 是否为 1
- 检查 `ghost_renderer.set_ghost_qpos()` 是否被调用

### 问题 2：Ghost 姿态错误
- 检查四元数格式转换是否正确
- 打印 `base_quat_ref` 和 `base_quat_mujoco` 对比
- 检查 `motionrefquat` 的数据格式

### 问题 3：Ghost 关节角度错误
- 检查关节顺序转换：`joint_seq` → `joint_xml`
- 确认没有重复转换（`motionrefinputpos` 只转换一次）
- 打印 `joint_pos_ref_seq` 和 `ghost_joint_pos_dof` 对比

### 问题 4：性能问题
- Ghost 渲染非常轻量，不应影响性能
- 如果有性能问题，可以降低 ghost 更新频率：
  ```python
  if timestep % 2 == 0:  # 每2帧更新一次 ghost
      ghost_renderer.set_ghost_qpos(ghost_qpos)
  ```

---

## 📊 调试技巧

### 打印 Ghost 信息
在构造 ghost_qpos 后添加：
```python
if timestep % 50 == 0:
    print(f"\n[Ghost Debug @ step {timestep}]")
    print(f"  base_pos: {base_pos_ref}")
    print(f"  base_quat_ref: {base_quat_ref}")
    print(f"  base_quat_mujoco: {base_quat_mujoco}")
    print(f"  joint_pos shape: {ghost_joint_pos_dof.shape}")
    print(f"  ghost_qpos shape: {ghost_qpos.shape}")
```

### 对比主环境和 Ghost
```python
if timestep % 50 == 0:
    print(f"\n[对比 @ step {timestep}]")
    print(f"  主环境 root pos: {d.qpos[0:3]}")
    print(f"  Ghost   root pos: {ghost_qpos[0:3]}")
    print(f"  位置差异: {np.linalg.norm(d.qpos[0:3] - ghost_qpos[0:3]):.4f} 米")
```

---

## ✨ 预期效果

运行正常后，你应该看到：
1. **主机器人**：正常颜色，执行策略输出的动作
2. **Ghost 机器人**：半透明绿色，显示参考轨迹（ground truth）
3. **两者对比**：可以直观看出策略跟踪效果
4. **奖励曲线**：右侧实时显示 5 个奖励指标
5. **流畅交互**：所有键盘快捷键正常工作

这就是 deploy_robot-main-v1.0.0 中 ghost 功能的效果！🎉
