# 多环境Viewer交互说明

## 问题回答总结

### 1. mjlab中的环境切换机制

**是的，mjlab可以通过快捷键切换主环境！**

#### 快捷键：
- **`,` (逗号键)** - 切换到上一个环境 (`PREV_ENV`)
- **`.` (句号键)** - 切换到下一个环境 (`NEXT_ENV`)

#### 工作原理：
mjlab在 `native.py` 中实现了环境切换：

```python
# 按键处理
elif key == KEY_COMMA:
    self.request_action("PREV_ENV")
elif key == KEY_PERIOD:
    self.request_action("NEXT_ENV")

# 环境切换逻辑
if action == ViewerAction.PREV_ENV and self.env.unwrapped.num_envs > 1:
    self.env_idx = (self.env_idx - 1) % self.env.unwrapped.num_envs
    self.log(f"[INFO] Switched to environment {self.env_idx}", VerbosityLevel.INFO)
```

#### 环境渲染机制：
- **主环境** (`self.env_idx`)：直接绑定到 `viewer` 的主数据 `mjd`，**可以交互**
- **其他环境**：通过 `mjv_addGeoms()` 渲染为附加几何体，**只能观看，不能交互**

```python
# 主环境的数据同步到viewer
self.mjd.qpos[:] = sim_data.qpos[self.env_idx].cpu().numpy()
self.mjd.qvel[:] = sim_data.qvel[self.env_idx].cpu().numpy()
mujoco.mj_forward(self.mjm, self.mjd)

# 其他环境作为几何体添加
for i in range(self.env.unwrapped.num_envs):
    if i == self.env_idx:
        continue
    self.vd.qpos[:] = sim_data.qpos[i].cpu().numpy()
    self.vd.qvel[:] = sim_data.qvel[i].cpu().numpy()
    mujoco.mj_forward(self.mjm, self.vd)
    mujoco.mjv_addGeoms(self.mjm, self.vd, self.vopt, self.pert, 
                        self.catmask, v.user_scn)
```

### 2. Viewer交互机制

**是的，默认情况下viewer的交互（拖拽施力）只能作用于主环境！**

#### 原理：
```python
def sync_viewer_to_env(self):
    """将viewer的扰动力复制到环境"""
    # 只从主环境的mjd中提取扰动力
    xfrc = torch.as_tensor(
        self.mjd.xfrc_applied, dtype=torch.float, device=self.env.device
    )
    # 只应用到当前选中的环境 (env_idx)
    self.env.unwrapped.sim.data.xfrc_applied[self.env_idx] = xfrc
```

#### 为什么其他环境不能交互？
- Viewer的鼠标操作直接作用于 `mjd`（主环境的MjData）
- 其他环境只是通过 `mjv_addGeoms` 渲染的"只读快照"
- 扰动力 `xfrc_applied` 只从 `mjd` 中读取

---

## 你的代码修改说明

我已经为你的 `bmwoyaw_multienv.py` 添加了类似mjlab的环境切换功能：

### 修改1: 添加环境索引和键盘回调

```python
# 当前选择的主环境索引（用于交互）
current_env_idx = Value('i', 0)  # 使用共享内存变量

# 定义快捷键切换环境的回调函数
def key_callback(key):
    # 逗号键 - 切换到上一个环境
    if key == 44:  # ord(',')
        current_env_idx.value = (current_env_idx.value - 1) % num_envs
        print(f"[切换环境] 当前主环境: {current_env_idx.value}")
    # 句号键 - 切换到下一个环境
    elif key == 46:  # ord('.')
        current_env_idx.value = (current_env_idx.value + 1) % num_envs
        print(f"[切换环境] 当前主环境: {current_env_idx.value}")

# 启动viewer时传入回调
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
```

### 修改2: 主循环中同步选中环境到viewer

```python
# 先执行所有环境的仿真步
for i in range(num_envs):
    mujoco.mj_step(m, dlist[i])
    tau_i = pd_control(...)
    dlist[i].ctrl[:] = tau_i

# 将当前选中的环境数据同步到viewer的主数据d
idx = current_env_idx.value
d.qpos[:] = dlist[idx].qpos[:]
d.qvel[:] = dlist[idx].qvel[:]
d.ctrl[:] = dlist[idx].ctrl[:]
mujoco.mj_forward(m, d)

# 如果有外力扰动（通过viewer交互施加），将其应用回选中的环境
if np.any(d.xfrc_applied != 0):
    dlist[idx].xfrc_applied[:] = d.xfrc_applied[:]
```

### 修改3: 渲染时跳过主环境

```python
# 清空user_scn中的geoms
viewer.user_scn.ngeom = 0

# 渲染所有其他环境（除了主环境）
for i in range(num_envs):
    if i == current_env_idx.value:
        continue  # 跳过当前主环境（已经在viewer的主场景中显示）
    mujoco.mjv_addGeoms(m, dlist[i], vopt, pert, catmask, viewer.user_scn)

viewer.sync()
```

---

## 使用方法

### 启动程序后：

1. **查看所有环境**：所有4个环境的机器人都会显示在窗口中
2. **切换主环境**：
   - 按 `,` (逗号) 键 - 切换到上一个环境
   - 按 `.` (句号) 键 - 切换到下一个环境
   - 终端会打印：`[切换环境] 当前主环境: X`

3. **交互操作**：
   - 按住 **Ctrl + 鼠标左键** 可以拖拽**当前选中的环境**
   - 施加的力会作用于 `dlist[current_env_idx]`
   - 其他环境不受影响

### 注意事项：

- **主环境**（current_env_idx指向的）：高亮显示，可以交互
- **其他环境**：通过 `mjv_addGeoms` 渲染，只能观看
- 切换环境后，viewer会立即显示新选中环境的状态
- 扰动力会正确应用到选中的环境

---

## 关键技术点

### Ghost vs 环境切换

- **Ghost**：半透明的参考姿态，用于显示目标位置（如运动捕捉数据）
- **环境切换**：切换哪个环境作为"主环境"进行交互和详细观察

它们是两个独立的功能，不要混淆！

### 为什么需要 `current_env_idx = Value('i', 0)`？

因为 `key_callback` 在不同的线程中运行，需要线程安全的共享变量。

### mjv_addGeoms 的作用

`mjv_addGeoms` 将模型的几何体添加到场景中，用于：
- 渲染多个环境
- 渲染ghost（参考姿态）
- 添加调试可视化

---

## 参考mjlab代码位置

- 环境切换实现：`mjlab/src/mjlab/viewer/native.py` 第250-280行
- 扰动力同步：`mjlab/src/mjlab/viewer/native.py` 第207-215行
- 多环境渲染：`mjlab/src/mjlab/viewer/native.py` 第188-202行
