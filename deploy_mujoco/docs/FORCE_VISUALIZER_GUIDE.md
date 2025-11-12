# ForceVisualizer 使用指南

## 概述

`ForceVisualizer` 类提供了两种方式在 MuJoCo viewer 中渲染外力箭头：

1. **`render_force_arrow`** - 手动方式（推荐）
2. **`render_force_perturb`** - MuJoCo 原生方式（调试用）

## 方法对比

| 特性 | render_force_arrow | render_force_perturb |
|------|-------------------|---------------------|
| **渲染方式** | 手动 `mjv_connector` | MuJoCo 自动（`MjvPerturb`） |
| **箭头颜色** | ✅ 自定义（默认紫色） | ❌ 固定浅红色 |
| **性能** | ✅ 高（不需要 updateScene） | ❌ 低（每次重新 updateScene） |
| **xfrc_applied** | 🔵 不修改（需手动设置） | ⚠️ 会覆盖 |
| **多环境支持** | ✅ 优秀 | ⚠️ 需谨慎 |
| **箭头长度** | 🔵 手动缩放 | ✅ 自动计算 |
| **实现复杂度** | 🔵 需要理解 mjv_connector | ✅ 简单 |
| **与 MuJoCo 一致性** | 🔵 自定义实现 | ✅ 完全一致 |

## 使用示例

### 方法 1：render_force_arrow（推荐用于生产）

```python
from force_visualizer import ForceVisualizer
import numpy as np

# 创建可视化器（紫色箭头）
visualizer = ForceVisualizer(arrow_color=[0.8, 0.0, 0.8, 0.6])

# 在主循环中
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
force_vector = np.array([0.0, 20.0, 0.0])  # Y 方向 20N

# 1. 先施加力到 xfrc_applied
data.xfrc_applied[body_id, 0:3] = force_vector

# 2. 然后渲染箭头
visualizer.render_force_arrow(
    viewer.user_scn,  # viewer_scene
    model,
    data,
    body_id,
    force_vector
)
```

**优点：**
- ✅ 紫色箭头与红色区分开
- ✅ 不需要重新 updateScene，性能好
- ✅ 适合多环境渲染
- ✅ 完全控制箭头样式

### 方法 2：render_force_perturb（调试用）

```python
from force_visualizer import ForceVisualizer
import numpy as np

# 创建可视化器
visualizer = ForceVisualizer()

# 在渲染时
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
force_vector = np.array([0.0, 20.0, 0.0])  # Y 方向 20N

# 直接调用（会自动施加力并渲染）
visualizer.render_force_perturb(
    viewer.user_scn,  # viewer_scene
    opt,              # MjvOption
    cam,              # MjvCamera
    model,
    data,
    body_id,
    force_vector
)

# 注意：此方法会修改 data.xfrc_applied
# 如果之前已经设置过，会被覆盖
```

**优点：**
- ✅ 使用 MuJoCo 原生机制
- ✅ 箭头长度自动缩放
- ✅ 与 Ctrl+右键箭头完全一致

**缺点：**
- ❌ 箭头颜色固定为浅红色
- ❌ 会覆盖 xfrc_applied
- ❌ 性能较低（重新 updateScene）

## 在 bmwoyaw_multienv.py 中的使用

### 当前实现（使用 render_force_arrow）

```python
# 在渲染部分
if force_applicator.is_active:
    force_info = force_applicator.get_force_info()
    if force_info:
        # 在主环境渲染紫色箭头
        force_visualizer.render_force_arrow(
            viewer.user_scn,
            mujoco_model,
            dlist[render_index],
            force_info['body_id'],
            force_info['force_vector']
        )
```

### 如果使用 render_force_perturb

```python
# 需要传入更多参数
if force_applicator.is_active:
    force_info = force_applicator.get_force_info()
    if force_info:
        force_visualizer.render_force_perturb(
            viewer.user_scn,
            opt,  # 需要添加这个
            cam,  # 需要添加这个
            mujoco_model,
            dlist[render_index],
            force_info['body_id'],
            force_info['force_vector']
        )
```

## MjvPerturb 原理详解

### 输入要求

```python
pert = mujoco.MjvPerturb()
mujoco.mjv_defaultPerturb(pert)

# 必须设置
pert.select = body_id                                    # 目标 body
pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE.value  # 1=平移, 2=旋转

# 初始化（计算 scale, refpos, refquat）
mujoco.mjv_initPerturb(model, data, scene, pert)

# 设置力的方向（间接方式）
pert.localpos = np.array([dx, dy, dz])  # 局部坐标偏移

# 应用力
mujoco.mjv_applyPerturbForce(model, data, pert)
# 结果：data.xfrc_applied[body_id] 被修改
```

### 力的计算公式

**MuJoCo 内部公式：**
```
force = -localpos * scale * localmass * 73.316
```

其中：
- `localpos`: 用户设置的局部坐标偏移向量（模拟鼠标拖拽）
- `scale`: 由 `mjv_initPerturb` 自动计算的缩放因子（与 body 大小相关）
- `localmass`: body 的质量（kg）
- `73.316`: MuJoCo 内部常数（通过实验反推得到）
- 负号：localpos 和 force 方向相反

**反向计算：从目标力得到 localpos**
```python
localpos = -force_vector / (scale * localmass * 73.316)
```

### 可视化标志

要让 `mjv_updateScene` 自动渲染箭头，需要启用：

```python
opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE.value] = 1  # index 12 - 箭头
opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ.value] = 1    # index 13 - 球体+圆柱
```

然后调用：
```python
mujoco.mjv_updateScene(model, data, opt, pert, cam, category, scene)
#                                        ^^^^
#                                    传入 pert 参数
```

### 渲染结果

启用后，scene 中会自动添加：

1. **球体** (type=2, mjGEOM_SPHERE)
   - 颜色：红色 [0.9, 0.0, 0.0, 1.0]
   - 位置：body 位置
   - 用途：标记选中的 body

2. **圆柱体** (type=3, mjGEOM_CYLINDER)
   - 颜色：红色 [0.9, 0.0, 0.0, 1.0]
   - 用途：连接球体到 perturbation 点

3. **箭头** (type=100, mjGEOM_ARROW)
   - 颜色：浅红色 [1.0, 0.5, 0.5, 1.0]
   - 位置：body 位置
   - 方向：力的方向
   - 长度：与力大小成正比

## 推荐用法

### 多环境场景（当前项目）
✅ **使用 `render_force_arrow`**
- 性能好，颜色可自定义
- 不干扰 xfrc_applied 管理
- 紫色箭头易于区分

### 单环境调试
🔵 **可选 `render_force_perturb`**
- 查看 MuJoCo 原生效果
- 验证力的计算是否正确
- 学习 MuJoCo perturbation 机制

### 交互式工具
🔵 **可选 `render_force_perturb`**
- 如果要实现类似 Ctrl+右键的交互
- 需要与 MuJoCo 原生行为一致

## 测试验证

运行测试脚本：
```bash
conda activate mujoco
python test_force_perturb_visualizer.py
```

预期输出：
- 方法1（手动）：紫色箭头，type=100
- 方法2（原生）：浅红色箭头，type=100

## 参考文档

- `PERTURBATION_RENDERING_DISCOVERY.md` - MuJoCo perturbation 渲染机制详解
- `FORCE_SYSTEM_README.md` - 外力系统总体文档
- `test_perturbation.py` - MjvPerturb 基础测试
- `test_perturb_force.py` - 力公式推导测试
- `test_force_perturb_visualizer.py` - 两种方法对比测试
