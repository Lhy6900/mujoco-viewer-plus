"""
Ghost Renderer for Reference Trajectory Visualization
参考 deploy_robot-main-v1.0.0/deploy_robot/sim/viewer_plus/viewer_plus.py
显示半透明绿色的参考轨迹 ghost
"""

import mujoco
import numpy as np
import copy
from typing import Optional


class GhostRenderer:
    """渲染参考轨迹的半透明 ghost 模型"""
    
    def __init__(self, model: mujoco.MjModel):
        """
        初始化 ghost 渲染器
        
        Args:
            model: 主 MuJoCo 模型
        """
        # 创建 ghost 模型（深拷贝主模型并设置半透明绿色）
        self._ghost_model = copy.deepcopy(model)
        
        # 设置 ghost 颜色：半透明绿色 [R, G, B, A]
        # 参考 deploy_robot-main-v1.0.0 的配置：[0.5, 0.7, 0.5, 0.5]
        ghost_color = np.array([0.5, 0.7, 0.5, 0.5], dtype=np.float32)
        self._ghost_model.geom_rgba[:] = ghost_color
        
        # 创建用于 ghost 渲染的 MjData
        self._ghost_data = mujoco.MjData(self._ghost_model)
        
        # 渲染选项
        self._vopt = mujoco.MjvOption()
        self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # 启用透明渲染
        
        self._pert = mujoco.MjvPerturb()
        self._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
        
        # 当前 ghost 的 qpos（由外部设置）
        self._ghost_qpos: Optional[np.ndarray] = None
        
        print("[GhostRenderer] 初始化完成，颜色: 半透明绿色 RGBA [0.5, 0.7, 0.5, 0.5]")
    
    def set_ghost_qpos(self, qpos: np.ndarray) -> None:
        """
        设置 ghost 的姿态（qpos）
        
        Args:
            qpos: 完整的 qpos 数组 (nq,)，包含 root 位置/姿态和关节角度
        """
        if qpos is None:
            self._ghost_qpos = None
            return
        
        arr = np.asarray(qpos, dtype=np.float32)
        if arr.ndim != 1:
            print(f"[警告] set_ghost_qpos 需要 1D 数组，得到形状 {arr.shape}")
            return
        
        if arr.shape[0] != self._ghost_model.nq:
            print(f"[警告] qpos 长度 {arr.shape[0]} != model.nq {self._ghost_model.nq}")
            return
        
        self._ghost_qpos = arr.copy()
    
    def render_ghost(self, viewer_user_scn) -> None:
        """
        渲染 ghost 到 viewer 的用户场景
        
        Args:
            viewer_user_scn: viewer.user_scn 对象
        """
        if self._ghost_qpos is None:
            return  # 没有设置 ghost 姿态，跳过渲染
        
        try:
            # 设置 ghost_data 的姿态
            self._ghost_data.qpos[:] = self._ghost_qpos
            
            # 执行前向动力学（更新 ghost 的位置和速度）
            mujoco.mj_forward(self._ghost_model, self._ghost_data)
            
            # 将 ghost 的几何体添加到 viewer 的用户场景
            mujoco.mjv_addGeoms(
                self._ghost_model,
                self._ghost_data,
                self._vopt,
                self._pert,
                self._catmask,
                viewer_user_scn
            )
        except Exception as e:
            print(f"[警告] Ghost 渲染失败: {e}")
    
    def construct_ghost_qpos(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        joint_pos_dof_order: np.ndarray,
        current_qpos: np.ndarray
    ) -> np.ndarray:
        """
        构造完整的 ghost qpos
        
        Args:
            base_pos: root 位置 (3,)
            base_quat: root 四元数 (4,)
            joint_pos_dof_order: 关节角度（dof 顺序）(num_joints,)
            current_qpos: 当前仿真的 qpos（用于获取正确的大小）
            
        Returns:
            完整的 ghost qpos 数组
        """
        ghost_qpos = np.zeros_like(current_qpos)
        ghost_qpos[0:3] = base_pos      # XYZ 位置
        ghost_qpos[3:7] = base_quat     # 四元数姿态 (w, x, y, z)
        ghost_qpos[7:7+len(joint_pos_dof_order)] = joint_pos_dof_order  # 关节角度
        return ghost_qpos


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("GhostRenderer 测试")
    print("=" * 60)
    
    # 这里只是演示 API，实际使用需要在主仿真循环中
    print("\n✅ GhostRenderer 类定义完成")
    print("\n使用方法：")
    print("1. ghost_renderer = GhostRenderer(model)")
    print("2. ghost_qpos = ghost_renderer.construct_ghost_qpos(base_pos, base_quat, joint_pos, current_qpos)")
    print("3. ghost_renderer.set_ghost_qpos(ghost_qpos)")
    print("4. ghost_renderer.render_ghost(viewer.user_scn)")
