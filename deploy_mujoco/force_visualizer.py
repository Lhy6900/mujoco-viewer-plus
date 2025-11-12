"""
外力可视化模块
用于在 MuJoCo viewer 中渲染外力箭头（类似 viewer 自带的红色箭头）

使用 MuJoCo 的 mjv_connector 函数创建箭头几何体。
参考：https://mujoco.readthedocs.io/en/stable/APIreference/APIglobals.html#mjv-connector
"""

import numpy as np
import mujoco


class ForceVisualizer:
    """
    外力可视化器
    在 MuJoCo viewer 中渲染紫色箭头表示施加的外力
    
    使用方法：
    1. 创建实例: visualizer = ForceVisualizer()
    2. 在渲染循环中调用: visualizer.render_force_arrow(viewer.user_scn, model, data, body_id, force_vector)
    """
    
    def __init__(self, arrow_color=None):
        """
        初始化外力可视化器
        
        Args:
            arrow_color: 箭头颜色 RGBA，默认紫色 [0.8, 0.0, 0.8, 0.6]
        """
        if arrow_color is None:
            self.arrow_color = np.array([0.8, 0.0, 0.8, 0.6], dtype=np.float32)  # 紫色半透明
        else:
            self.arrow_color = np.array(arrow_color, dtype=np.float32)
        
        # 箭头缩放因子（力大小到箭头长度的转换）
        self.force_scale = 0.02  # 1N = 0.02m 长度
        self.arrow_width = 0.02  # 箭头宽度
    
    def render_force_arrow(self, viewer_scene, model, data, body_id, force_vector, force_scale=None):
        """
        手动渲染外力（紫色 CAPSULE + 紫色 SPHERE）
        
        在所有环境中为指定 body 渲染外力的可视化表示：
        1. 紫色细圆柱连接 body 位置到力的终点
        2. 紫色球体标记力的终点
        
        Args:
            viewer_scene: mujoco.MjvScene 对象（viewer.user_scn）
            model: mujoco.MjModel 对象
            data: mujoco.MjData 对象
            body_id: body 的 ID
            force_vector: 力向量 (3,) - [fx, fy, fz]
            force_scale: 力缩放因子（1N = force_scale 米长度）
                        如果为 None，使用默认值 self.force_scale
                        Spring 模式下，会自动传入 1/k 确保箭头终点 = 引力中心
        """
        # 获取 body 的世界坐标位置
        body_pos = data.xpos[body_id].copy()
        
        # 计算力的大小
        force_magnitude = np.linalg.norm(force_vector)
        if force_magnitude < 1e-6:
            return  # 力太小，不渲染
        
        # 使用传入的 force_scale，如果没有传入则使用默认值
        current_force_scale = force_scale if force_scale is not None else self.force_scale
        
        # 计算力的方向和终点
        force_direction = force_vector / force_magnitude
        arrow_length = force_magnitude * current_force_scale
        force_end = body_pos + force_direction * arrow_length
        
        # 紫色
        purple_color = np.array([0.8, 0.0, 0.8, 0.6], dtype=np.float32)
        
        # 1. 添加细圆柱（CAPSULE）- 从 body 到力的终点
        if viewer_scene.ngeom < viewer_scene.maxgeom:
            capsule_geom = viewer_scene.geoms[viewer_scene.ngeom]
            capsule_radius = 0.015  # 细圆柱半径
            
            mujoco.mjv_initGeom(
                capsule_geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=purple_color
            )
            
            # 使用 mjv_connector 创建从 body_pos 到 force_end 的圆柱
            mujoco.mjv_connector(
                capsule_geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                capsule_radius,
                body_pos,
                force_end
            )
            capsule_geom.rgba[:] = purple_color
            viewer_scene.ngeom += 1
        
        # 2. 添加球体（SPHERE）- 标记力的终点
        if viewer_scene.ngeom < viewer_scene.maxgeom:
            sphere_geom = viewer_scene.geoms[viewer_scene.ngeom]
            sphere_radius = 0.03  # 球体半径
            
            mujoco.mjv_initGeom(
                sphere_geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([sphere_radius, sphere_radius, sphere_radius]),
                pos=force_end,  # 球体位置在力的终点
                mat=np.eye(3).flatten(),
                rgba=purple_color
            )
            viewer_scene.ngeom += 1
    
    def render_multi_env_forces(self, viewer_scene, model, data_list, force_info_list, env_origins):
        """
        渲染多个环境的外力箭头
        
        Args:
            viewer_scene: mujoco.MjvScene 对象（viewer.user_scn）
            model: mujoco.MjModel 对象
            data_list: mujoco.MjData 对象列表
            force_info_list: ForceApplicator.get_force_info() 返回的信息列表
            env_origins: 环境原点位置数组 (num_envs, 2)
        """
        for env_idx, (data, force_info) in enumerate(zip(data_list, force_info_list)):
            if force_info is None:
                continue  # 该环境没有施加外力
            
            body_id = force_info['body_id']
            force_vector = force_info['force_vector']
            
            # 渲染箭头
            self.render_force_arrow(viewer_scene, model, data, body_id, force_vector)
