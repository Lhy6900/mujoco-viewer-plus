"""Ghost renderer for reference trajectory visualization.

This module handles the construction and rendering of ghost models,
which show the reference/target trajectory for the robot.
"""

import numpy as np
import mujoco
from typing import Optional, Dict, Any
import copy

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

from .base_renderer import BaseVisualElement


class GhostRenderer(BaseVisualElement):
    """Renders a semi-transparent ghost model showing reference trajectory.
    
    The ghost is constructed from policy outputs and robot state,
    displaying where the robot should be according to the reference motion.
    """
    
    def __init__(self, model: mujoco.MjModel, config: Optional[dict] = None):
        """Initialize ghost renderer.
        
        Args:
            model: MuJoCo model (will be deep-copied for ghost)
            config: Configuration including:
                - ghost_color: RGBA color for ghost [default: [0.5, 0.7, 0.5, 0.5]]
                - enabled: Whether ghost is initially visible [default: True]
        """
        super().__init__(model, config)
        
        # Create ghost model (deep copy with custom appearance)
        self._ghost_model = copy.deepcopy(model)
        ghost_color = self.config.get("ghost_color", [0.5, 0.7, 0.5, 0.5])
        self._ghost_model.geom_rgba[:] = np.array(ghost_color, dtype=np.float32)
        
        # Auxiliary data for ghost rendering
        self._ghost_data = mujoco.MjData(self._ghost_model)
        
        # Current ghost pose (qpos)
        self._ghost_qpos: Optional[np.ndarray] = None
    
    def update(self, qpos: np.ndarray) -> None:
        """Update ghost pose.
        
        Args:
            qpos: Joint positions (shape: (nq,)) for ghost model
        """
        if qpos is None:
            self._ghost_qpos = None
            return
        
        arr = np.asarray(qpos, dtype=np.float32)
        
        # Validation
        if arr.ndim != 1:
            logger_mp.warning("GhostRenderer: expected 1D qpos, got shape %s", arr.shape)
            return
        
        if arr.shape[0] != self.model.nq:
            logger_mp.warning("GhostRenderer: qpos length %d != model.nq %d", 
                            arr.shape[0], self.model.nq)
            return
        
        self._ghost_qpos = arr.copy()
    
    def render(self, viewer_scene: Any, viewer_option: Any) -> None:
        """Render ghost into the viewer scene.
        
        Args:
            viewer_scene: MuJoCo viewer scene (mjvScene)
            viewer_option: MuJoCo viewer options (mjvOption)
        """
        if not self.enabled or self._ghost_qpos is None:
            return
        
        try:
            # Update ghost data with current pose
            self._ghost_data.qpos[:] = self._ghost_qpos
            mujoco.mj_forward(self._ghost_model, self._ghost_data)
            
            # Add ghost geometries to viewer scene
            pert = mujoco.MjvPerturb()
            catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
            
            mujoco.mjv_addGeoms(
                self._ghost_model,
                self._ghost_data,
                viewer_option,
                pert,
                catmask,
                viewer_scene,
            )
        except Exception:
            logger_mp.exception("GhostRenderer: failed to render ghost")
    
    def clear(self) -> None:
        """Clear current ghost pose."""
        self._ghost_qpos = None


def build_ghost_qpos_from_policy(
    policy_output: Dict[str, Any],
    robot: Any,
    current_qpos: np.ndarray
) -> Optional[np.ndarray]:
    """Build ghost qpos from policy output and robot interface.
    
    This is a utility function that encapsulates the logic of converting
    policy outputs (body poses, joint positions) into a full qpos vector
    suitable for ghost rendering.
    
    Args:
        policy_output: Dictionary containing policy outputs with keys:
            - 'joint_pos': Joint positions from policy
            - 'body_pos_w': Body positions in world frame
            - 'body_quat_w': Body quaternions in world frame
        robot: Robot interface with joint2dof() method
        current_qpos: Current qpos from simulation (for sizing)
    
    Returns:
        Ghost qpos array, or None if construction fails
    """
    try:
        if not policy_output or "joint_pos" not in policy_output:
            return None
        
        # Extract policy joint positions
        policy_joint_pos = np.asarray(policy_output["joint_pos"])
        if policy_joint_pos.ndim >= 2:
            policy_joint_pos = policy_joint_pos.squeeze(0)
        
        # Extract base pose from policy output
        policy_body_pos_w = np.asarray(policy_output.get("body_pos_w", None))
        policy_body_quat_w = np.asarray(policy_output.get("body_quat_w", None))
        
        if policy_body_pos_w is None or policy_body_quat_w is None:
            logger_mp.warning("build_ghost_qpos: missing body pose in policy output")
            return None
        
        # Use first body (index 0) as base pose
        base_pos_policy = policy_body_pos_w[0, 0, :]  # (3,)
        base_quat_policy = policy_body_quat_w[0, 0, :]  # (4,)
        
        # Construct full ghost qpos
        ghost_qpos = np.zeros_like(current_qpos)
        ghost_qpos[0:3] = base_pos_policy
        ghost_qpos[3:7] = base_quat_policy
        
        # Map policy joint positions to MuJoCo/dof order
        ghost_joint_pos_dof = robot.joint2dof(policy_joint_pos)
        ghost_qpos[7:7+len(ghost_joint_pos_dof)] = ghost_joint_pos_dof
        
        return ghost_qpos
        
    except Exception:
        logger_mp.exception("build_ghost_qpos: failed to construct ghost qpos")
        return None
