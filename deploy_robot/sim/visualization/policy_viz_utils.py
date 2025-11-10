"""Utilities for policy visualization in simulation.

This module provides helper functions to bridge between policy outputs
and visualization systems (ghost rendering, reward tracking, etc.).
"""

import numpy as np
from typing import Dict, Any, Optional
import logging_mp

logger_mp = logging_mp.get_logger(__name__)


def update_ghost_from_policy(simulator: Any, policy: Any, robot: Any) -> None:
    """Update ghost visualization from policy output.
    
    This function encapsulates the logic of extracting reference trajectory
    from policy output and sending it to the simulator's ghost renderer.
    
    Args:
        simulator: Simulator instance with add_ghost() method
        policy: Policy instance with output_tensors attribute
        robot: Robot interface with joint2dof() method
    """
    try:
        if not hasattr(policy, "output_tensors") or "joint_pos" not in policy.output_tensors:
            return
        
        # Extract policy's joint positions (reference trajectory)
        policy_joint_pos = np.asarray(policy.output_tensors["joint_pos"])
        if policy_joint_pos.ndim >= 2:
            policy_joint_pos = policy_joint_pos.squeeze(0)  # Remove batch dimension
        
        # Extract base pose from policy output
        policy_body_pos_w = np.asarray(policy.output_tensors.get("body_pos_w", None))
        policy_body_quat_w = np.asarray(policy.output_tensors.get("body_quat_w", None))
        
        if policy_body_pos_w is None or policy_body_quat_w is None:
            logger_mp.warning("update_ghost_from_policy: missing body pose in policy output")
            return
        
        # Use first body (index 0) as base pose
        base_pos_policy = policy_body_pos_w[0, 0, :]  # (3,)
        base_quat_policy = policy_body_quat_w[0, 0, :]  # (4,)
        
        # Construct full ghost qpos
        current_qpos = simulator.data.qpos.copy()
        ghost_qpos = np.zeros_like(current_qpos)
        ghost_qpos[0:3] = base_pos_policy
        ghost_qpos[3:7] = base_quat_policy
        
        # Map policy joint positions to MuJoCo/dof order using robot.joint2dof()
        # This converts from policy/joint order to hardware/dof order
        ghost_joint_pos_dof = robot.joint2dof(policy_joint_pos)
        
        # Assign to ghost_qpos
        ghost_qpos[7:7+len(ghost_joint_pos_dof)] = ghost_joint_pos_dof
        
        # Send to viewer for rendering
        if hasattr(simulator, "add_ghost"):
            simulator.add_ghost(ghost_qpos)
            
    except Exception:
        logger_mp.exception("update_ghost_from_policy: failed to update ghost")


def compute_tracking_rewards(policy: Any, robot: Any) -> Dict[str, float]:
    """Compute tracking error rewards from policy output and robot state.
    
    Args:
        policy: Policy instance with output_tensors
        robot: Robot interface with joint_pos, joint_vel attributes
    
    Returns:
        Dictionary of reward metrics
    """
    rewards = {
        "smoothness": 0.0,
        "pos_tracking_global": 0.0,
        "pos_tracking_local": 0.0,
        "quat_tracking_global": 0.0,
        "quat_tracking_local": 0.0,
    }
    
    try:
        current_joint_pos = robot.joint_pos
        current_joint_vel = robot.joint_vel
        
        # Smoothness: measure joint velocity changes (requires state from previous frame)
        # This is a placeholder - caller should track last_joint_vel externally
        if hasattr(robot, '_last_joint_vel'):
            joint_vel_change = np.linalg.norm(current_joint_vel - robot._last_joint_vel)
            rewards["smoothness"] = -joint_vel_change
        
        # Position tracking errors (if reference available)
        if hasattr(policy, "output_tensors") and "joint_pos" in policy.output_tensors:
            ref_joint_pos = policy.output_tensors["joint_pos"]
            ref_joint_pos = ref_joint_pos.squeeze(0) if hasattr(ref_joint_pos, 'squeeze') else ref_joint_pos
            ref_joint_pos = np.asarray(ref_joint_pos)
            
            if ref_joint_pos.shape == current_joint_pos.shape:
                pos_error = np.linalg.norm(current_joint_pos - ref_joint_pos)
                rewards["pos_tracking_global"] = -pos_error
                rewards["pos_tracking_local"] = -np.mean(np.abs(current_joint_pos - ref_joint_pos))
        
        # Orientation tracking (simplified - placeholder for full implementation)
        # Would need base_quat from policy and robot
        rewards["quat_tracking_global"] = 0.0
        rewards["quat_tracking_local"] = 0.0
        
    except Exception:
        logger_mp.exception("compute_tracking_rewards: failed to compute rewards")
    
    return rewards
