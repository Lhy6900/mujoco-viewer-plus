"""Trajectory manager for ghost robot visualization.

This module implements trajectory playback similar to mjlab's MotionCommand,
tracking time steps and providing current reference poses for ghost rendering.
"""

import numpy as np
from typing import Optional, Dict, Any
import logging_mp

logger_mp = logging_mp.get_logger(__name__)


class TrajectoryManager:
    """Manages reference trajectory playback for ghost visualization.
    
    This class mimics mjlab's MotionCommand behavior:
    - Tracks current time_step in the trajectory
    - Provides body_pos_w, body_quat_w, joint_pos at current time_step
    - Handles env_origins offset for multi-environment scenarios
    - Automatically loops when reaching trajectory end
    
    In mjlab, the trajectory is loaded from .npz files containing:
    - body_pos_w: (T, num_bodies, 3) - world-frame body positions
    - body_quat_w: (T, num_bodies, 4) - world-frame body quaternions
    - joint_pos: (T, num_joints) - joint positions
    
    For policy-based deployment, these come from policy outputs instead.
    """
    
    def __init__(
        self,
        num_envs: int = 1,
        env_origins: Optional[np.ndarray] = None,
    ):
        """Initialize trajectory manager.
        
        Args:
            num_envs: Number of parallel environments (typically 1 for deployment)
            env_origins: Environment origin offsets, shape (num_envs, 3).
                        Defaults to zeros (all envs at world origin).
        """
        self.num_envs = num_envs
        
        # Environment origins offset (following mjlab's scene.env_origins)
        if env_origins is None:
            self.env_origins = np.zeros((num_envs, 3), dtype=np.float32)
        else:
            self.env_origins = np.asarray(env_origins, dtype=np.float32)
            assert self.env_origins.shape == (num_envs, 3), \
                f"env_origins must be shape ({num_envs}, 3), got {self.env_origins.shape}"
        
        # Current time step in trajectory (increments each update)
        self.time_steps = np.zeros(num_envs, dtype=np.int64)
        
        # Trajectory data storage (set via update_trajectory)
        self._body_pos_w_raw: Optional[np.ndarray] = None  # (T, num_bodies, 3)
        self._body_quat_w_raw: Optional[np.ndarray] = None  # (T, num_bodies, 4)
        self._joint_pos_raw: Optional[np.ndarray] = None  # (T, num_joints)
        
        self._trajectory_length = 0
        
        logger_mp.info(
            "TrajectoryManager initialized: num_envs=%d, env_origins=%s",
            num_envs, self.env_origins.tolist()
        )
    
    def update_trajectory(
        self,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        joint_pos: np.ndarray,
    ) -> None:
        """Update the reference trajectory data.
        
        This replaces the entire trajectory with new data.
        Call this when loading a new motion or when policy outputs change.
        
        Args:
            body_pos_w: Body positions in world frame, shape (T, num_bodies, 3)
            body_quat_w: Body quaternions in world frame, shape (T, num_bodies, 4)
            joint_pos: Joint positions, shape (T, num_joints)
        """
        self._body_pos_w_raw = np.asarray(body_pos_w, dtype=np.float32)
        self._body_quat_w_raw = np.asarray(body_quat_w, dtype=np.float32)
        self._joint_pos_raw = np.asarray(joint_pos, dtype=np.float32)
        
        # Validate shapes
        T = self._body_pos_w_raw.shape[0]
        assert self._body_quat_w_raw.shape[0] == T, "body_quat_w must have same T dimension"
        assert self._joint_pos_raw.shape[0] == T, "joint_pos must have same T dimension"
        
        self._trajectory_length = T
        
        # Reset time steps to start of trajectory
        self.time_steps[:] = 0
        
        logger_mp.info(
            "Trajectory updated: T=%d, num_bodies=%d, num_joints=%d",
            T, self._body_pos_w_raw.shape[1], self._joint_pos_raw.shape[1]
        )
        
        # DEBUG: Print first frame data to verify trajectory content and quaternion format
        logger_mp.info("=" * 80)
        logger_mp.info("FIRST FRAME TRAJECTORY DATA (t=0):")
        logger_mp.info("-" * 80)
        logger_mp.info("Base (body[0]) position [x,y,z]: %s", self._body_pos_w_raw[0, 0])
        logger_mp.info("Base (body[0]) quaternion (AS STORED): %s", self._body_quat_w_raw[0, 0])
        logger_mp.info("  NOTE: MuJoCo expects [w,x,y,z] format")
        logger_mp.info("  If your data is [x,y,z,w], quaternion needs reordering!")
        logger_mp.info("Joint positions (first 10): %s", self._joint_pos_raw[0, :10])
        logger_mp.info("Joint positions (remaining): %s", self._joint_pos_raw[0, 10:])
        logger_mp.info("-" * 80)
        logger_mp.info("QUATERNION FORMAT CHECK:")
        quat_first_frame = self._body_quat_w_raw[0, 0]
        logger_mp.info("  quat[0] = %.4f  (should be ~1.0 if [w,x,y,z], or ~0.0 if [x,y,z,w])", quat_first_frame[0])
        logger_mp.info("  quat[1] = %.4f", quat_first_frame[1])
        logger_mp.info("  quat[2] = %.4f", quat_first_frame[2])
        logger_mp.info("  quat[3] = %.4f  (should be ~1.0 if [x,y,z,w], or ~0.0 if [w,x,y,z])", quat_first_frame[3])
        quat_norm = np.linalg.norm(quat_first_frame)
        logger_mp.info("  ||quat|| = %.6f  (should be ~1.0 for valid quaternion)", quat_norm)
        
        # Heuristic detection of quaternion format
        if abs(quat_first_frame[3]) > abs(quat_first_frame[0]) and abs(quat_first_frame[3]) > 0.9:
            logger_mp.warning("⚠️  DETECTED: Quaternion likely in [x,y,z,w] format (quat[3]=%.3f is large)", quat_first_frame[3])
            logger_mp.warning("⚠️  MuJoCo needs [w,x,y,z] format. Consider reordering!")
        elif abs(quat_first_frame[0]) > 0.9:
            logger_mp.info("✓ Quaternion appears to be in correct [w,x,y,z] format (quat[0]=%.3f)", quat_first_frame[0])
        else:
            logger_mp.warning("⚠️  Quaternion format unclear. Please verify manually!")
        logger_mp.info("=" * 80)
    
    def convert_quaternion_format(self, from_format: str = "xyzw") -> None:
        """Convert quaternion format from [x,y,z,w] to [w,x,y,z] or vice versa.
        
        MuJoCo uses [w,x,y,z] format, but many libraries (PyTorch, numpy-quaternion, etc.)
        use [x,y,z,w] format. This method reorders the quaternion components.
        
        Args:
            from_format: Current format of stored quaternions
                - "xyzw": Convert from [x,y,z,w] to [w,x,y,z] (MuJoCo format)
                - "wxyz": Convert from [w,x,y,z] to [x,y,z,w] (NOT typically needed)
        """
        if self._body_quat_w_raw is None:
            logger_mp.warning("No trajectory loaded, cannot convert quaternion format")
            return
        
        if from_format == "xyzw":
            # Input: [x, y, z, w] -> Output: [w, x, y, z]
            logger_mp.info("Converting quaternions from [x,y,z,w] to [w,x,y,z] (MuJoCo format)...")
            original = self._body_quat_w_raw.copy()
            self._body_quat_w_raw = np.concatenate([
                original[..., 3:4],  # w component (was last)
                original[..., 0:3],  # x, y, z components (were first)
            ], axis=-1)
            logger_mp.info("Quaternion conversion complete. Example before/after:")
            logger_mp.info("  Before: %s", original[0, 0])
            logger_mp.info("  After:  %s", self._body_quat_w_raw[0, 0])
        elif from_format == "wxyz":
            # Input: [w, x, y, z] -> Output: [x, y, z, w]
            logger_mp.info("Converting quaternions from [w,x,y,z] to [x,y,z,w]...")
            original = self._body_quat_w_raw.copy()
            self._body_quat_w_raw = np.concatenate([
                original[..., 1:4],  # x, y, z components (were after w)
                original[..., 0:1],  # w component (was first)
            ], axis=-1)
            logger_mp.info("Quaternion conversion complete. Example before/after:")
            logger_mp.info("  Before: %s", original[0, 0])
            logger_mp.info("  After:  %s", self._body_quat_w_raw[0, 0])
        else:
            logger_mp.error("Unknown quaternion format: %s. Use 'xyzw' or 'wxyz'", from_format)
    
    def step(self) -> None:
        """Advance time step by 1, wrapping at trajectory end.
        
        This mimics mjlab's _update_command behavior:
        - Increment time_steps by 1
        - If reaching trajectory end, wrap to 0 (loop playback)
        
        Call this once per simulation step.
        """
        if self._trajectory_length == 0:
            return  # No trajectory loaded
        
        self.time_steps += 1
        
        # Wrap around at trajectory end (following mjlab's auto-resample)
        env_ids_to_wrap = np.where(self.time_steps >= self._trajectory_length)[0]
        if len(env_ids_to_wrap) > 0:
            self.time_steps[env_ids_to_wrap] = 0
            logger_mp.debug("Trajectory wrapped for envs: %s", env_ids_to_wrap)
    
    @property
    def body_pos_w(self) -> Optional[np.ndarray]:
        """Get current body positions in world frame with env_origins offset.
        
        This follows mjlab's MotionCommand.body_pos_w property:
            return motion.body_pos_w[time_steps] + env_origins[:, None, :]
        
        Returns:
            Array of shape (num_envs, num_bodies, 3), or None if no trajectory
        """
        if self._body_pos_w_raw is None:
            return None
        
        # Index trajectory at current time_steps
        positions = self._body_pos_w_raw[self.time_steps]  # (num_envs, num_bodies, 3)
        
        # Add env_origins offset (broadcast over num_bodies dimension)
        positions = positions + self.env_origins[:, None, :]
        
        return positions
    
    @property
    def body_quat_w(self) -> Optional[np.ndarray]:
        """Get current body quaternions in world frame.
        
        Returns:
            Array of shape (num_envs, num_bodies, 4), or None if no trajectory
        """
        if self._body_quat_w_raw is None:
            return None
        
        return self._body_quat_w_raw[self.time_steps]  # (num_envs, num_bodies, 4)
    
    @property
    def joint_pos(self) -> Optional[np.ndarray]:
        """Get current joint positions.
        
        Returns:
            Array of shape (num_envs, num_joints), or None if no trajectory
        """
        if self._joint_pos_raw is None:
            return None
        
        return self._joint_pos_raw[self.time_steps]  # (num_envs, num_joints)
    
    def get_ghost_qpos(self, env_idx: int = 0, nq: Optional[int] = None) -> Optional[np.ndarray]:
        """Construct full qpos array for ghost visualization.
        
        This follows mjlab's ghost rendering pattern:
            qpos[free_joint_q_adr[0:3]] = body_pos_w[env_idx, 0]
            qpos[free_joint_q_adr[3:7]] = body_quat_w[env_idx, 0]
            qpos[joint_q_adr] = joint_pos[env_idx]
        
        Args:
            env_idx: Which environment to get qpos for
            nq: Total qpos size (if None, inferred from data)
        
        Returns:
            1D qpos array of length nq, or None if no trajectory
        """
        if self.body_pos_w is None or self.body_quat_w is None or self.joint_pos is None:
            return None
        
        # Get current state at env_idx
        base_pos = self.body_pos_w[env_idx, 0]  # First body (root/base)
        base_quat = self.body_quat_w[env_idx, 0]  # First body orientation
        joints = self.joint_pos[env_idx]
        
        # Infer nq if not provided
        if nq is None:
            nq = 7 + len(joints)  # base (3 pos + 4 quat) + joints
        
        # Construct qpos (assuming standard layout: pos, quat, joints)
        qpos = np.zeros(nq, dtype=np.float32)
        qpos[0:3] = base_pos
        qpos[3:7] = base_quat
        qpos[7:7+len(joints)] = joints
        
        return qpos
    
    def reset(self) -> None:
        """Reset time steps to 0 (restart trajectory playback)."""
        self.time_steps[:] = 0
        logger_mp.info("Trajectory playback reset to t=0")
