import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
import time
from enum import Enum

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class ReferenceFrame(Enum):
    """Reference frame options."""
    WORLD = "world"  # World/global frame
    BODY = "body"    # Robot's local/body frame


@dataclass
class JointConfig:
    """Configuration for joint remapping and limits."""
    joints_order: Optional[List[str]] = None  # Custom order of joints
    joint_limits_min: Optional[List[float]] = None  # Min joint limits
    joint_limits_max: Optional[List[float]] = None  # Max joint limits
    default_positions: Optional[List[float]] = None  # Default joint positions
    torque_limits: Optional[List[float]] = None  # Torque limits


@dataclass
class HardwareSimConfig:
    """Configuration for hardware simulation features."""
    lag_steps: int = 0  # Number of steps to lag commands
    add_noise: bool = False  # Add sensor noise
    noise_stddev: Dict[str, float] = field(default_factory=lambda: {
        "position": 0.0001,
        "velocity": 0.001,
        "acceleration": 0.01,
        "orientation": 0.0001
    })


@dataclass
class MujocoSimulatorConfig:
    """Configuration for MuJoCo simulator."""
    xml_path: str
    robot_type: str = "generic"  # Identifier for robot type (go1, a1, etc.)
    headless: bool = True
    decimation: int = 4  # Run N iterations of simulation every step
    num_envs: int = 1  # Number of parallel environments
    dt: float = 0.005  # Simulation timestep
    root_body_name: Optional[str] = None  # Optional explicit root body name
    joint_config: JointConfig = field(default_factory=JointConfig)  # May be None for standard robots
    hardware_sim: HardwareSimConfig = field(default_factory=HardwareSimConfig)
    # Viewer options
    viewer_type: str = "native"  # "native" or "viewer_plus"
    viewer_plus_config: Optional[dict] = None


class MultiMujocoSimulator:
    """MuJoCo simulator that is completely independent from ROS."""
    
    def __init__(self, config: Union[MujocoSimulatorConfig, Dict[str, Any]]):
        """Initialize MuJoCo simulator with configuration."""
        # Convert dict config to proper config object
        if isinstance(config, dict):
            # Handle joint config
            if "joint_config" in config and isinstance(config["joint_config"], dict):
                config["joint_config"] = JointConfig(**config["joint_config"])
                        
            # Handle hardware_sim
            if "hardware_sim" in config and isinstance(config["hardware_sim"], dict):
                config["hardware_sim"] = HardwareSimConfig(**config["hardware_sim"])
            
            self.cfg = MujocoSimulatorConfig(**config)
        elif isinstance(config, MujocoSimulatorConfig):
            self.cfg = config
        else:
            raise ValueError("Invalid configuration type for MujocoSimulator")
            
        # Initialize MuJoCo
        self.model = mujoco.MjModel.from_xml_path(Path(self.cfg.xml_path).as_posix())
        self.model.opt.timestep = self.cfg.dt
        self.data = mujoco.MjData(self.model)
        if self.cfg.num_envs > 1:
            self.datalist = [mujoco.MjData(self.model) for _ in range(self.cfg.num_envs)]

        # Viewer setup: support native viewer or viewer_plus
        self.viewer = None
        if not self.cfg.headless:
            if getattr(self.cfg, "viewer_type", "native") == "viewer_plus":
                try:
                    from deploy_robot.sim.viewer_plus.viewer_plus import ViewerPlus

                    vcfg = None
                    if self.cfg.viewer_plus_config is not None and isinstance(self.cfg.viewer_plus_config, dict):
                        vcfg = self.cfg.viewer_plus_config
                    self.viewer = ViewerPlus(self.model, self.data, config=vcfg, num_envs=self.cfg.num_envs)
                except Exception:
                    # Fallback to native viewer on any error
                    logger_mp.exception("Failed to initialize ViewerPlus, falling back to native viewer")
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Get all joints from model (excluding world joint)
        self._joint_names_orig = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
            for i in range(self.model.njnt)
        ][1:]  # Skip the first joint (world joint)
        
        # Handle joint mapping based on config
        self._setup_joint_mapping()
        
        # Find the root body index
        self._root_body_id = self._find_root_body()
        self.root_body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self._root_body_id)
        
        # Initialize state variables
        self._world_to_body_rot = np.eye(3)
        self._pos_world = None
        self._quat_world = None
        self._lin_vel_world = None
        self._ang_vel_body = None
        self._ang_vel_world = None
        self._lin_vel_body = None
        self._lin_acc_world = None
        self._dof_pos = None
        self._tar_dof_pos = None
        self._dof_vel = None
        self._dof_trq = np.zeros(len(self._joint_names_orig))
        
        # Setup control lag buffer if needed
        self._setup_lag_buffer()
        
        # Set initial joint positions if specified
        self._set_initial_positions()
        
        # Initialize state
        self.update_state()
        
        # For multi-env: initialize per-environment lag buffers
        if self.cfg.num_envs > 1:
            self.lag_buffer_list = []
            num_dofs = len(self._joint_names)
            neutral_cmd = {
                "q": np.zeros(num_dofs),
                "dq": np.zeros(num_dofs),
                "trq": np.zeros(num_dofs),
                "kp": np.zeros(num_dofs),
                "kd": np.zeros(num_dofs),
            }
            
            for i in range(self.cfg.num_envs):
                if self.cfg.hardware_sim and self.cfg.hardware_sim.lag_steps > 0:
                    self.lag_buffer_list.append([dict(neutral_cmd) for _ in range(self.cfg.hardware_sim.lag_steps + 1)])
                else:
                    # Even without lag, we need at least one buffer entry
                    self.lag_buffer_list.append([dict(neutral_cmd)])
    
    def _setup_joint_mapping(self):
        """Set up joint mapping based on configuration."""
        if not self.cfg.joint_config or not self.cfg.joint_config.joints_order:
            # No custom joint order - use default order
            self._joint_mapping = list(range(len(self._joint_names_orig)))
            self._joint_names = self._joint_names_orig.copy()
            return
            
        # Use custom joint order
        self._joint_names = self.cfg.joint_config.joints_order
        self._joint_mapping = []
        
        for joint_name in self._joint_names:
            if joint_name in self._joint_names_orig:
                self._joint_mapping.append(self._joint_names_orig.index(joint_name))
            else:
                raise ValueError(f"Joint {joint_name} not found in MuJoCo model")
    
    def _setup_lag_buffer(self):
        """Set up control lag buffer."""
        if not self.cfg.hardware_sim or self.cfg.hardware_sim.lag_steps <= 0:
            self.lag_buffer = None
            return
            
        # Initialize lag buffer with neutral commands
        num_dofs = len(self._joint_names)
        neutral_cmd = {
            "q": np.zeros(num_dofs),
            "dq": np.zeros(num_dofs),
            "trq": np.zeros(num_dofs),
            "kp": np.zeros(num_dofs),
            "kd": np.zeros(num_dofs),
        }
        self.lag_buffer = [dict(neutral_cmd) for _ in range(self.cfg.hardware_sim.lag_steps + 1)]
    
    def _set_initial_positions(self):
        """Set initial joint positions if specified in config."""
        if not self.cfg.joint_config or not self.cfg.joint_config.default_positions:
            return
            
        default_pos = self.cfg.joint_config.default_positions
        for mapped_idx, orig_idx in enumerate(self._joint_mapping):
            if mapped_idx < len(default_pos):
                self.data.qpos[7 + orig_idx] = default_pos[mapped_idx]
                
        mujoco.mj_forward(self.model, self.data)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
    
    def _find_root_body(self):
        """Find the root body of the robot."""
        # Check explicit configuration first
        if self.cfg.root_body_name:
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.root_body_name)
            except Exception:
                pass
        
        # Common root body names by robot type
        common_names = []
        
        # Add robot-specific names
        if self.cfg.robot_type.lower() in ["go1", "a1", "quadruped"]:
            common_names.extend(["trunk", "base", "chassis"])
        elif self.cfg.robot_type.lower() in ["g1"]:
            common_names.extend(["pelvis", "torso", "trunk"])
        
        # Add general fallbacks
        common_names.extend(["root", "base_link", "body", "main"])
        
        # Try each name
        for name in common_names:
            try:
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            except:
                continue
        
        # Default to first body after world
        return 1 if self.model.nbody > 1 else 0

    def update_state(self):
        """Update internal state from MuJoCo data."""
        # Get base state
        self._pos_world = self.data.qpos[0:3].copy()
        self._quat_world = self.data.qpos[3:7].copy()      # [w,x,y,z] format
        self._lin_vel_world = self.data.qvel[0:3].copy()   # World frame
        self._ang_vel_body = self.data.qvel[3:6].copy()    # Body frame
        
        # Update rotation matrix from quaternion
        self._world_to_body_rot = self._quaternion_to_rotation_matrix(self._quat_world)
        
        # Transform velocities between frames
        self._lin_vel_body = self._world_to_body_rot.T @ self._lin_vel_world
        self._ang_vel_world = self._world_to_body_rot @ self._ang_vel_body
        
        # Linear acceleration
        self._lin_acc_world = self.data.qacc[0:3].copy() if hasattr(self.data, 'qacc') else np.zeros(3)
        
        # Joint states with remapping
        dof_pos_all = self.data.qpos[7:].copy()
        dof_vel_all = self.data.qvel[6:].copy()
        
        # Apply remapping
        self._dof_pos = np.array([dof_pos_all[i] for i in self._joint_mapping])
        self._dof_vel = np.array([dof_vel_all[i] for i in self._joint_mapping])
        
        # Note: _dof_trq is now already set in compute_torque()
        # so we don't need to update it here
        
        # Apply sensor noise if configured
        if self.cfg.hardware_sim and self.cfg.hardware_sim.add_noise:
            self._apply_sensor_noise()
    
    def _apply_sensor_noise(self):
        """Apply configured sensor noise to measurements."""
        noise_cfg = self.cfg.hardware_sim.noise_stddev
        self._dof_pos += np.random.normal(0, noise_cfg["position"], self._dof_pos.shape)
        self._dof_vel += np.random.normal(0, noise_cfg["velocity"], self._dof_vel.shape)
        
        if self._dof_trq is not None:
            self._dof_trq += np.random.normal(0, noise_cfg["velocity"], self._dof_trq.shape)
            
        self._lin_vel_world += np.random.normal(0, noise_cfg["velocity"], self._lin_vel_world.shape)
        self._ang_vel_body += np.random.normal(0, noise_cfg["velocity"], self._ang_vel_body.shape)
        self._lin_acc_world += np.random.normal(0, noise_cfg["acceleration"], self._lin_acc_world.shape)
        
        # Recalculate transformed quantities
        self._lin_vel_body = self._world_to_body_rot.T @ self._lin_vel_world
        self._ang_vel_world = self._world_to_body_rot @ self._ang_vel_body
        
        # Add small noise to orientation
        if noise_cfg["orientation"] > 0:
            axis_angle = self._quaternion_to_axis_angle(self._quat_world)
            axis_angle += np.random.normal(0, noise_cfg["orientation"], axis_angle.shape)
            self._quat_world = self._axis_angle_to_quaternion(axis_angle)
    
    def step(self):
        """Step the simulation, with decimation."""
        # Run multiple physics steps according to decimation factor
        for _ in range(self.cfg.decimation):
            # Compute and apply control torques
            self.data.ctrl[:] = self.compute_torque(torque_limitation=self.cfg.joint_config.torque_limits)
            
            # Step physics
            mujoco.mj_step(self.model, self.data)
        
        # Update state after all physics steps
        self.update_state()
            
        # Render if viewer is active
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
            
        return True

    def compute_torque(self, torque_limitation=None):
        """Compute joint control torques using PD control.
        
        torque_limitation: Optional limit for joint torques, in controller order.
        return: Full joint torques in mujoco order.
        """
        # Get current unmapped joint state
        q_all = self.data.qpos[7:].copy()
        dq_all = self.data.qvel[6:].copy()
        
        if not self.lag_buffer:
            # No lag buffer or commands available
            return np.zeros(len(self._joint_names_orig))
            
        # Get commands from lag buffer
        cmd = self.lag_buffer[0]
        target_q = cmd["q"]
        self._tar_dof_pos = target_q.copy()
        target_dq = cmd["dq"]
        target_trq = cmd["trq"]
        target_kp = cmd["kp"]
        target_kd = cmd["kd"]
        
        # Get current mapped joint state
        q = np.array([q_all[i] for i in self._joint_mapping])
        dq = np.array([dq_all[i] for i in self._joint_mapping])
        
        # Calculate PD control torque
        tau = target_trq + target_kp * (target_q - q) + target_kd * (target_dq - dq)
        
        # Apply torque limits if specified
        if torque_limitation is not None:
            torque_limitation = np.array(torque_limitation)
            tau = np.clip(tau, -torque_limitation, torque_limitation)
        
        # Store the mapped torques directly in _dof_trq for reporting
        self._dof_trq = tau.copy()
        
        # Map torques back to full joint array
        full_tau = np.zeros(len(self._joint_names_orig))
        for mapped_idx, orig_idx in enumerate(self._joint_mapping):
            if mapped_idx < len(tau):
                full_tau[orig_idx] = tau[mapped_idx]
        print('full_tau:', full_tau)
        print('tau:', tau)
        return full_tau

    def set_joint_commands(self, q, dq, trq, kp, kd):
        """Set joint commands with lag simulation if configured."""
        # Normalize inputs to 1D float arrays (avoid unexpected shapes/types)
        q   = np.asarray(q, dtype=np.float32).ravel()
        dq  = np.asarray(dq, dtype=np.float32).ravel()
        trq = np.asarray(trq, dtype=np.float32).ravel()
        kp  = np.asarray(kp, dtype=np.float32).ravel()
        kd  = np.asarray(kd, dtype=np.float32).ravel()
        # optional: validate length if you expect a fixed number of DOFs
        if q.size >= len(self._joint_names):
            q = q[:len(self._joint_names)]
            dq = dq[:len(self._joint_names)]
            trq = trq[:len(self._joint_names)]
            kp = kp[:len(self._joint_names)]
            kd = kd[:len(self._joint_names)]
        else:
            raise ValueError("q length mismatch")

        # Apply joint limits if configured
        if (self.cfg.joint_config and 
            self.cfg.joint_config.joint_limits_min is not None and 
            self.cfg.joint_config.joint_limits_max is not None):
            try:
                limits_min = np.array(self.cfg.joint_config.joint_limits_min)
                limits_max = np.array(self.cfg.joint_config.joint_limits_max)
                
                if len(limits_min) >= len(q) and len(limits_max) >= len(q):
                    q = np.clip(q, limits_min[:len(q)], limits_max[:len(q)])
            except Exception as e:
                logger_mp.warning(f"Warning: Could not apply joint limits: {e}")
        
        # Create new command
        new_cmd = {
            "q": q.copy(), 
            "dq": dq.copy(), 
            "trq": trq.copy(), 
            "kp": kp.copy(), 
            "kd": kd.copy()
        }
        
        # Initialize lag buffer if it doesn't exist
        if self.lag_buffer is None:
            self.lag_buffer = [new_cmd]
            return
        
        # Update lag buffer based on lag configuration
        if self.cfg.hardware_sim and self.cfg.hardware_sim.lag_steps > 0:
            # Push new command to the end of the buffer
            self.lag_buffer = self.lag_buffer[1:] + [new_cmd]
        else:
            # No lag - directly set current command
            self.lag_buffer = [new_cmd]

    def set_joint_commands_multi(self, env_idx: int, q, dq, trq, kp, kd):
        """Set joint commands for a specific environment (multi-env mode)."""
        if self.cfg.num_envs <= 1:
            # Fall back to single-env method
            return self.set_joint_commands(q, dq, trq, kp, kd)
        
        if env_idx < 0 or env_idx >= self.cfg.num_envs:
            raise ValueError(f"Invalid env_idx {env_idx}, must be 0 to {self.cfg.num_envs-1}")
        
        # Normalize inputs to 1D float arrays
        q   = np.asarray(q, dtype=np.float32).ravel()
        dq  = np.asarray(dq, dtype=np.float32).ravel()
        trq = np.asarray(trq, dtype=np.float32).ravel()
        kp  = np.asarray(kp, dtype=np.float32).ravel()
        kd  = np.asarray(kd, dtype=np.float32).ravel()
        
        if q.size >= len(self._joint_names):
            q = q[:len(self._joint_names)]
            dq = dq[:len(self._joint_names)]
            trq = trq[:len(self._joint_names)]
            kp = kp[:len(self._joint_names)]
            kd = kd[:len(self._joint_names)]
        else:
            raise ValueError("q length mismatch")

        # Apply joint limits if configured
        if (self.cfg.joint_config and 
            self.cfg.joint_config.joint_limits_min is not None and 
            self.cfg.joint_config.joint_limits_max is not None):
            try:
                limits_min = np.array(self.cfg.joint_config.joint_limits_min)
                limits_max = np.array(self.cfg.joint_config.joint_limits_max)
                
                if len(limits_min) >= len(q) and len(limits_max) >= len(q):
                    q = np.clip(q, limits_min[:len(q)], limits_max[:len(q)])
            except Exception as e:
                logger_mp.warning(f"Warning: Could not apply joint limits: {e}")
        
        # Create new command
        new_cmd = {
            "q": q.copy(), 
            "dq": dq.copy(), 
            "trq": trq.copy(), 
            "kp": kp.copy(), 
            "kd": kd.copy()
        }
        
        # Initialize lag buffer if it doesn't exist
        if self.lag_buffer_list[env_idx] is None:
            self.lag_buffer_list[env_idx] = [new_cmd]
            return
        
        # Update lag buffer based on lag configuration
        if self.cfg.hardware_sim and self.cfg.hardware_sim.lag_steps > 0:
            # Push new command to the end of the buffer
            self.lag_buffer_list[env_idx] = self.lag_buffer_list[env_idx][1:] + [new_cmd]
        else:
            # No lag - directly set current command
            self.lag_buffer_list[env_idx] = [new_cmd]
    
    def compute_torque_multi(self, env_idx: int, torque_limitation=None):
        """Compute joint control torques for a specific environment using PD control."""
        if self.cfg.num_envs <= 1:
            return self.compute_torque(torque_limitation)
        
        if env_idx < 0 or env_idx >= self.cfg.num_envs:
            raise ValueError(f"Invalid env_idx {env_idx}")
        
        # Get current unmapped joint state from this environment's data
        data_i = self.datalist[env_idx]
        q_all = data_i.qpos[7:].copy()
        dq_all = data_i.qvel[6:].copy()
        
        if not self.lag_buffer_list[env_idx]:
            # No commands available
            return np.zeros(len(self._joint_names_orig))
            
        # Get commands from lag buffer
        cmd = self.lag_buffer_list[env_idx][0]
        target_q = cmd["q"]
        target_dq = cmd["dq"]
        target_trq = cmd["trq"]
        target_kp = cmd["kp"]
        target_kd = cmd["kd"]
        
        # Get current mapped joint state
        q = np.array([q_all[i] for i in self._joint_mapping])
        dq = np.array([dq_all[i] for i in self._joint_mapping])
        
        # Calculate PD control torque
        tau = target_trq + target_kp * (target_q - q) + target_kd * (target_dq - dq)
        
        # Apply torque limits if specified
        if torque_limitation is not None:
            torque_limitation = np.array(torque_limitation)
            tau = np.clip(tau, -torque_limitation, torque_limitation)
        
        # Map torques back to full joint array
        full_tau = np.zeros(len(self._joint_names_orig))
        for mapped_idx, orig_idx in enumerate(self._joint_mapping):
            if mapped_idx < len(tau):
                full_tau[orig_idx] = tau[mapped_idx]
        
        return full_tau

    def reset(self, joint_positions=None, base_position=None, base_orientation=None):
        """Reset the simulator state.
        
        joint_positions: List of joint positions to set, in controller order.
        """
        if base_position is not None:
            self.data.qpos[0:3] = base_position
        
        if base_orientation is not None:
            self.data.qpos[3:7] = base_orientation
            
        if joint_positions is not None:
            # Apply joint mapping
            full_positions = self.data.qpos[7:].copy()
            for mapped_idx, orig_idx in enumerate(self._joint_mapping):
                if mapped_idx < len(joint_positions):
                    full_positions[orig_idx] = joint_positions[mapped_idx]
            self.data.qpos[7:] = full_positions
            
        # Zero out velocities
        self.data.qvel[:] = 0.0
        
        # Reset lag buffer if present
        if self.lag_buffer is not None:
            num_dofs = len(self._joint_names)
            neutral_cmd = {
                "q": np.zeros(num_dofs) if joint_positions is None else joint_positions.copy(),
                "dq": np.zeros(num_dofs),
                "trq": np.zeros(num_dofs),
                "kp": np.zeros(num_dofs),
                "kd": np.zeros(num_dofs),
            }
            self.lag_buffer = [dict(neutral_cmd) for _ in range(self.cfg.hardware_sim.lag_steps + 1)]
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
        self.update_state()
    
    def get_state_dict(self):
        """Get complete state as a dictionary."""
        return {
            "base_position": self._pos_world.copy(),
            "base_orientation": self._quat_world.copy(),
            "base_linear_velocity_world": self._lin_vel_world.copy(),
            "base_angular_velocity_body": self._ang_vel_body.copy(),
            "joint_positions": self._dof_pos.copy(),
            "joint_velocities": self._dof_vel.copy(),
            "joint_torques": self._dof_trq.copy(),
        }
    
    def _quaternion_to_rotation_matrix(self, q):
        # find a function `mujoco.mju_quat2Mat()`, which may be useful
        """Convert quaternion to rotation matrix."""
        # q is in w, x, y, z format
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
    
    def _quaternion_to_axis_angle(self, q):
        """Convert quaternion to axis-angle representation."""
        # q is in w, x, y, z format
        w, x, y, z = q
        
        # Handle the case where w is close to 1 (identity rotation)
        if abs(w - 1.0) < 1e-12:
            return np.zeros(3)
            
        angle = 2 * np.arccos(w)
        s = np.sqrt(1 - w*w)
        if abs(s) < 1e-12:
            # If s is close to zero, the direction is arbitrary
            return np.zeros(3)
            
        axis = np.array([x, y, z]) / s
        return axis * angle
    
    def _axis_angle_to_quaternion(self, axis_angle):
        # find a function `mujoco.mju_axisAngle2Quat()`, which may be useful
        """Convert axis-angle representation to quaternion."""
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-12:
            # If angle is very small, return identity quaternion
            return np.array([1.0, 0.0, 0.0, 0.0])
            
        axis = axis_angle / angle
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        
        return np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])

    def _quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # q is in w, x, y, z format
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    # Properties with explicit frame indication
    @property
    def position(self):
        """Base position in world frame."""
        return self._pos_world
    
    @property
    def quaternion(self):
        """Base orientation as quaternion [w, x, y, z] in world frame."""
        return self._quat_world

    @property
    def quaternion_body(self):
        """Base orientation as quaternion [w, x, y, z] relative to world but expressed in body frame."""
        # return self._quat_world  * [1, -1, -1, -1]  # Invert vector part for body frame
        quat_body = np.zeros(4)
        mujoco.mju_negQuat(quat_body, self._quat_world)
        return quat_body
    
    @property
    def linear_velocity(self):
        """Linear velocity in world frame by default."""
        return self._lin_vel_world

    @property
    def angular_velocity(self):
        """Angular velocity in body frame by default (native MuJoCo format)."""
        return self._ang_vel_body

    def get_linear_velocity(self, frame=ReferenceFrame.WORLD):
        """Get linear velocity in specified frame."""
        if frame == ReferenceFrame.BODY:
            return self._lin_vel_body
        return self._lin_vel_world

    def get_angular_velocity(self, frame=ReferenceFrame.BODY):
        """Get angular velocity in specified frame."""
        if frame == ReferenceFrame.WORLD:
            return self._ang_vel_world
        return self._ang_vel_body
    
    @property
    def linear_acceleration(self):
        """Linear acceleration in world frame."""
        return self._lin_acc_world
    
    @property
    def joint_positions(self):
        """Joint positions after mapping."""
        return self._dof_pos

    @property
    def target_joint_positions(self):
        """Target joint positions after mapping."""
        return self._tar_dof_pos
    
    @property
    def joint_velocities(self):
        """Joint velocities after mapping."""
        return self._dof_vel
    
    @property
    def joint_torques(self):
        """Joint torques after mapping."""
        return self._dof_trq
    
    @property
    def joint_names(self):
        """Joint names after mapping."""
        return self._joint_names
    
    # Direct frame-specific accessors
    @property
    def linear_velocity_world(self):
        """Linear velocity in world frame."""
        return self._lin_vel_world
    
    @property
    def linear_velocity_body(self):
        """Linear velocity in body frame."""
        return self._lin_vel_body
    
    @property
    def angular_velocity_world(self):
        """Angular velocity in world frame."""
        return self._ang_vel_world
    
    @property
    def angular_velocity_body(self):
        """Angular velocity in body frame (native MuJoCo format)."""
        return self._ang_vel_body
    
    # Frame transformation methods
    def transform_to_world_frame(self, vector, is_position=False):
        """Transform a vector from body to world frame."""
        if is_position:
            return (self._world_to_body_rot @ vector) + self._pos_world
        else:
            return self._world_to_body_rot @ vector
    
    def transform_to_body_frame(self, vector, is_position=False):
        """Transform a vector from world to body frame."""
        if is_position:
            return self._world_to_body_rot.T @ (vector - self._pos_world)
        else:
            return self._world_to_body_rot.T @ vector

    # --- ViewerPlus integration helpers ---
    def add_ghost(self, qpos: np.ndarray) -> None:
        """Add a ghost pose to the viewer if ViewerPlus is enabled.
        
        Following mjlab pattern: simple wrapper around viewer.add_ghost().
        
        Args:
            qpos: 1D array (nq,) - full qpos for ghost rendering
        """
        if self.viewer is None:
            return
        try:
            if hasattr(self.viewer, "add_ghost"):
                self.viewer.add_ghost(qpos)
        except Exception:
            logger_mp.exception("add_ghost failed")

    def register_reward_terms(self, term_names: List[str]) -> None:
        """Register custom reward terms for plotting (ViewerPlus only)."""
        if self.viewer is None:
            return
        try:
            if hasattr(self.viewer, "register_reward_terms"):
                self.viewer.register_reward_terms(term_names)
        except Exception:
            logger_mp.exception("register_reward_terms failed")

    def update_reward_plot(self, rewards: Dict[str, float]) -> None:
        """Update reward curves (ViewerPlus only)."""
        if self.viewer is None:
            return
        try:
            if hasattr(self.viewer, "update_rewards"):
                self.viewer.update_rewards(rewards)
        except Exception:
            logger_mp.exception("update_reward_plot failed")
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()