import mujoco
import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


class ViewerPlus:
    """A lightweight enhanced MuJoCo viewer supporting multi-env ghost rendering and reward plots.

    This is intentionally minimal and uses MuJoCo native APIs so it has small dependencies.
    Features:
    - Multi-environment ghost rendering (each env can have own trajectory)
    - Trajectory playback aligned with simulation time_step
    - Reward plotting with MjvFigure
    - Ctrl+G to toggle ghost, Ctrl+R to toggle reward plots
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Optional[dict] = None, num_envs: int = 1):
        self.model = model
        self.data = data
        self.num_envs = int(num_envs)
        self.cfg = config or {}

        # Native viewer handle
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._key_callback)

        # Create ghost model (deep copy with semi-transparent appearance)
        # Following mjlab's approach: create a separate model for ghost with custom colors
        import copy
        self._ghost_model = copy.deepcopy(model)
        ghost_color = self.cfg.get("ghost_color", [0.5, 0.7, 0.5, 0.5])  # Semi-transparent green
        self._ghost_model.geom_rgba[:] = np.array(ghost_color, dtype=np.float32)
        
        # Auxiliary data used for ghost rendering (one MjData reused per ghost render)
        self._viz_data = mujoco.MjData(self._ghost_model)
        self._vopt = mujoco.MjvOption()
        self._vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        self._pert = mujoco.MjvPerturb()
        self._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value

        # Ghost storage: per-frame list of tuples (env_idx, qpos)
        self._ghost_list: List[tuple[int, np.ndarray]] = []
        
        # Trajectory playback: dict env_idx -> (trajectory_array, current_step)
        # trajectory_array shape: (T, nq)
        self._trajectories: Dict[int, tuple[np.ndarray, int]] = {}
        self._trajectory_step = 0  # Global step counter for trajectory playback

        # Reward plotter
        from deploy_robot.sim.viewer_plus.reward_plotter import RewardPlotter
        history_len = int(self.cfg.get("reward_history_length", 300))
        self._reward_plotter = RewardPlotter(history_length=history_len)
        self._registered_reward_terms: List[str] = []

        # Control flags for UI toggles
        self._show_ghost = bool(self.cfg.get("enable_ghost", True))
        self._show_reward = bool(self.cfg.get("enable_reward_plot", False))
        
        # Track ctrl/shift modifier state (simple heuristic)
        self._ctrl_pressed = False

    def _key_callback(self, key: int) -> None:
        """Handle keyboard input with Ctrl modifier support.
        
        MuJoCo viewer key codes:
        - Ctrl is typically encoded in high bits or modifier field
        - For simplicity, we detect common Ctrl+key patterns
        """
        # GLFW key codes for modifier keys
        GLFW_MOD_CONTROL = 341  # Left Ctrl
        GLFW_MOD_CONTROL_R = 345  # Right Ctrl
        
        # Track Ctrl key state
        if key in (GLFW_MOD_CONTROL, GLFW_MOD_CONTROL_R):
            self._ctrl_pressed = True
            return
        
        # Extract base character
        base_key = key & 0xFF
        try:
            ch = chr(base_key) if base_key < 128 else ''
        except Exception:
            ch = ''
        
        # Handle shortcuts
        if ch.lower() == 'g':
            if self._ctrl_pressed:
                self._show_ghost = not self._show_ghost
                logger_mp.info("ViewerPlus: toggle ghost (Ctrl+G) -> %s", self._show_ghost)
            else:
                # Fallback: plain 'g' also works
                # self._show_ghost = not self._show_ghost
                logger_mp.info("ViewerPlus: toggle ghost (g) -> %s", self._show_ghost)
            self._ctrl_pressed = False  # Reset after use
            
        elif ch.lower() == 'r':
            if self._ctrl_pressed:
                self._show_reward = not self._show_reward
                logger_mp.info("ViewerPlus: toggle reward (Ctrl+R) -> %s", self._show_reward)
            else:
                # Fallback: plain 'r' also works
                # self._show_reward = not self._show_reward
                logger_mp.info("ViewerPlus: toggle reward (r) -> %s", self._show_reward)
            self._ctrl_pressed = False  # Reset after use
        else:
            # Reset ctrl state if another key is pressed
            self._ctrl_pressed = False

    def add_ghost(self, qpos: np.ndarray, env_idx: int = 0) -> None:
        """Schedule a ghost pose to be rendered on next sync (single frame).

        qpos: 1D array of qpos length (nq) or 2D shaped (num_envs, nq)
        env_idx: which environment index this ghost corresponds to (for multi-env)
        """
        if qpos is None:
            return
        arr = np.asarray(qpos).copy()
        if arr.ndim == 2:
            # if batch, pick env_idx row (if in range) or 0
            if arr.shape[0] > env_idx:
                arr = arr[env_idx]
            else:
                arr = arr[0]
        self._ghost_list.append((int(env_idx), arr))

    def set_trajectory(self, trajectory: np.ndarray, env_idx: int = 0) -> None:
        """Set a trajectory to play back aligned with simulation time_step.
        
        trajectory: shape (T, nq) or (nq,) - if 1D, treat as single pose
        env_idx: which environment this trajectory belongs to
        
        The trajectory will be played back frame-by-frame synchronized with 
        the simulation step counter (self._trajectory_step).
        """
        if trajectory is None:
            if env_idx in self._trajectories:
                del self._trajectories[env_idx]
            return
        
        arr = np.asarray(trajectory).copy()
        if arr.ndim == 1:
            # Single pose, wrap in array
            arr = arr.reshape(1, -1)
        
        # Store (trajectory, start_step=0)
        self._trajectories[env_idx] = (arr, 0)
        logger_mp.info("ViewerPlus: set trajectory for env %d with %d frames", env_idx, arr.shape[0])

    def reset_trajectory_playback(self) -> None:
        """Reset trajectory playback to frame 0 for all environments."""
        self._trajectory_step = 0
        for env_idx in self._trajectories:
            traj, _ = self._trajectories[env_idx]
            self._trajectories[env_idx] = (traj, 0)

    def register_reward_terms(self, term_names: List[str]) -> None:
        """Register reward terms to be plotted.
        
        This allows users to define custom reward metrics.
        """
        self._registered_reward_terms = list(term_names)
        self._reward_plotter.register_terms(term_names)
        logger_mp.info("ViewerPlus: registered %d reward terms", len(term_names))

    def update_rewards(self, rewards: Dict[str, float]) -> None:
        """Update reward data for plotting.
        
        rewards: dict of {term_name: value}
        """
        if not rewards:
            return
        
        # Auto-register new terms if not already registered
        new_terms = [k for k in rewards.keys() if k not in self._registered_reward_terms]
        if new_terms:
            self.register_reward_terms(self._registered_reward_terms + new_terms)
        
        self._reward_plotter.update(rewards)

    def sync(self) -> None:
        """Synchronize visual elements and render. Call this every frame after simulation step."""
        if not self.viewer:
            return

        # Increment trajectory playback step
        self._trajectory_step += 1

        # clear previous debug geoms
        try:
            self.viewer.user_scn.ngeom = 0
        except Exception:
            pass

        if self._show_ghost:
            # 1. Render trajectory ghosts (if any)
            for env_idx, (traj, start_step) in list(self._trajectories.items()):
                frame_idx = (self._trajectory_step - start_step) % traj.shape[0]
                qpos = traj[frame_idx]
                try:
                    self._viz_data.qpos[:] = qpos
                    mujoco.mj_forward(self._ghost_model, self._viz_data)
                    mujoco.mjv_addGeoms(
                        self._ghost_model,
                        self._viz_data,
                        self._vopt,
                        self._pert,
                        self._catmask,
                        self.viewer.user_scn,
                    )
                except Exception:
                    logger_mp.exception("ViewerPlus: failed to add trajectory ghost for env %d", env_idx)
            
            # 2. Render per-frame ghosts (from add_ghost calls)
            for (_env_idx, qpos) in self._ghost_list:
                try:
                    self._viz_data.qpos[:] = qpos
                    mujoco.mj_forward(self._ghost_model, self._viz_data)
                    mujoco.mjv_addGeoms(
                        self._ghost_model,
                        self._viz_data,
                        self._vopt,
                        self._pert,
                        self._catmask,
                        self.viewer.user_scn,
                    )
                except Exception:
                    logger_mp.exception("ViewerPlus: failed to add per-frame ghost")

        # Reset per-frame ghost list (caller must re-add each frame if needed)
        self._ghost_list.clear()

        # Reward plotting: render MjvFigures if enabled
        if self._show_reward and self._registered_reward_terms:
            try:
                figures_data = self._reward_plotter.get_figures_for_viewer(self.viewer.viewport)
                if figures_data:
                    self.viewer.set_figures(figures_data)
                else:
                    self.viewer.set_figures([])
            except Exception:
                logger_mp.exception("ViewerPlus: failed to set reward figures")
        else:
            try:
                self.viewer.set_figures([])
            except Exception:
                pass

        # Finally sync native viewer state
        try:
            self.viewer.sync()
        except Exception:
            logger_mp.exception("ViewerPlus: viewer.sync() failed")

    def is_running(self) -> bool:
        return bool(self.viewer and self.viewer.is_running())

    def close(self) -> None:
        if self.viewer is not None:
            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except Exception:
                logger_mp.exception("ViewerPlus: error closing viewer")
