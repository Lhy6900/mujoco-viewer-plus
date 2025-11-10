import mujoco
import numpy as np
from typing import Optional, List, Dict, Any
from collections import deque

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


class ViewerPlus:
    """A lightweight enhanced MuJoCo viewer with ghost rendering and reward plots.

    Simplified design following mjlab pattern:
    - Direct ghost rendering (no complex trajectory playback)
    - Single environment support (first env only)
    - Reward plotting with MjvFigure
    - Ctrl+G to toggle ghost, Ctrl+R to toggle reward plots
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Optional[dict] = None, num_envs: int = 1):
        self.model = model
        self.data = data
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

        # Ghost qpos for current frame (simple: just one qpos per frame)
        self._ghost_qpos: Optional[np.ndarray] = None

        # Reward plotter
        from deploy_robot.sim.viewer_plus.reward_plotter import RewardPlotter
        history_len = int(self.cfg.get("reward_history_length", 300))
        self._reward_plotter = RewardPlotter(history_length=history_len)
        self._registered_reward_terms: List[str] = []

        # Control flags for UI toggles
        self._show_ghost = bool(self.cfg.get("enable_ghost", True))
        self._show_reward = bool(self.cfg.get("enable_reward_plot", False))
        self._show_multi_envs = bool(self.cfg.get("enable_multi_envs", False))  # Show other environments as solid robots
        
        # Track ctrl/shift modifier state (simple heuristic)
        self._ctrl_pressed = False
        
        # Key event buffer for external consumption (e.g. environment switching)
        self._key_events = []
        
        # Multi-environment rendering: store other environments' qpos
        self._num_envs = num_envs
        self._other_envs_qpos: List[Optional[np.ndarray]] = [None] * num_envs
        self._current_env_idx = 0  # Track which environment is currently active/main
        
        # Create models for other environments (keeping original colors)
        self._other_env_models = []
        self._other_env_data = []
        
        for i in range(num_envs):
            # Simply deep copy the model without any color changes
            env_model = copy.deepcopy(model)
            self._other_env_models.append(env_model)
            self._other_env_data.append(mujoco.MjData(env_model))

    def _key_callback(self, key: int) -> None:
        """Handle keyboard input with Ctrl modifier support.
        
        MuJoCo viewer key codes:
        - Ctrl is typically encoded in high bits or modifier field
        - For simplicity, we detect common Ctrl+key patterns
        """
        # GLFW key codes for modifier keys
        GLFW_MOD_CONTROL = 341  # Left Ctrl
        GLFW_MOD_CONTROL_R = 345  # Right Ctrl
        
        # GLFW arrow key codes
        GLFW_KEY_UP = 265
        GLFW_KEY_DOWN = 264
        GLFW_KEY_LEFT = 263
        GLFW_KEY_RIGHT = 262
        
        # Track Ctrl key state
        if key in (GLFW_MOD_CONTROL, GLFW_MOD_CONTROL_R):
            self._ctrl_pressed = True
            return
        
        # Handle arrow keys - store as special strings
        if key == GLFW_KEY_UP:
            self._key_events.append('ARROW_UP')
            return
        elif key == GLFW_KEY_DOWN:
            self._key_events.append('ARROW_DOWN')
            return
        elif key == GLFW_KEY_LEFT:
            self._key_events.append('ARROW_LEFT')
            return
        elif key == GLFW_KEY_RIGHT:
            self._key_events.append('ARROW_RIGHT')
            return
        
        # Extract base character
        base_key = key & 0xFF
        try:
            ch = chr(base_key) if base_key < 128 else ''
        except Exception:
            ch = ''
        
        # Store key event for external consumers (e.g., environment switching)
        if ch:
            self._key_events.append(ch)
        
        # Handle shortcuts
        if ch.lower() == 'g':
            if self._ctrl_pressed:
                self._show_ghost = not self._show_ghost
                logger_mp.info("ViewerPlus: toggle ghost (Ctrl+G) -> %s", self._show_ghost)
            else:
                # Fallback: plain 'g' also works
                self._show_ghost = not self._show_ghost
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
            
        elif ch.lower() == 'm':
            # Toggle multi-environment rendering
            self._show_multi_envs = not self._show_multi_envs
            logger_mp.info("ViewerPlus: toggle multi-env rendering (m) -> %s", self._show_multi_envs)
            self._ctrl_pressed = False  # Reset after use
            
        else:
            # Reset ctrl state if another key is pressed
            self._ctrl_pressed = False

    def add_ghost(self, qpos: np.ndarray) -> None:
        """Set ghost pose for rendering in next sync() call.
        
        Following mjlab pattern: simple and direct.
        qpos: 1D array of shape (nq,) - joint positions for ghost
        
        Note: This is called every frame before sync(). Only one ghost pose
        is stored at a time (replaces previous if called multiple times).
        """
        if qpos is None:
            self._ghost_qpos = None
            return
        
        arr = np.asarray(qpos, dtype=np.float32)
        if arr.ndim != 1:
            logger_mp.warning("add_ghost expects 1D qpos, got shape %s", arr.shape)
            return
        
        if arr.shape[0] != self.model.nq:
            logger_mp.warning("add_ghost qpos length %d != model.nq %d", arr.shape[0], self.model.nq)
            return
        
        self._ghost_qpos = arr.copy()

    def set_other_env_qpos(self, env_idx: int, qpos: np.ndarray) -> None:
        """Set qpos for other environments to render as solid robots.
        
        Args:
            env_idx: Environment index (0 to num_envs-1)
            qpos: 1D array of shape (nq,) - joint positions for that environment
        """
        if env_idx < 0 or env_idx >= self._num_envs:
            logger_mp.warning("set_other_env_qpos: invalid env_idx %d", env_idx)
            return
            
        if qpos is None:
            self._other_envs_qpos[env_idx] = None
            return
        
        arr = np.asarray(qpos, dtype=np.float32)
        if arr.ndim != 1:
            logger_mp.warning("set_other_env_qpos expects 1D qpos, got shape %s", arr.shape)
            return
        
        if arr.shape[0] != self.model.nq:
            logger_mp.warning("set_other_env_qpos: qpos length %d != model.nq %d", arr.shape[0], self.model.nq)
            return
        
        self._other_envs_qpos[env_idx] = arr.copy()

    def set_current_env_idx(self, env_idx: int) -> None:
        """Set the current active environment index.
        
        This tells the viewer which environment is the main one,
        so it won't be rendered again in multi-env mode.
        
        Args:
            env_idx: Current active environment index (0 to num_envs-1)
        """
        if env_idx < 0 or env_idx >= self._num_envs:
            logger_mp.warning("set_current_env_idx: invalid env_idx %d", env_idx)
            return
        self._current_env_idx = env_idx

    def get_key_events(self) -> List[str]:
        """Get and clear buffered key events.
        
        Returns:
            List of key characters pressed since last call.
            List is cleared after retrieval.
        """
        events = self._key_events.copy()
        self._key_events.clear()
        return events

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

        # Clear previous debug geoms
        try:
            self.viewer.user_scn.ngeom = 0
        except Exception:
            pass

        # Render other environments as solid robots if enabled
        if self._show_multi_envs:
            vopt_solid = mujoco.MjvOption()
            vopt_solid.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            
            for env_idx in range(self._num_envs):
                # Skip the current main environment to avoid double rendering
                if env_idx == self._current_env_idx:
                    continue
                    
                qpos = self._other_envs_qpos[env_idx]
                if qpos is not None:
                    try:
                        self._other_env_data[env_idx].qpos[:] = qpos
                        mujoco.mj_forward(self._other_env_models[env_idx], self._other_env_data[env_idx])
                        mujoco.mjv_addGeoms(
                            self._other_env_models[env_idx],
                            self._other_env_data[env_idx],
                            vopt_solid,
                            self._pert,
                            self._catmask,
                            self.viewer.user_scn,
                        )
                    except Exception:
                        logger_mp.exception(f"ViewerPlus: failed to render env {env_idx}")

        # Render ghost if enabled and qpos is set
        if self._show_ghost and self._ghost_qpos is not None:
            try:
                self._viz_data.qpos[:] = self._ghost_qpos

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
                logger_mp.exception("ViewerPlus: failed to render ghost")

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
