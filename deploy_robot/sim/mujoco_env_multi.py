from deploy_robot.sim.dds.dds_master import dds_manager
from deploy_robot.sim.dds.g1_robot_dds import G1RobotDDS

from deploy_robot.sim.mujoco_simulator import MujocoSimulator, MujocoSimulatorConfig
from deploy_robot.sim.mujoco_simulator_multi import MultiMujocoSimulator
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

import mujoco
import threading
import time
from typing import Optional, List
import numpy as np


class MultiMujocoEnv:
    """Wrapper of MultiMujocoSimulator with fixed-Hz threaded stepping.

    - Supports multiple parallel environments (num_envs > 1)
    - Each environment has its own DDS communication object
    - Steps all environments on a dedicated thread at fixed rate
    - Uses a minimal lock only around simulator access (apply cmd, step, read state)
    """

    def __init__(self, cfg: MujocoSimulatorConfig, step_hz: Optional[float] = None):
        # Use MultiMujocoSimulator for multi-environment support
        self.num_envs = cfg.num_envs if hasattr(cfg, 'num_envs') else 1
        self.simulator = MujocoSimulator(cfg)

        # DDS object registration - create one DDS object per environment
        # Each environment uses a different domain ID (1, 2, 3, ...)
        self.g1_robot_dds_list: List[G1RobotDDS] = []
        for i in range(self.num_envs):
            # Use domain_id = 1 + i for each environment
            dds_domain_id = 1 + i
            dds_obj = G1RobotDDS(node_name=f"g1_robot_{i}", dds_domain_id=dds_domain_id)
            dds_name = f"g1_robot_{i}" if self.num_envs > 1 else "g1_robot"
            dds_manager.register_object(dds_name, dds_obj)
            self.g1_robot_dds_list.append(dds_obj)
            logger_mp.info(f"Registered DDS object: {dds_name} (domain_id={dds_domain_id})")

        # Start DDS comms immediately (keep existing behavior)
        try:
            self.start_communication()
        except Exception:
            logger_mp.exception("Failed to start DDS communication in __init__")

        # Timing configuration
        if step_hz is None:
            # simulator.step() advances decimation * dt seconds per call
            sim_period = max(1e-6, self.simulator.cfg.dt * max(1, self.simulator.cfg.decimation))
            self.step_hz = 1.0 / sim_period
        else:
            self.step_hz = float(step_hz)
        self._step_period = 1.0 / self.step_hz

        # Threading state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Default: not paused
        self._running = False

        # Lock only for simulator access
        self._sim_lock = threading.Lock()

    def start_communication(self):
        dds_manager.start_publishing()
        dds_manager.start_subscribing()
        logger_mp.info("DDS communication started")

    def stop_communication(self):
        try:
            dds_manager.stop_all_communication()
        except Exception:
            logger_mp.exception("Error stopping DDS communication")
        else:
            logger_mp.info("DDS communication stopped")

    def start(self, start_communication: bool = False):
        """Start the simulation thread. Optionally (re)start DDS."""
        if self._running:
            logger_mp.debug("MujocoEnv.start() called but already running")
            return

        if start_communication:
            try:
                self.start_communication()
            except Exception:
                logger_mp.exception("Failed to (re)start communication in start()")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, name="MujocoEnvThread", daemon=True)
        self._running = True
        self._thread.start()
        logger_mp.info("MujocoEnv thread started at %.1f Hz", self.step_hz)

    def pause(self):
        """Pause the simulation thread (stops stepping but keeps thread alive)."""
        if not self._running:
            logger_mp.warning("Cannot pause: MujocoEnv is not running")
            return
        self._pause_event.clear()
        logger_mp.info("MujocoEnv simulation paused")

    def resume(self):
        """Resume the simulation thread."""
        if not self._running:
            logger_mp.warning("Cannot resume: MujocoEnv is not running")
            return
        self._pause_event.set()
        logger_mp.info("MujocoEnv simulation resumed")

    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return not self._pause_event.is_set()

    def stop(self, timeout: float = 2.0):
        """Signal the run thread to stop, join, close simulator, and stop DDS."""
        if not self._running and (self._thread is None or not self._thread.is_alive()):
            logger_mp.debug("MujocoEnv.stop() called but not running")
            # Ensure comms are down
            self.stop_communication()
            return

        logger_mp.info("Stopping MujocoEnv thread")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger_mp.warning("MujocoEnv thread did not exit within %.3f seconds", timeout)
        self._thread = None
        self._running = False

        # Close simulator safely
        try:
            with self._sim_lock:
                self.simulator.close()
        except Exception:
            logger_mp.exception("Error while closing simulator")

        # Stop DDS after thread stopped/cleanup
        self.stop_communication()
        logger_mp.info("MujocoEnv stopped and cleaned up")

    def run(self):
        """Read commands, step simulator, publish state, at fixed Hz.
        
        For multi-environment: reads commands from each DDS object, 
        applies to corresponding datalist entry, steps all environments,
        and publishes each environment's state back to its DDS object.
        """
        logger_mp.debug("MultiMujocoEnv.run() entering loop")
        next_time = time.perf_counter()
        behind_warn_interval = 1.0
        last_behind_log = 0.0

        try:
            while not self._stop_event.is_set():
                # Check if paused (but don't block - just skip stepping)
                is_paused = not self._pause_event.is_set()
                
                # Process each environment
                with self._sim_lock:
                    for i in range(self.num_envs):
                        # Read latest command for this environment (non-blocking)
                        low_cmd = self.g1_robot_dds_list[i].get_robot_command()
                        
                        # Get the appropriate data object
                        if self.num_envs > 1:
                            data_i = self.simulator.datalist[i]
                        else:
                            data_i = self.simulator.data
                        
                        # Apply command to this environment's data
                        if low_cmd:
                            motor_cmd = low_cmd["motor_cmd"]
                            # Set joint commands for this environment
                            if self.num_envs > 1:
                                self.simulator.set_joint_commands_multi(
                                    env_idx=i,
                                    q=motor_cmd["positions"],
                                    dq=motor_cmd["velocities"],
                                    trq=motor_cmd["torques"],
                                    kp=motor_cmd["kp"],
                                    kd=motor_cmd["kd"],
                                )
                            else:
                                self.simulator.set_joint_commands(
                                    q=motor_cmd["positions"],
                                    dq=motor_cmd["velocities"],
                                    trq=motor_cmd["torques"],
                                    kp=motor_cmd["kp"],
                                    kd=motor_cmd["kd"],
                                )
                        
                        # Only step the simulator if not paused
                        if not is_paused:
                            # Compute and apply control torques for this environment
                            if self.num_envs > 1:
                                torque_limits = self.simulator.cfg.joint_config.torque_limits if self.simulator.cfg.joint_config else None
                                tau = self.simulator.compute_torque_multi(i, torque_limitation=torque_limits)
                                data_i.ctrl[:] = tau
                                # Step individual environment
                                mujoco.mj_step(self.simulator.model, data_i)
                            else:
                                self.simulator.step()
                        
                        # Always read state (even when paused) for DDS publishing
                        quat = data_i.qpos[3:7].copy()
                        rpy = self.simulator._quaternion_to_euler(quat)
                        
                        # Transform linear acceleration to body frame
                        # Get rotation matrix from quaternion
                        rot_mat = self.simulator._quaternion_to_rotation_matrix(quat)
                        lin_acc_world = data_i.qacc[0:3].copy() if hasattr(data_i, 'qacc') else np.zeros(3)
                        lin_acc_body = rot_mat.T @ lin_acc_world
                        
                        ang_vel_body = data_i.qvel[3:6].copy()
                        
                        # Get joint states with remapping
                        dof_pos_all = data_i.qpos[7:].copy()
                        dof_vel_all = data_i.qvel[6:].copy()
                        joint_positions = np.array([dof_pos_all[j] for j in self.simulator._joint_mapping])
                        joint_velocities = np.array([dof_vel_all[j] for j in self.simulator._joint_mapping])
                        
                        # Get joint torques (from ctrl or compute from PD)
                        joint_torques = data_i.ctrl[:len(self.simulator._joint_names)].copy()
                        
                        # Publish without holding the lock (always publish, even when paused)
                        imu_data = np.concatenate([rpy, quat, lin_acc_body, ang_vel_body])
                        self.g1_robot_dds_list[i].write_robot_state(
                            joint_positions=joint_positions,
                            joint_velocities=joint_velocities,
                            joint_torques=joint_torques,
                            imu_data=imu_data,
                        )

                # Fixed-rate timing
                next_time += self._step_period
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    now = time.perf_counter()
                    if (now - last_behind_log) >= behind_warn_interval:
                        logger_mp.debug("MultiMujocoEnv loop falling behind by %.6f s", -sleep_time)
                        last_behind_log = now

        except Exception:
            logger_mp.exception("Unhandled exception in MultiMujocoEnv.run()")
        finally:
            self._running = False
            logger_mp.debug("MultiMujocoEnv.run() exiting")

    @property
    def is_running(self) -> bool:
        return self._running
