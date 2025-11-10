from deploy_robot.sim.dds.dds_master import dds_manager
from deploy_robot.sim.dds.g1_robot_dds import G1RobotDDS

from deploy_robot.sim.mujoco_simulator import MujocoSimulator, MujocoSimulatorConfig

import logging_mp
logger_mp = logging_mp.get_logger(__name__)
from deploy_robot.sim.mujoco_multisimulator import MultiMujocoSimulator
import threading
import time
from typing import Optional
import numpy as np


class MultiMujocoEnv:
    """Wrapper of MujocoSimulator with fixed-Hz threaded stepping.

    - Publishes/consumes DDS messages to mimic hardware.
    - Steps simulator on a dedicated thread at fixed rate.
    - Uses a minimal lock only around simulator access (apply cmd, step, read state).
    """

    def __init__(self, cfg: MujocoSimulatorConfig, step_hz: Optional[float] = None):
        self.num_envs = cfg.num_envs
        self.simulator = MujocoSimulator(cfg) if self.num_envs == 1 else MultiMujocoSimulator(cfg)
        
        # DDS object registration
        # self.g1_robot_dds = G1RobotDDS()
        # dds_manager.register_object("g1_robot", self.g1_robot_dds)
        if self.num_envs > 1:
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
        """Read commands, step simulator, publish state, at fixed Hz."""
        logger_mp.debug("MujocoEnv.run() entering loop")
        next_time = time.perf_counter()
        behind_warn_interval = 1.0
        last_behind_log = 0.0

        try:
            while not self._stop_event.is_set():
                # Check if paused (but don't block - just skip stepping)
                is_paused = not self._pause_event.is_set()
                
                # Read latest command (non-blocking, no lock)
                # low_cmd = self.g1_robot_dds.get_robot_command()
                low_cmds = [self.g1_robot_dds_list[i].get_robot_command() for i in range(self.num_envs)] if self.num_envs > 1 else None

                # Apply command, step, and copy state under short lock
                with self._sim_lock:
                    # if low_cmd:
                    #     motor_cmd = low_cmd["motor_cmd"]
                    #     self.simulator.set_joint_commands(
                    #         q=motor_cmd["positions"],
                    #         dq=motor_cmd["velocities"],
                    #         trq=motor_cmd["torques"],
                    #         kp=motor_cmd["kp"],
                    #         kd=motor_cmd["kd"],
                    #     )
                    if low_cmds:
                        for i in range(self.num_envs):
                            if low_cmds[i]:
                                motor_cmd = low_cmds[i]["motor_cmd"]
                                self.simulator.set_multi_joint_commands(
                                    q=motor_cmd["positions"],
                                    dq=motor_cmd["velocities"],
                                    trq=motor_cmd["torques"],
                                    kp=motor_cmd["kp"],
                                    kd=motor_cmd["kd"],
                                    env_idx=i
                                )

                    # Only step the simulator if not paused
                    if not is_paused:
                        self.simulator.step()

                    # Always read state (even when paused) for DDS publishing
                    rpy = self.simulator._quaternion_to_euler(self.simulator.quaternion)
                    quat = self.simulator.quaternion.copy()
                    lin_acc_body = self.simulator.transform_to_body_frame(self.simulator.linear_acceleration).copy()
                    ang_vel_body = self.simulator.angular_velocity_body.copy()
                    joint_positions = self.simulator.joint_positions.copy()
                    joint_velocities = self.simulator.joint_velocities.copy()
                    joint_torques = self.simulator.joint_torques.copy()

                    joint_positions_list = []
                    joint_velocities_list = []
                    joint_torques_list = []
                    rpy_list = []
                    quat_list = []
                    lin_acc_body_list = []
                    ang_vel_body_list = []
    
                    for i in range(self.num_envs):
                        # 假设 MultiMujocoSimulator 有 get_state_for_env(i) 或类似接口
                        # 或者直接访问 _dof_pos_list[i] 等
                        rpy_i = self.simulator._quaternion_to_euler(self.simulator.quaternion_list[i])
                        quat_i = self.simulator.quaternion_list[i].copy()
                        lin_acc_body_i = self.simulator.transform_to_body_frame(
                            self.simulator.linear_acceleration_list[i]
                        ).copy()
                        ang_vel_body_i = self.simulator.angular_velocity_body_list[i].copy()
                        joint_positions_i = self.simulator.joint_positions_list[i].copy()
                        joint_velocities_i = self.simulator.joint_velocities_list[i].copy()
                        joint_torques_i = self.simulator.joint_torques_list[i].copy()

                        joint_positions_list.append(joint_positions_i)
                        joint_velocities_list.append(joint_velocities_i)
                        joint_torques_list.append(joint_torques_i)
                        rpy_list.append(rpy_i)
                        quat_list.append(quat_i)
                        lin_acc_body_list.append(lin_acc_body_i)
                        ang_vel_body_list.append(ang_vel_body_i)


                # Publish without holding the lock (always publish, even when paused)
                # imu_data = np.concatenate([rpy, quat, lin_acc_body, ang_vel_body])
                # self.g1_robot_dds.write_robot_state(
                #     joint_positions=joint_positions,
                #     joint_velocities=joint_velocities,
                #     joint_torques=joint_torques,
                #     imu_data=imu_data,
                # )
                for i in range(self.num_envs):
                    imu_data = np.concatenate([rpy_list[i], quat_list[i], 
                                            lin_acc_body_list[i], ang_vel_body_list[i]])
                    self.g1_robot_dds_list[i].write_robot_state(
                        joint_positions=joint_positions_list[i],
                        joint_velocities=joint_velocities_list[i],
                        joint_torques=joint_torques_list[i],
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
                        logger_mp.debug("MujocoEnv loop falling behind by %.6f s", -sleep_time)
                        last_behind_log = now

        except Exception:
            logger_mp.exception("Unhandled exception in MujocoEnv.run()")
        finally:
            self._running = False
            logger_mp.debug("MujocoEnv.run() exiting")

    @property
    def is_running(self) -> bool:
        return self._running
