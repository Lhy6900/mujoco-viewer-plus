import time
import numpy as np
from pathlib import Path

# multi-process safe logger
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

import os, logging
_lvl = getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO)
logging.getLogger().setLevel(_lvl)
logger_mp.setLevel(_lvl)

# data logger (writer)
from deploy_robot.utils.writer import SimpleWriter
logger = SimpleWriter()

# 1. prepare model and robot interface
from deploy_robot.controllers.beyondmimic import make_policy, make_robot, run_policy

model_path = "model/dalafan_prog.onnx"

policy = make_policy(model_path)
robot = make_robot({
    "joint_names": policy.joint_names,
    "joint_stiffness": policy.joint_stiffness,
    "joint_damping": policy.joint_damping,
    "default_joint_pos": policy.default_joint_positions,
}, env='sim')

# In simulation, we will also need to start the simulator
from deploy_robot.sim.mujoco_env import MujocoEnv
from deploy_robot.sim.config.g1 import G1MujocoConfig

# Enable ViewerPlus with ghost and reward plotting
G1MujocoConfig.viewer_type = "viewer_plus"
G1MujocoConfig.viewer_plus_config = {
    "enable_ghost": True,
    "enable_reward_plot": True,
    "reward_history_length": 200,
}

env = MujocoEnv(G1MujocoConfig)

env.start()

# Register reward terms for plotting (after env.start() so viewer is initialized)
if hasattr(env.simulator, "register_reward_terms"):
    reward_terms = [
        "smoothness",
        "pos_tracking_global",
        "pos_tracking_local",
        "quat_tracking_global",
        "quat_tracking_local",
    ]
    env.simulator.register_reward_terms(reward_terms)
    logger_mp.info("Registered %d reward terms for plotting", len(reward_terms))

# 2. run the robot
robot.start_communication()

# phase 1: control robot to the default pose
try:
    logger_mp.info("Running to default pose...")
    robot.set_default_posture()
    logger_mp.info("Reached default pose.")
except Exception as e:
    logger_mp.error(f"Error during reaching default pose: {e}")

    env.stop()
    exit(1)

# phase 2: run the policy loop 
time_step = 0
control_dt = 0.02  # 50 Hz
last_time = time.time()

# Variables for tracking error computation
last_joint_vel = np.zeros(len(robot.joint_pos))

try:
    # logger_mp.info("Press 'A' button to start the main control loop...")
    # while True:
    #     time.sleep(0.2)
    #     robot.update_state()
    #     if robot.joystick.A.pressed:
    #         break

    logger_mp.info("Starting main control loop...")
    while True:
        # safety control
        # if robot.joystick.B.pressed:
        #     robot.enable_control = False
        #     robot.enable_motor = False
        #     break

        while (time.time()-last_time) < control_dt:
            time.sleep(0.0001)
        last_time = time.time()

        run_policy(policy, robot, time_step, logger)
        
        # If policy provides reference joint positions (joint_pos), forward them to the simulator viewer as ghosts
        try:
            if hasattr(policy, "output_tensors") and "joint_pos" in policy.output_tensors:
                qref = policy.output_tensors["joint_pos"]
                # qref may be shape (1, n) or (n,); pick first batch row
                import numpy as _np
                qref_arr = _np.asarray(qref)
                if qref_arr.ndim >= 2:
                    qpose_joints = qref_arr.squeeze(0)
                else:
                    qpose_joints = qref_arr
                
                # Expand joint positions to full qpos (base_pos + base_quat + joints)
                # MuJoCo qpos for humanoid: [base_xyz(3), base_quat(4), joints(29)] = 36
                if hasattr(env.simulator, "add_ghost_trajectory"):
                    # Get current base position and orientation from robot state
                    qpose_full = _np.zeros(env.simulator.data.qpos.shape)
                    qpose_full[:7] = env.simulator.data.qpos[:7]  # Copy current base pose
                    qpose_full[7:7+len(qpose_joints)] = qpose_joints  # Set joint positions
                    env.simulator.add_ghost_trajectory(qpose_full)
        except Exception:
            logger_mp.exception("Failed to push ghost pose to simulator viewer")
        
        # Compute and update reward metrics
        try:
            current_joint_pos = robot.joint_pos
            current_joint_vel = robot.joint_vel
            
            # Smoothness: measure joint velocity changes
            joint_vel_change = np.linalg.norm(current_joint_vel - last_joint_vel)
            smoothness = -joint_vel_change  # Negative: smaller change is better
            last_joint_vel = current_joint_vel.copy()
            
            # Tracking errors (if reference available)
            pos_tracking_global = 0.0
            pos_tracking_local = 0.0
            quat_tracking_global = 0.0
            quat_tracking_local = 0.0
            
            if hasattr(policy, "output_tensors") and "joint_pos" in policy.output_tensors:
                ref_joint_pos = policy.output_tensors["joint_pos"].squeeze(0) if hasattr(policy.output_tensors["joint_pos"], 'squeeze') else policy.output_tensors["joint_pos"]
                ref_joint_pos = np.asarray(ref_joint_pos)
                if ref_joint_pos.shape == current_joint_pos.shape:
                    pos_error = np.linalg.norm(current_joint_pos - ref_joint_pos)
                    pos_tracking_global = -pos_error  # Negative error as reward
                    pos_tracking_local = -np.mean(np.abs(current_joint_pos - ref_joint_pos))
            
            # Orientation tracking (simplified: use base orientation if available)
            # For full implementation, would need base_quat from policy and robot
            # Placeholder for now
            quat_tracking_global = 0.0
            quat_tracking_local = 0.0
            
            rewards = {
                "smoothness": smoothness,
                "pos_tracking_global": pos_tracking_global,
                "pos_tracking_local": pos_tracking_local,
                "quat_tracking_global": quat_tracking_global,
                "quat_tracking_local": quat_tracking_local,
            }
            
            if hasattr(env, "simulator"):
                env.simulator.update_reward_plot(rewards)
        except Exception:
            logger_mp.exception("Failed to compute/update rewards")
        
        logger_mp.debug(f"run one step cost time: {time.time()-last_time:.4f} sec")
        time_step += 1
except KeyboardInterrupt:
    logger_mp.info("Control loop interrupted by user.")
except Exception as e:
    logger_mp.error(f"Error during control loop: {e}")
finally:
    robot.stop_communication()
    logger_mp.info("Robot communication stopped.")

    # TODO: save logs and visualize if needed
    nowtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logs_dir = Path("./logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    filename = logs_dir / f"experiment_{nowtime}.npz"
    np.savez(filename, **logger)
    logger_mp.info(f"Logs saved to {filename}")

    # stop the simulator
    env.stop()