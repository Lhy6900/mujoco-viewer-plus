import time
import numpy as np
from pathlib import Path
import sys
import os

# 确保项目根目录在 Python 路径中
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# multi-process safe logger
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

import logging
_lvl = getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO)
logging.getLogger().setLevel(_lvl)
logger_mp.setLevel(_lvl)

# data logger (writer)
from deploy_robot.utils.writer import SimpleWriter
logger = SimpleWriter()

# Visualization utilities
from deploy_robot.sim.visualization import update_ghost_from_policy, compute_tracking_rewards

# 1. prepare model and robot interface
from deploy_robot.controllers.beyondmimic import make_policy, make_robot, run_policy
import argparse
# 重构为模块，比如viewer_plus，num_envs,model_path等参数,尤其将viewer_plus与num_envs区分开
parser = argparse.ArgumentParser(description="Inspect and benchmark an ONNX model.")
parser.add_argument("--model", type=str, default="model/dalafan_prog.onnx", help="Path to the .onnx model file")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate")
# Choose viewer type: 'native' (default) or 'plus' (ViewerPlus)
parser.add_argument(
    "--viewer",
    choices=["native", "plus"],
    default="plus",
    help="Viewer type to use: 'native' or 'plus' (enables ViewerPlus features)",
)
args = parser.parse_args()

model_path = args.model
if args.num_envs > 1:
    policylist = [make_policy(model_path) for _ in range(args.num_envs)]
    robotlist = [make_robot({
        "joint_names": policy.joint_names,
        "joint_stiffness": policy.joint_stiffness,
        "joint_damping": policy.joint_damping,
        "default_joint_pos": policy.default_joint_positions,
        "dds_domain_id": 1 + i,  # Each robot uses a different domain ID
    }, env='sim') for i, policy in enumerate(policylist)]
    print('multi robot init')
else:
    policy = make_policy(model_path)
    robot = make_robot({
        "joint_names": policy.joint_names,
        "joint_stiffness": policy.joint_stiffness,
        "joint_damping": policy.joint_damping,
        "default_joint_pos": policy.default_joint_positions,
    }, env='sim')
    print('single robot initialized')
# In simulation, we will also need to start the simulator
from deploy_robot.sim.mujoco_env import MujocoEnv
from deploy_robot.sim.config.g1 import G1MujocoConfig
from deploy_robot.sim.mujoco_multienv import MultiMujocoEnv

# Configure viewer based on CLI choice
if args.viewer == "plus":
    # Enable ViewerPlus with ghost and reward plotting
    G1MujocoConfig.viewer_type = "viewer_plus"
    G1MujocoConfig.viewer_plus_config = {
        "enable_ghost": True,
        "enable_reward_plot": True,
        "reward_history_length": 200,
    }
else:
    # Use native viewer (or no special viewer-plus features)
    G1MujocoConfig.viewer_type = "native"

# Set number of environments in config
G1MujocoConfig.num_envs = args.num_envs

env = MultiMujocoEnv(G1MujocoConfig) if args.num_envs > 1 else MujocoEnv(G1MujocoConfig)

# Start env but immediately pause it (pauses physics stepping but keeps DDS alive)
env.start()
logger_mp.info("Environment started, immediately pausing physics for robot initialization...")
if hasattr(env, 'pause'):
    env.pause()
    logger_mp.info("Physics paused, waiting for DDS to publish initial states...")
    # Give DDS time to publish at least a few state messages before starting robot communication
    time.sleep(0.5)
    logger_mp.info("DDS communication channels established, initial states published.")

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
logger_mp.info("Starting robot communications...")
if args.num_envs > 1:
    for i, robot in enumerate(robotlist):
        logger_mp.info(f"Starting communication for robot {i}...")
        robot.start_communication()
        logger_mp.info(f"Robot {i} communication started, initial state received.")
else:
    robot.start_communication()
logger_mp.info("All robot communications started.")

# Give robots a moment to send their initial holding commands before resuming physics
logger_mp.info("Waiting for robots to establish control...")
time.sleep(0.2)

# Now resume the environment simulation
logger_mp.info("Resuming environment simulation...")
if hasattr(env, 'resume'):
    env.resume()

# phase 1: control robot to the default pose
if args.num_envs > 1:
    # 并行地让所有 robot 设置目标姿态
    # 策略：同时发送所有 robot 的命令，而不是串行等待
    logger_mp.info("Setting default posture for all robots in parallel...")
    
    import threading
    def set_posture_thread(robot, robot_id):
        try:
            logger_mp.info(f"Robot {robot_id}: Starting to set default posture...")
            robot.set_default_posture()
            logger_mp.info(f"Robot {robot_id}: Reached default pose.")
        except Exception as e:
            logger_mp.error(f"Robot {robot_id}: Error during set_default_posture: {e}")
    
    # 启动所有线程
    threads = []
    for i, robot in enumerate(robotlist):
        thread = threading.Thread(target=set_posture_thread, args=(robot, i))
        thread.start()
        threads.append(thread)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    logger_mp.info("All robots reached default pose.")

else:
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
for robot in robotlist if args.num_envs > 1 else [robot]:
    robot.update_state()
time_step = 0
# Variables for tracking error computation
if args.num_envs > 1:
    last_joint_vel_list = [np.zeros(len(robot.joint_pos)) for robot in robotlist]
else:   
    last_joint_vel = np.zeros(len(robot.joint_pos))

try:
    logger_mp.info("Starting main control loop...")
    while True:
        if hasattr(env.simulator, 'handle_viewer_input'):
            print('debug: handling viewer input callback:', env.simulator._current_env_idx)
            env.simulator.handle_viewer_input()
        while (time.time()-last_time) < control_dt:
            time.sleep(0.0001)
        last_time = time.time()
        if args.num_envs > 1:
            for i in range(args.num_envs):
                run_policy(policylist[i], robotlist[i], time_step, logger)
        else:
            run_policy(policy, robot, time_step, logger)
        
        # Update ghost visualization from policy output
        try:
            if args.num_envs > 1:
                # For multi-env, use first environment's policy for ghost
                update_ghost_from_policy(env.simulator, policylist[0], robotlist[0])
            else:
                update_ghost_from_policy(env.simulator, policy, robot)
        except Exception:
            logger_mp.exception("Failed to update ghost")
        
        # Compute and update reward metrics
        try:
            if args.num_envs > 1:
                active_robot = robotlist[0]
                active_policy = policylist[0]
                last_joint_vel = last_joint_vel_list[0]
            else:
                active_robot = robot
                active_policy = policy
            
            # Store last velocity for smoothness calculation
            active_robot._last_joint_vel = last_joint_vel
            
            # Compute rewards using utility function
            rewards = compute_tracking_rewards(active_policy, active_robot)
            
            # Update smoothness with actual velocity change
            joint_vel_change = np.linalg.norm(active_robot.joint_vel - last_joint_vel)
            rewards["smoothness"] = -joint_vel_change
            
            # Update last velocity
            if args.num_envs > 1:
                last_joint_vel_list[0] = active_robot.joint_vel.copy()
            else:
                last_joint_vel = active_robot.joint_vel.copy()
            
            # Send to viewer
            if hasattr(env, "simulator"):
                env.simulator.update_reward_plot(rewards)
        except Exception:
            logger_mp.exception("Failed to compute/update rewards")
        
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