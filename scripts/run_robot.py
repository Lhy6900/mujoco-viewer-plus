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
}, env='real')

# 2. run the robot
robot.start_communication()

# phase 1: control robot to the default pose
try:
    logger_mp.info("Running to default pose...")
    robot.set_default_posture()
    logger_mp.info("Reached default pose.")
except Exception as e:
    logger_mp.error(f"Error during reaching default pose: {e}")
    exit(1)

# phase 2: run the policy loop 
time_step = 0
control_dt = 0.02  # 50 Hz
last_time = time.time()
try:
    logger_mp.info("Press 'A' button to start the main control loop...")
    while True:
        time.sleep(0.2)
        robot.update_state()
        if robot.joystick.A.pressed:
            break

    logger_mp.info("Starting main control loop...")
    while True:
        # safety control
        robot.update_joystick()
        if robot.joystick.B.pressed:
            robot.enable_control = False
            robot.enable_motor = False
            logger_mp.info("Control disabled by B button.")
            break

        while (time.time()-last_time) < control_dt:
            time.sleep(0.0001)
        last_time = time.time()

        run_policy(policy, robot, time_step, logger)
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
    filename = logs_dir / f"real_experiment_{nowtime}.npz"
    np.savez(filename, **logger)
    logger_mp.info(f"Logs saved to {filename}")
