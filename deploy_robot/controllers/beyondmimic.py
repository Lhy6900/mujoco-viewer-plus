import numpy as np

from deploy_robot.policies.onnx_policy import OnnxPolicy
from deploy_robot.g1.g1 import G1, G1Config
from deploy_robot.utils.writer import SimpleWriter

# 1. load the model
def make_policy(model_path: str) -> OnnxPolicy:
    policy = OnnxPolicy(model_path, device="cpu")
    return policy

# 2. prepare the policy, check the policy, eval mode...

# 3. setup interface, communication...
def make_robot(config: dict, env: str = 'real') -> G1:
    robot_cfg = G1Config(
        env=env,
        joint_names=config["joint_names"],
        joint_kps=config["joint_stiffness"],
        joint_kds=config["joint_damping"],
        default_joint_pos=config["default_joint_pos"],
    )
    robot = G1(robot_cfg)
    return robot

# 4. run the policy in main loop
def run_policy(policy: OnnxPolicy, robot: G1, time_step: int, logger: SimpleWriter|None=None) -> None:
    """
    Run inference on the ONNX model with the provided input data.

    Args:
        policy (OnnxPolicy).
        robot: The robot interface to interact with.
    Returns:
        list: The outputs from the model inference.
    """
    
    # a. get observation
    robot.update_state()
    # b. get action from policy
    actions = policy.forward(get_obs(policy, robot, time_step))
    # c. execute action
    robot.control(target_joint_pos=actions)
    # d. logging
    if logger is not None:
        logger.log('target_dof_pos',   robot.target_dof_pos)
        logger.log('dof_pos',          robot.dof_pos)  
        logger.log('dof_vel',          robot.dof_vel)  
        logger.log('dof_trq',          robot.dof_trq)

        # logger.log('base_lin_vel',       lin_vel_est)  

        logger.log('base_vel_roll',    robot.base_ang_vel_body[0])  
        logger.log('base_vel_pitch',   robot.base_ang_vel_body[1])  
        logger.log('base_vel_yaw',     robot.base_ang_vel_body[2])  

        logger.log('roll',             robot.base_rpy[0])  
        logger.log('pitch',            robot.base_rpy[1])  
        logger.log('yaw',              robot.base_rpy[2])  
        logger.log('projected_gravity',robot.projected_gravity)
        logger.log('timestep',       time_step)


def get_obs(policy: OnnxPolicy, robot: G1, time_step:int) -> np.ndarray:
    obs = np.zeros((1, policy.get_observation_size()), dtype=np.float32)

    # construct observation with your configuration
    # Observation Names (Original): ['command', 'motion_anchor_ori_b', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']

    # Observation Names (Prog) ['command', 'projected_gravity', 'base_ang_vel', 'joint_pos', 'joint_vel', 'actions']
    # get reference motion from policy
    policy.run_partial_inference(inputs={'obs': obs, 'time_step': np.array([[time_step]], dtype=np.float32)}, output_names=['joint_pos', 'joint_vel'])
    obs[0, 0:29] = policy.output_tensors["joint_pos"].squeeze(0).copy()
    obs[0, 29:58] = policy.output_tensors["joint_vel"].squeeze(0).copy()

    # get current robot state from robot interface
    obs[0, 58:61] = robot.projected_gravity.copy()
    obs[0, 61:64] = robot.base_ang_vel_body.copy()
    obs[0, 64:93] = robot.joint_pos.copy()     # joint positions
    obs[0, 93:122] = robot.joint_vel.copy()     # joint velocities
    obs[0, 122:151] = policy.get_unscaled_last_action().copy()  # previous actions

    return obs