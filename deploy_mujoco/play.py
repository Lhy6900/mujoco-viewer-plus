from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
import math


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.push_robots = True
    
    # env_cfg.external_forces.env_external_force_proportion = 1.0


    env_cfg.commands.resampling_time = 20.0

    # env_cfg.commands.ranges.lin_vel_x = [2.0, 2.0]
    # env_cfg.commands.ranges.lin_vel_x = [1.5, 1.5]
    env_cfg.commands.ranges.lin_vel_x = [1.0, 1.0]
    # env_cfg.commands.ranges.lin_vel_x = [-1.0, -1.0]
    # env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    # env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    # env_cfg.commands.ranges.lin_vel_y = [1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    # env_cfg.commands.ranges.ang_vel_yaw = [1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    
    env_cfg.terrain.play = True
    # env_cfg.terrain.play = False


    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    

    _, _= env.reset() 

    obs, _ = env.get_observations()
    obs_history, base_vel = env.get_extra_info()

    # attacker_obs, _, _ = env.get_attacker_infos()

    
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    attacker_policy = ppo_runner.attacker_get_inference_policy(device=env.device)


    # ppo_runner.alg.attacker_ac.to(env.device)
    # attacker_policy = ppo_runner.alg.attacker_ac.act


    
    logger = Logger(env.dt) ### env dt  is  control dt
    
    robot_index = 0 # which robot is used for logging
    joint_index = 9 # which joint is used for logging


    start_state_log = 0
    stop_state_log = 400 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
        

    for i in range(10*int(env.max_episode_length)): ### 10 * 20s
        actions, base_vel_est = policy(obs.detach(), obs_history.detach())
        # attacker_actions = attacker_policy(attacker_obs.detach())
        # obs, _, rews, dones, infos = env.step(actions.detach(), attacker_actions.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        obs_history, base_vel = env.get_extra_info()


        # attacker_obs, _, _ = env.get_attacker_infos()



        if i > start_state_log and i< stop_state_log:
            logger.log_states(
                {

                    # 'external_force': torch.tanh(env.attacker_actions[robot_index, :3]).detach().cpu().numpy(),
                    # 'external_force_pos': torch.tanh(env.attacker_actions[robot_index, 3:]).detach().cpu().numpy(),

                    'base_vel_est':base_vel_est[robot_index,:].detach().cpu().numpy(),
                    # 'base_height_est':base_height_est[robot_index,:].detach().cpu().numpy(),


                    'base_rpy': env.base_rpy[robot_index,:].detach().cpu().numpy(),

                    'leg_phase':env.leg_phase[robot_index,:].detach().cpu().numpy(),

                    'base_height':(env.base_pos[robot_index,2]- env.measured_heights[robot_index,93]).detach().cpu().numpy(),
                    # 'actions_dof':env.actions[robot_index,:].detach().cpu().numpy() * 0.25 + env.default_dof_pos[0].detach().cpu().numpy(),
                    'dof_pos':env.dof_pos[robot_index, :].detach().cpu().numpy(),
                    
                    # 'base_vel_est': base_vel_est[robot_index,:].detach().cpu().numpy(),
                    # 'contact_est': foot_contact_est[robot_index,:].detach().cpu().numpy(),
                    'contact_flag': env.contact_flag[robot_index, :].detach().cpu().numpy(),

                    'actions_dof':env.actions[robot_index,:].detach().cpu().numpy() * 0.5 + env.default_dof_pos[0].detach().cpu().numpy(),
                    # 'actions_dof':env.actions[robot_index,:].detach().cpu().numpy(),

                    'dof_pos':env.dof_pos[robot_index, :].detach().cpu().numpy(),
                    'dof_vel': env.dof_vel[robot_index, :].detach().cpu().numpy(),
                    'dof_trq': env.torques[robot_index, :].detach().cpu().numpy(),

                    'commands':env.commands[robot_index, :].detach().cpu().numpy(),
                    'base_vel': env.base_lin_vel[robot_index, :3].detach().cpu().numpy(),
                    'base_ang_vel': env.base_ang_vel[robot_index, :3].detach().cpu().numpy(),


                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item() ### num  traj
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
