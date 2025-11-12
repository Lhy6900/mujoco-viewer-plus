from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from torch import Tensor
from typing import Tuple, Dict
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from isaacgym.torch_utils import quat_conjugate, quat_mul
from legged_gym.utils.math import wrap_to_pi, quat_apply_yaw
from legged_gym.utils.math import get_euler_xyz__ as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.utils.terrain import Terrain

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.critic_obs_buf = torch.clip(self.critic_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.critic_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def get_extra_info(self):
        return self.obs_history_buf, self.base_lin_vel
            ## 825, 
        # return torch.cat((self.obs_history_buf, self.traj_future), dim=-1), self.base_lin_vel, 
    

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.episode_length_buf += 1 ### num_env
        self.push_interval_counter += 1  ### Scalar
        self.env_traj_index += 1
        # print("self.env_traj_index:",self.env_traj_index)

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        ### in world frame
        self.feet_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,0:3]
        self.feet_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,7:10]
        ### in body frame
        for i in range(self.feet_num):
            self.feet_pos_in_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, self.feet_pos[:,i,:]- self.base_pos)

        ### dof_pos ,  dof_vel
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        ### for imitation learning local frame
        self.keyponts[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.keypoints_indices,0:7]
        for i in range(29):
            self.keyponts[:,i,:3] = quat_rotate_inverse(self.base_quat, self.keyponts[:,i,:3]-self.base_pos)
            self.keyponts[:,i,3:7] = quat_mul(self.keyponts[:,i,3:7], quat_conjugate(self.base_quat))
            
            self.keyponts_delta[:,i,:3] = self.keyponts[:,i,:3] - self.last_keyponts[:,i,:3]
            self.keyponts_delta[:,i,3:7] = quat_mul(self.keyponts[:,i,3:7], quat_conjugate(self.last_keyponts[:,i,3:7]))


        #### 检测到一次接触 就是 接触，  检测到两次没有接触，才是没有接触
        self.contact_flag_last = self.contact_flag
        contact_ = torch.where(self.contact_forces[:, self.feet_indices, 2] >1.0, 1.0, 0.0)
        self.contact_flag = torch.where(torch.logical_or(contact_, self.contact_flag_logical_or), 1.0, 0.0)
        self.contact_flag_logical_or = contact_
        
        ### push robot
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(reset_env_ids)

        ####  self.env_traj_index  every step +1,  dim(env,num)
        self.target_base_pos[:] = self.traj_data[self.env_traj_index,:3]
        self.target_base_quat[:] = self.traj_data[self.env_traj_index,3:7]
        ## body frame, target delta
        self.target_base_pos_delta[:] = quat_rotate_inverse(self.target_base_quat, (self.target_base_pos - self.last_target_base_pos))
        self.target_base_quat_delta[:] = quat_mul(self.target_base_quat, quat_conjugate(self.last_target_base_quat)) 
        
        self.target_dof_pos[:] = self.traj_data[self.env_traj_index,7:36]
        
        self.target_keyponts[:] = self.traj_data[self.env_traj_index,36:].view(self.num_envs, -1, 7)
        for i in range(29):
            self.target_keyponts[:,i,:3] = quat_rotate_inverse(self.target_base_quat, self.target_keyponts[:,i,:3]-self.target_base_pos)
            self.target_keyponts[:,i,3:7] = quat_mul(self.target_keyponts[:,i,3:7], quat_conjugate(self.target_base_quat))
            
            self.target_keyponts_delta[:, i, :3] = self.target_keyponts[:,i,:3] - self.last_target_keyponts[:,i,:3]
            self.target_keyponts_delta[:, i, 3:7] = quat_mul(self.target_keyponts[:,i,3:7], quat_conjugate(self.last_target_keyponts[:,i,3:7]))


        # self.target_base_pos_episodes[torch.arange(self.num_envs), self.episode_length_buf, :] = self.target_base_pos
        # self.base_pos_episodes[torch.arange(self.num_envs), self.episode_length_buf, :] = self.base_pos - self.env_origins
        # delta_length = 100
        # index = torch.where(self.episode_length_buf>delta_length, self.episode_length_buf-delta_length, 0)
        # self.target_base_pos_delta_2s[:] = quat_rotate_inverse(self.target_base_quat, self.target_base_pos - self.target_base_pos_episodes[torch.arange(self.num_envs), index, :])
        # self.base_pos_delta_2s[:] = quat_rotate_inverse(self.base_quat, self.base_pos - self.env_origins - self.base_pos_episodes[torch.arange(self.num_envs), index, :])
                ## body frame
        # self.target_base_pos_error[:] = quat_rotate_inverse(self.base_quat, (self.target_base_pos + self.env_origins - self.base_pos))


        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_target_base_pos[:] = self.target_base_pos[:]
        self.last_target_base_quat[:] = self.target_base_quat[:]
        self.last_target_keyponts[:] = self.target_keyponts[:]        
       
        self.last_base_pos[:] = self.base_pos[:]
        self.last_base_quat[:] = self.base_quat[:]
        self.last_keyponts[:] = self.keyponts[:]

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]



    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # self.reset_buf |= torch.logical_or(torch.abs(self.base_rpy[:,1])>1.0, torch.abs(self.base_rpy[:,0])>0.8)
        self.reset_buf |= (self.base_pos[:,2] < 0.4)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_traj(self, env_ids):
        num_envs = len(env_ids)
        traj_select = torch.randint(0, self.num_traj, size=(num_envs,)).to(self.device) ## env, 1(0~20)
        max_randint = (self.each_traj_length - self.max_episode_length - 10 )[traj_select]  ### env, 1(0~max_length)
        traj_begin_index = self.each_traj_begin_index[traj_select] ### env,1(20,40)
        begin_index_from_traj_begin_index = \
                torch.tensor([torch.randint(low=0, high=int(val), size=(1,)).item() for val in max_randint]).to(self.device) ## env, 
        begin_index = traj_begin_index + begin_index_from_traj_begin_index  ### env,1
        self.env_traj_index[env_ids] = begin_index.long().to(self.device)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # reset buffers
        self.reset_buf[env_ids] = 1
        self.episode_length_buf[env_ids] = 0

        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.contact_flag[env_ids] = 0.
        self.contact_flag_logical_or[env_ids] = 0.
        self.contact_flag_last[env_ids] = 0.

        self.obs_history_buf[env_ids] = 0.

        self.reset_traj(env_ids)
        ###  self.env_traj_index  every step +1,  dim(env,num)
            ### 由于 机器人数据在这里刷新不了，所以 都不更新
        self.last_target_base_pos[env_ids] = self.traj_data[self.env_traj_index[env_ids],:3]
        self.last_target_base_quat[env_ids] = self.traj_data[self.env_traj_index[env_ids],3:7]
        self.last_target_keyponts[env_ids] = self.traj_data[self.env_traj_index[env_ids],36:].view(len(env_ids), -1, 7)
        for i in range(29):
            self.last_target_keyponts[env_ids,i,:3] = quat_rotate_inverse(self.last_target_base_quat[env_ids], self.last_target_keyponts[env_ids,i,:3]-self.last_target_base_pos[env_ids])
            self.last_target_keyponts[env_ids,i,3:7] = quat_mul(self.last_target_keyponts[env_ids,i,3:7], quat_conjugate(self.last_target_base_quat[env_ids]))


        self.last_base_pos[env_ids] = self.last_target_base_pos[env_ids] + self.env_origins[env_ids]
        self.last_base_quat[env_ids] = self.last_target_base_quat[env_ids]
        self.last_keyponts[env_ids] = self.last_target_keyponts[env_ids]


        # reset robot states, TODO , redefined
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)


        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf


    def compute_observations(self):
        ### 129+ 203 = 332
        self.obs_buf = torch.cat((  self.base_ang_vel * self.obs_scales.ang_vel, ### 3
                                    self.projected_gravity, ### 3
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, ### 29
                                    self.dof_vel * self.obs_scales.dof_vel,### 29
                                    self.actions,  ## 29
                                    self.target_base_pos_delta * 100.0,  ## 3
                                    self.target_base_quat_delta,  ## 4
                                    (self.target_dof_pos- self.default_dof_pos)* self.obs_scales.dof_pos,  ## 29
                                    self.target_keyponts_delta.view(self.num_envs, -1), ### 203   包含 quat ，不能乘系数
                                    ),dim=-1)

        ### 332 + 363 = 695
        self.critic_obs_buf = torch.cat((   self.obs_buf,
                                            self.base_lin_vel * self.obs_scales.lin_vel, ## 3
                                            # self.target_base_pos_error * 1.0, ## 3
                                            (self.target_base_pos + self.env_origins - self.base_pos),  ### 3
                                            quat_mul(self.target_base_quat, quat_conjugate(self.base_quat)), ## 4
                                            (self.target_dof_pos - self.dof_pos) , ## 29
                                            (self.target_keyponts - self.keyponts).view(self.num_envs, -1), ### 203
                                            self.rand_mass.unsqueeze(-1), ### 1
                                            self.rand_friction.unsqueeze(-1), ### 1
                                            self.rand_base_com, ### 3
                                            self.motor_strengths, ### 29
                                            self.motor_offsets, ### 29
                                            self.Kp_factors, ### 29
                                            self.Kd_factors, ### 29
                                    ),dim=-1)

        # # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        ### get obs_history  ### 正数， 向后移动,  
        self.obs_history_buf[:,:] = torch.roll(self.obs_history_buf, self.num_obs_step, dims=-1)
        self.obs_history_buf[:,:self.num_obs_step].copy_(self.obs_buf)


        # for i in range(5):
        #     target_base_pos_delta_ = quat_rotate_inverse(self.traj_data[self.env_traj_index+i+1,3:7], \
        #                         (self.traj_data[self.env_traj_index+i+1,:3] - self.traj_data[self.env_traj_index+i,:3])) * 100.0
        #     target_base_quat_delta_ = quat_mul(self.traj_data[self.env_traj_index+i+1,3:7], \
        #                                                 quat_conjugate(self.traj_data[self.env_traj_index+i,3:7])) 
        #     target_dof_pos_default_ = (self.traj_data[self.env_traj_index+i+1,7:36] - self.default_dof_pos)* self.obs_scales.dof_pos
            
        #     self.traj_future[:, i*36: (i+1)*36] = torch.cat((  
        #                                                         target_base_pos_delta_,
        #                                                         target_base_quat_delta_,
        #                                                         target_dof_pos_default_),dim=-1)
 


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='trimesh':
            self._create_trimesh()

        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

#------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        return props, self.friction_coeffs[env_id]

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        if self.cfg.domain_rand.randomize_motor_strength:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]
     
        if self.cfg.domain_rand.randomize_motor_offset:
            rng = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]
     
        if self.cfg.domain_rand.randomize_Kp_factor:
            rng = self.cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]

        if self.cfg.domain_rand.randomize_Kd_factor:
            rng = self.cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_id, :] = torch.rand(self.num_dof, dtype=torch.float, device=self.device,requires_grad=False).unsqueeze(0) \
                                                            * (rng[1] - rng[0]) + rng[0]

                
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            rand_mass = np.random.uniform(rng[0], rng[1])  ### TODO
            props[15].mass += rand_mass

        if self.cfg.domain_rand.randomize_base_com:
            rng = self.cfg.domain_rand.base_com_range
            rand_com_displacements = torch.rand(3, dtype=torch.float, device=self.device,requires_grad=False)\
                                                                        * (rng[1] - rng[0]) + rng[0]
            rand_com_displacements[2] = rand_com_displacements[2] * 1.5
            props[15].com += gymapi.Vec3(rand_com_displacements[0], rand_com_displacements[1],rand_com_displacements[2])

        return props, rand_mass, rand_com_displacements
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        if self.cfg.domain_rand.push_robots and  (self.push_interval_counter % self.push_interval_length == 0):
            self._push_robots()
            self.push_interval_counter = 0

    def _compute_torques(self, actions):
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if self.cfg.domain_rand.randomize_Kp_factor:
                torques = self.p_gains*self.Kp_factors*(actions_scaled + self.default_dof_pos + self.motor_offsets - self.dof_pos)\
                                                    - self.d_gains*self.Kd_factors*self.dof_vel
            else:
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.cfg.domain_rand.randomize_motor_strength:
            torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # print("env_ids.shape:",env_ids.shape)
        # print("self.traj_data.shape:",self.traj_data.shape)
        # print("self.env_traj_index.shape:",self.env_traj_index.shape)
        # print("self.env_traj_index[env_ids].shape:",self.env_traj_index[env_ids].shape)
        # print("self.env_traj_index[env_ids]:",self.env_traj_index[env_ids])
        # print("self.traj_data[self.env_traj_index[env_ids]].shape:",self.traj_data[self.env_traj_index[env_ids]].shape)
        # self.dof_pos[env_ids] = self.traj_data[self.env_traj_index[env_ids]][:,7:] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_pos[env_ids] = self.traj_data[self.env_traj_index[env_ids]][:,7:36]
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, reset_env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[reset_env_ids] = self.base_init_state
        self.root_states[reset_env_ids, :3] += self.env_origins[reset_env_ids]
        # self.root_states[reset_env_ids, :2] += torch_rand_float(-1., 1., (len(reset_env_ids), 2), device=self.device) # xy position within 1m of the center
        # self.root_states[reset_env_ids, :2] += torch_rand_float(-0., 0., (len(reset_env_ids), 2), device=self.device) # xy position within 1m of the center
        self.root_states[reset_env_ids,2:7] = 0.0
        self.root_states[reset_env_ids, :7] += self.traj_data[self.env_traj_index[reset_env_ids]][:,:7]

        # print("self.env_traj_index:",self.env_traj_index)
        # print("self.traj_data[self.env_traj_index[reset_env_ids]][:,:7]:",self.traj_data[self.env_traj_index[reset_env_ids]][:,3:7])
        # print("self.root_states[reset_env_ids, 3:7]:",self.root_states[reset_env_ids, 3:7])
        # base velocities
        # self.root_states[reset_env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(reset_env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        # self.root_states[reset_env_ids, 7:13] = torch_rand_float(-0., 0., (len(reset_env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = reset_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
          
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:35] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[35:64] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[64:] = 0.
        return noise_vec






    def _init_traj_buffer(self):
        print("loading traj data........")
        # self.num_traj = 1  
        self.num_traj = 13
        
        self.each_traj_length = torch.zeros(self.num_traj).to(self.device)
        self.each_traj_begin_index = torch.zeros(self.num_traj).to(self.device)
        # traj_data
        # 存储读取的数据
        for i in range(self.num_traj):
            file_name = str(i)  # 获取随机生成的 file_name
            # file_name = str(i+13)  # 获取随机生成的 file_name
            
            csv_file = LEGGED_GYM_ROOT_DIR+ '/trajectory/' + file_name + '.csv'  # 形成文件路径
            data = np.genfromtxt(csv_file, delimiter=',')   # 从 CSV 文件中加载数据

            self.each_traj_length[i] = data.shape[0]

            if i==0:
                traj_data = np.copy(data)
            else:
                traj_data = np.concatenate((traj_data,data),axis=0)
                
                self.each_traj_begin_index[i] = self.each_traj_begin_index[i-1]+ self.each_traj_length[i-1]

        self.traj_data = torch.from_numpy(traj_data).float().to(self.device)

        print("self.each_traj_length:",self.each_traj_length)
        print("self.each_traj_begin_index:",self.each_traj_begin_index)
        print("self.traj_data.shape:",self.traj_data.shape)

        print("loading traj data finish...")


    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        ### dof_pos ,  dof_vel
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
            ### used to diff dof_acc
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        ### feet_pos,  feet_vel, in  world frame   pos 足端 平地上 有 正的 0.02cm， 偏置 
        self.feet_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,0:3]
        self.feet_vel = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,self.feet_indices,7:10]
        self.feet_num = len(self.feet_indices)

        ### base_pos ,  base_quat, base_euler,   lin_vel,  ang_vel   in world frame
        self.base_pos = self.root_states[:,0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_rpy = get_euler_xyz_in_tensor(self.base_quat)

        ### in body frame
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.feet_pos_in_body_frame = torch.zeros_like(self.feet_pos)
        for i in range(self.feet_num):
            self.feet_pos_in_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, self.feet_pos[:,i,:]- self.base_pos) 

        ###  contact_flag  
        self.contact_flag = torch.zeros(self.num_envs, self.feet_num, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_flag_logical_or = torch.zeros(self.num_envs, self.feet_num, dtype=torch.float, device=self.device, requires_grad=False)
        self.contact_flag_last = torch.zeros(self.num_envs, self.feet_num, dtype=torch.float, device=self.device, requires_grad=False)
        
        ### in rough terrain  foot clearance
        self.foot_clearance = torch.zeros_like(self.feet_pos[...,2])

        ### height_map_points
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()

        # reinit
        self.obs_history_buf = torch.zeros(self.num_envs, self.num_obs_step*self.num_obs_history, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.push_interval_counter = 0
        
        
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        # self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

### --------------  traj  imitation learning  --------------
        self.last_base_pos = torch.zeros_like(self.base_pos)
        self.last_base_quat = torch.zeros_like(self.base_quat)


        self.target_base_pos = torch.zeros_like(self.base_pos)
        self.target_base_quat = torch.zeros_like(self.base_quat)
        self.target_dof_pos = torch.zeros_like(self.dof_pos)

        self.last_target_base_pos = torch.zeros_like(self.base_pos)
        self.last_target_base_quat = torch.zeros_like(self.base_quat) 

        ### body frame
        self.target_base_pos_delta = torch.zeros_like(self.base_pos)
        self.target_base_quat_delta = torch.zeros_like(self.base_quat)

        ### body frame
        self.target_base_pos_error = torch.zeros_like(self.base_pos)

        ### body frame
        self.target_base_pos_episodes = torch.zeros(self.num_envs, 1001, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_pos_episodes = torch.zeros(self.num_envs, 1001, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.target_base_pos_delta_2s = torch.zeros_like(self.base_pos)
        self.base_pos_delta_2s = torch.zeros_like(self.base_pos)

        ### keypoints, local frame
        self.target_keyponts = torch.zeros(self.num_envs, 29, 7, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_target_keyponts = torch.zeros_like(self.target_keyponts)
        self.target_keyponts_delta = torch.zeros_like(self.target_keyponts)
        
        ### local frame
        self.keyponts = torch.zeros(self.num_envs, 29, 7, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_keyponts = torch.zeros_like(self.keyponts)
        self.keyponts_delta = torch.zeros_like(self.keyponts)
        self.keypoints_indices = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], dtype=torch.long, device=self.device, requires_grad=False)

        self.traj_future = torch.zeros(self.num_envs, 36*5, dtype=torch.float, device=self.device)


        self._init_traj_buffer()
        self.env_traj_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)


        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # if name=="termination":
            #     continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        
        self.num_height_points = grid_x.numel()  ### 187= 17*11
        
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        # print("init_height_point:",points[:, :, 0])
        return points

    def _get_heights(self):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw
            Z axis  is  world base
            在机器人脚下的 对齐 yaw 的 高程图
        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        ### quat_apply_yaw 主要考虑 (x,y) 的变化
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size  ### 主要考虑 (x,y) 的变化
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        
        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_foot_clearance(self):
        if self.cfg.terrain.mesh_type == 'plane':
            raise ValueError("in plane ,  this is error")
            # return torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
   
        # self.feet_pos  ### in world frame  (env_num,2,3)
        points = torch.clone(self.feet_pos)
        # print("self.feet_pos:",self.feet_pos)
        points[...,0:2] += self.terrain.cfg.border_size 
        # points[...,2:3] -= 0.02 
        
        ### (env_num,2,3)
        points[...,0:2] = (points[...,0:2]/self.terrain.cfg.horizontal_scale).long()
        px = points[..., 0].long().view(-1) ### env_num*2,0
        py = points[..., 1].long().view(-1) ### env_num*2,1
        heights1 = self.height_samples[px, py] 
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)  ### env*2

        foot_clearance = points[...,2] - heights.view(self.num_envs, 2)*self.terrain.cfg.vertical_scale
        ###foot_clearance  shape  :   [num_envs,2]
        return foot_clearance


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props, friction_coeffs = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, rand_mass, rand_com_displacements = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            ### rand mass, friction, base_com,  motor ,,,
               ## _process_dof_props had process motor rand
            self.rand_friction[i] = friction_coeffs
            self.rand_mass[i] = rand_mass
            self.rand_base_com[i, :] = rand_com_displacements


        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

            if self.cfg.terrain.play:
                self.rows = torch.randint(0, 1, (self.num_envs,), device=self.device)
                self.cols = torch.randint(0, 1, (self.num_envs,), device=self.device)
                self.env_origins[:] = self.terrain_origins[self.rows, self.cols]
            else:
                self.rand_rows = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,), device=self.device)
                self.rand_cols = torch.randint(0, self.cfg.terrain.num_cols, (self.num_envs,), device=self.device)
                self.env_origins[:] = self.terrain_origins[self.rand_rows, self.rand_cols]
        else:
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_length = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):  
        lin_vel_error = torch.sum(torch.square(self.target_base_lin_vel - self.base_lin_vel), dim=-1)
        return torch.exp(-lin_vel_error/0.25)
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.sum(torch.square(self.target_base_ang_vel - self.base_ang_vel), dim=-1)
        return torch.exp(-ang_vel_error/0.25)


    ###----------  used -------------
    def _reward_tracking_base_pos_delta(self):
        base_pos_delta = quat_rotate_inverse(self.base_quat, (self.base_pos - self.last_base_pos))
        error = torch.sum(torch.square(self.target_base_pos_delta - base_pos_delta), dim=-1)
        # print("base_pos_delta_error:",error)
        return torch.exp(-error / 0.01**2)
    

    def _reward_tracking_base_quat_delta(self):
        base_quat_delta = quat_mul(self.base_quat, quat_conjugate(self.last_base_quat))
        error = torch.sum(torch.square(self.target_base_quat_delta - base_quat_delta), dim=-1)
        # print("base_quat_delta_error:",error)
        return torch.exp(-error/0.1**2)


    def _reward_tracking_dof_pos(self):
        dof_pos_error = torch.sum(torch.square(self.target_dof_pos - self.dof_pos), dim=-1)
        # print("dof_pos_error:",dof_pos_error)
        return torch.exp(-dof_pos_error/ 2.0**2)
    

    # def _reward_tracking_base_pos_error(self):
    #     error = torch.sum(torch.square(self.target_base_pos_delta_2s - self.base_pos_delta_2s), dim=-1)
    #     # print("base_pos_error:",error)
    #     return torch.exp(-error / 0.8**2)
   
   
    def _reward_tracking_keypoints_delta(self):
        error = torch.sum(torch.square(self.target_keyponts_delta.view(self.num_envs,-1) - self.keyponts_delta.view(self.num_envs,-1)), dim=-1)
        # print("keypoints_error:",error)
        return torch.exp(-error / 0.5**2)
        
        
    # def _reward_tracking_keypoints(self):
    #     error = torch.sum(torch.square(self.target_keyponts.view(self.num_envs,-1)-self.keyponts.view(self.num_envs,-1)),dim=-1)
    #     # print("keypoints_error:",error)
    #     return torch.exp(-error / 2.0**2)
        
        


### --------  penalty -----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_feet_air_time(self):
        first_contact = (self.feet_air_time > 0.) * self.contact_flag
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        self.feet_air_time *= torch.where(torch.logical_not(self.contact_flag), 1.0, 0.0)
        return rew_airTime
    
    # def _reward_feet_clearance(self):
    #     feet_air = torch.where(self.contact_flag>0., 0.0, 1.0)
    #     feet_vel_norm = torch.norm(self.feet_vel[...,0:2],dim=-1)
    #     feet_air_vel_norm = feet_vel_norm * feet_air
    #     feet_air_z_cost = torch.sum(torch.square(self.foot_clearance - 0.10) * feet_air_vel_norm , dim=-1) 
    #     return feet_air_z_cost

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.15)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.5
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.06) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)

    def _reward_hip_yaw_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[2,8]]), dim=1)
    
    def _reward_hip_roll_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,7]]), dim=1)
    
    def _reward_feet_distance(self):
        distance_error = torch.abs((self.feet_pos_in_body_frame[:,0,1] - self.feet_pos_in_body_frame[:,1,1]) - 0.284) ##0.284
        return distance_error
    
    def _reward_not_fly(self):
        return torch.where(torch.logical_or(self.contact_flag[:,0], self.contact_flag[:,1]), 1.0, 0.0)
    
    
    def _reward_feet_contact_slip(self):
        feet_vel_norm = torch.norm(self.feet_vel, dim=-1)
        return torch.sum(feet_vel_norm * self.contact_flag , dim=-1)
    

    def _reward_waist_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,12:13]), dim=-1)

    def _reward_arm_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]] - self.default_dof_pos[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]]), dim=-1)
        # return torch.sum(torch.square(self.dof_pos[:,[14,15,16,17,18,19,  21,22,23,24,25,26]] - self.default_dof_pos[:,[14,15,16,17,18,19, 21,22,23,24,25,26]]), dim=-1)

    def _reward_ankle_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[4,5, 10,11]] - self.default_dof_pos[:,[4,5, 10,11]]), dim=-1)
   


    def _reward_arm_joint_power(self):
        power = torch.sum(torch.abs(self.dof_vel[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]] * self.torques[:,[13,14,15,16,17,18,19, 20,21,22,23,24,25,26]]), dim=-1) ### F*v= P (power)
        return power    
    

    def _reward_waist_upper_actions(self):
        return torch.sum(torch.square(self.actions[:,12:]), dim=-1)


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        # Penalize changes in actions_smooth
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=1)
    

    def _reward_upper_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,13:] - self.actions[:,13:]), dim=1)
    
    def _reward_upper_action_smoothness(self):
        # Penalize changes in actions_smooth
        return torch.sum(torch.square(self.actions[:,13:] - 2*self.last_actions[:,13:] + self.last_last_actions[:,13:]), dim=1)
    





















### ----------- attacker -------------------------
    def _reward_attacker_alive(self):
        return -1.0
    
    def _reward_attacker_termination(self):
        return self.reset_buf * ~self.time_out_buf
            
    def _reward_attacker_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_attacker_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_attacker_base_height(self):
        base_height = self.root_states[:, 2]
        return torch.square(0.8 - base_height - self.measured_heights[:,93])
    
    def _reward_attacker_rpy(self):
        return torch.abs(self.base_rpy[:,0]) + torch.abs(self.base_rpy[:,1])
