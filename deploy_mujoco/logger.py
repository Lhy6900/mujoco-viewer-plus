# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                # print("value.shape:",value.shape)
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes
        print("self.num_episodes:",self.num_episodes)
        print("============================================")

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        log= self.state_log
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break


        # external_force_pos = np.array(log['external_force_pos'])
        # fig_ext_pos1= plt.figure()
        # a = fig_ext_pos1.add_subplot(111)
        # a.plot(time, external_force_pos[:,0], label='ext_force_posx')
        # a.set(xlabel='time [s]')
        # a.legend() 

        # fig_ext_pos2 = plt.figure()
        # a = fig_ext_pos2.add_subplot(111)
        # a.plot(time, external_force_pos[:,1], label='ext_force_posy')
        # a.set(xlabel='time [s]')
        # a.legend() 

        # fig_ext_pos3 = plt.figure()
        # a = fig_ext_pos3.add_subplot(111)
        # a.plot(time, external_force_pos[:,2], label='ext_force_posz')
        # a.set(xlabel='time [s]')
        # a.legend() 

        # external_force = np.array(log['external_force'])
        # fig_actions_adap_2_foot_force_x = plt.figure()
        # a = fig_actions_adap_2_foot_force_x.add_subplot(111)
        # a.plot(time, external_force[:,0], label='ext_force_x')
        # a.set(xlabel='time [s]')
        # a.legend() 

        # fig_actions_adap_2_foot_force_y = plt.figure()
        # a = fig_actions_adap_2_foot_force_y.add_subplot(111)
        # a.plot(time, external_force[:,1], label='ext_force_y')
        # a.set(xlabel='time [s]')
        # a.legend() 

        # fig_actions_adap_2_foot_force_z = plt.figure()
        # a = fig_actions_adap_2_foot_force_z.add_subplot(111)
        # a.plot(time, external_force[:,2], label='ext_force_z')
        # a.set(xlabel='time [s]')
        # a.legend() 





        # base_rpy = np.array(log['base_rpy'])
        # fig_r = plt.figure()
        # a = fig_r.add_subplot(111)
        # a.plot(time, base_rpy[:,0], label='base_roll')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # fig_p = plt.figure()
        # a = fig_p.add_subplot(111)
        # a.plot(time, base_rpy[:,1], label='base_pitch')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # fig_y = plt.figure()
        # a = fig_y.add_subplot(111)
        # a.plot(time, base_rpy[:,2], label='base_yaw')
        # a.set(xlabel='time [s]')
        # a.legend() 

        # leg_phase = np.array(log['leg_phase'])
        # fig_leg_phase = plt.figure()
        # a = fig_leg_phase.add_subplot(111)
        # a.plot(time, leg_phase[:,0], label='left_leg_phase')
        # a.plot(time, leg_phase[:,1], label='right_leg_phase')
        # a.set(xlabel='time [s]')
        # a.legend() 


        # base_height = np.array(log['base_height'])
        # # base_height_est = np.array(log['base_height_est'])
        # fig_height = plt.figure()
        # a = fig_height.add_subplot(111)
        # a.plot(time, base_height[:], label='base_height')
        # # a.plot(time, base_height_est[:,0], label='base_height_est')
        # a.set(xlabel='time [s]')
        # a.legend() 


        # contact_flag = np.array(log['contact_flag'])
        # # contact_est = np.array(log['contact_est'])
        # fig_contact = plt.figure()
        # axs = fig_contact.subplots(2, 2)
        # a = axs[0, 0]
        # a.plot(time, contact_flag[:,0], label='contact0')
        # # a.plot(time, contact_est[:,0], label='est')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # a = axs[0, 1]
        # a.plot(time, contact_flag[:,1], label='1')
        # # a.plot(time, contact_est[:,1], label='est')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # a = axs[1, 0]
        # a.plot(time, contact_flag[:,2], label='1')
        # a.plot(time, contact_est[:,2], label='est')
        # a.plot(time, foot_contact_est[:,2], label='est1')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # a = axs[1, 1]
        # a.plot(time, contact_flag[:,3], label='1')
        # a.plot(time, contact_est[:,3], label='est')
        # a.plot(time, foot_contact_est[:,3], label='est1')
        # a.set(xlabel='time [s]')
        # a.legend() 


        base_vel = np.array(log['base_vel'])
        base_vel_est = np.array(log['base_vel_est'])
        commands = np.array(log['commands'])
        base_ang_vel = np.array(log['base_ang_vel'])
        fig_base_vel = plt.figure()
        axs = fig_base_vel.subplots(2, 2)
        a = axs[0, 0]
        a.plot(time, base_vel[:,0], label='base_vel_x')
        a.plot(time, base_vel_est[:,0], label='base_vel_x_est')
        a.plot(time, commands[:,0], label='cmd')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[0, 1]
        a.plot(time, base_vel[:,1], label='base_vel_y')
        a.plot(time, base_vel_est[:,1], label='base_vel_y_est')
        a.plot(time, commands[:,1], label='cmd')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 0]
        a.plot(time, base_vel[:,2], label='base_vel_z')
        a.plot(time, base_vel_est[:,2], label='base_vel_z_est')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 1]
        a.plot(time, base_ang_vel[:,2], label='base_ang_vel_z')
        a.plot(time, commands[:,2], label='cmd')
        a.set(xlabel='time [s]')
        a.legend() 
  

        dof_pos = np.array(log['dof_pos'])
        actions_dof = np.array(log['actions_dof'])
        fig_dof_pos = plt.figure()
        axs = fig_dof_pos.subplots(2, 2)
        a = axs[0, 0]
        a.plot(time, dof_pos[:,0], label='dof_pos_0')
        a.plot(time, actions_dof[:,0], label='cmd_0')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[0, 1]
        a.plot(time, dof_pos[:,1], label='dof_pos_1')
        a.plot(time, actions_dof[:,1], label='cmd_1')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 0]
        a.plot(time, dof_pos[:,2], label='dof_pos_2')
        a.plot(time, actions_dof[:,2], label='cmd_2')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 1]
        a.plot(time, dof_pos[:,3], label='dof_pos_3')
        a.plot(time, actions_dof[:,3], label='cmd_3')
        a.set(xlabel='time [s]')
        a.legend() 
        
        # fig_dof_pos2 = plt.figure()
        # axs = fig_dof_pos2.subplots(2, 2)
        # a = axs[0,0]
        # a.plot(time, dof_pos[:,4], label='dof_pos_4')
        # a.plot(time, actions_dof[:,4], label='cmd_4')
        # a.set(xlabel='time [s]')
        # a.legend()
        # a = axs[0,1]
        # a.plot(time, dof_pos[:,5], label='dof_pos_5')
        # a.plot(time, actions_dof[:,5], label='cmd_5')
        # a.set(xlabel='time [s]')
        # a.legend()

        # a = axs[1, 0]
        # a.plot(time, dof_pos[:,12], label='dof_pos_12')
        # a.plot(time, actions_dof[:,12], label='cmd_12')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # a = axs[1, 1]
        # a.plot(time, dof_pos[:,13], label='dof_pos_13')
        # a.plot(time, actions_dof[:,13], label='cmd_13')
        # a.set(xlabel='time [s]')
        # a.legend() 

        # fig_dof_pos3 = plt.figure()
        # axs = fig_dof_pos3.subplots(2, 2)
        # a = axs[0,0]
        # a.plot(time, dof_pos[:,14], label='dof_pos_14')
        # a.plot(time, actions_dof[:,14], label='cmd_14')
        # a.set(xlabel='time [s]')
        # a.legend()
        # a = axs[0,1]
        # a.plot(time, dof_pos[:,15], label='dof_pos_15')
        # a.plot(time, actions_dof[:,15], label='cmd_15')
        # a.set(xlabel='time [s]')
        # a.legend()

        # a = axs[1, 0]
        # a.plot(time, dof_pos[:,16], label='dof_pos_16')
        # a.plot(time, actions_dof[:,16], label='cmd_16')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # a = axs[1, 1]
        # a.plot(time, dof_pos[:,17], label='dof_pos_17')
        # a.plot(time, actions_dof[:,17], label='cmd_17')
        # a.set(xlabel='time [s]')
        # a.legend() 


        # fig_dof_pos4 = plt.figure()
        # axs = fig_dof_pos4.subplots(2, 2)
        # a = axs[0,0]
        # a.plot(time, dof_pos[:,18], label='dof_pos_18')
        # a.plot(time, actions_dof[:,18], label='cmd_18')
        # a.set(xlabel='time [s]')
        # a.legend()
        # a = axs[0,1]
        # a.plot(time, dof_pos[:,19], label='dof_pos_19')
        # a.plot(time, actions_dof[:,19], label='cmd_19')
        # a.set(xlabel='time [s]')
        # a.legend()

        # a = axs[1, 0]
        # a.plot(time, dof_pos[:,20], label='dof_pos_20')
        # a.plot(time, actions_dof[:,20], label='cmd_20')
        # a.set(xlabel='time [s]')
        # a.legend() 
        # a = axs[1, 1]
        # a.plot(time, dof_pos[:,21], label='dof_pos_21')
        # a.plot(time, actions_dof[:,21], label='cmd_21')
        # a.set(xlabel='time [s]')
        # a.legend() 


        dof_vel = np.array(log['dof_vel'])
        fig_dof_vel = plt.figure()
        axs = fig_dof_vel.subplots(2, 2)
        a = axs[0, 0]
        a.plot(time, dof_vel[:,0], label='dof_vel_0')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[0, 1]
        a.plot(time, dof_vel[:,1], label='dof_vel_1')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 0]
        a.plot(time, dof_vel[:,2], label='dof_vel_2')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 1]
        a.plot(time, dof_vel[:,3], label='dof_vel_3')
        a.set(xlabel='time [s]')
        a.legend() 
        
        

        dof_trq = np.array(log['dof_trq'])
        fig_dof_trq = plt.figure()
        axs = fig_dof_trq.subplots(2, 2)
        a = axs[0, 0]
        a.plot(time, dof_trq[:,0], label='dof_trq_0')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[0, 1]
        a.plot(time, dof_trq[:,1], label='dof_trq_1')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 0]
        a.plot(time, dof_trq[:,2], label='dof_trq_2')
        a.set(xlabel='time [s]')
        a.legend() 
        a = axs[1, 1]
        a.plot(time, dof_trq[:,3], label='dof_trq_3')
        a.set(xlabel='time [s]')
        a.legend() 
        
        # fig_dof_trq2 = plt.figure()
        # axs = fig_dof_trq2.subplots(2, 1)
        # a = axs[0]
        # a.plot(time, dof_trq[:,4], label='dof_trq_4')
        # a.set(xlabel='time [s]')
        # a.legend()
        # a = axs[1]
        # a.plot(time, dof_trq[:,5], label='dof_trq_5')
        # a.set(xlabel='time [s]')
        # a.legend()


        plt.show()

    def print_rewards(self):
        print("Average rewards per episode:") ###
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()