#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   visualize.py
@Time    :   2023/08/02 13:45:38
@Author  :   Cao Zhanxiang 
@Version :   1.0
@Contact :   caozx1110@163.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_dof_pos(dof_pos_list):
    for i in range(0, 29):
        fig = plt.figure(figsize=(20, 15))
        plt.plot(dof_pos_list[:, i], label=f"dof_pos_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/step')
        plt.title("dof's curve")
        plt.savefig(f'./data/dof_pos_{i}_curve.png', dpi=200)
        plt.close(fig)

def plot_dof_vel(dof_vel_list):
    for i in range(0, 29):
        fig = plt.figure(figsize=(20, 15))
        plt.plot(dof_vel_list[:, i], label=f"dof_vel_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/step')
        plt.title("dof's vel curve")
        plt.savefig(f'./data/dof_vel_{i}_curve.png', dpi=200)
        plt.close(fig)


def plot_dof_trq(dof_trq_list):
    for i in range(0, 29):
        fig = plt.figure(figsize=(20, 15))
        plt.plot(dof_trq_list[:, i], label=f"dof_trq_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/step')
        plt.title("dof's trq curve")
        plt.savefig(f'./data/dof_trq_{i}_curve.png', dpi=200)  
        plt.close(fig)

def plot_base_lin_vel(base_lin_vel_list):
    plt.figure(figsize=(20, 15))
    for i in range(0, 3):
        plt.subplot(2, 2, i+1)
        plt.plot(base_lin_vel_list[:, i], label=f"base_lin_vel_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/step')
    plt.title("base's lin vel curve")
    plt.savefig('./data/base_lin_vel_curve.png', dpi=200)


def plot_base_ang_vel(base_ang_vel_list):
    plt.figure(figsize=(20, 15))
    for i in range(0, 3):
        plt.subplot(2, 2, i+1)
        plt.plot(base_ang_vel_list[:, i], label=f"base_ang_vel_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/step')
    plt.title("base's ang vel curve")
    plt.savefig('./data/base_ang_vel_curve.png', dpi=200)


def plot_base_euler(base_euler_list):
    plt.figure(figsize=(20, 15))
    for i in range(0, 3):
        plt.subplot(2, 2, i+1)
        plt.plot(base_euler_list[:, i], label=f"base_euler_{i}")
        plt.grid()
        plt.legend()
        plt.xlabel(f't/step')
    plt.title("base's euler curve")
    plt.savefig('./data/base_euler_curve.png', dpi=200)



def plot_command(command_list, base_lin_vel_list, base_ang_vel_list):
    plt.figure(figsize=(20, 15))
        
    plt.subplot(2, 2, 1)
    plt.plot(base_lin_vel_list[:, 0], label=f"base_lin_vel_{0}")
    plt.plot(command_list[:, 0], label=f"command_list_{0}")
    plt.grid()
    plt.legend()
    plt.xlabel(f't/step')

    plt.subplot(2, 2, 2)
    plt.plot(base_lin_vel_list[:, 1], label=f"base_lin_vel_{1}")
    plt.plot(command_list[:, 1], label=f"command_list_{1}")
    plt.grid()
    plt.legend()
    plt.xlabel(f't/step')

    plt.subplot(2, 2, 3)
    plt.plot(base_ang_vel_list[:, 2], label=f"base_lin_vel_{2}")
    plt.plot(command_list[:, 2], label=f"command_list_{2}")
    plt.grid()
    plt.legend()
    plt.xlabel(f't/step')

    plt.savefig('./data/command_curve.png', dpi=200)
    

def plot_base_height(base_height_list):
    plt.figure(figsize=(20, 15))
    plt.plot(base_height_list[:], label=f"base_height")
    plt.grid()
    plt.legend()
    plt.xlabel(f't/step')
    plt.title("base height's curve")
    plt.savefig(f'./data/base_height_curve.png', dpi=200)


def visualize_helper(logger):
    dof_pos_list = np.array(logger['dof_pos'])
    # dof_vel_list = np.array(logger['dof_vel'])
    # dof_trq_list = np.array(logger['dof_trq'])

    # target_dof_pos_list = np.array(logger['target_dof_pos'])
    
    # # base_lin_vel_list = np.array(logger['base_lin_vel'])
    

    # base_ang_vel_list = np.stack(
    #     [logger['base_vel_roll'],
    #     logger['base_vel_pitch'],
    #     logger['base_vel_yaw']], axis=-1)
    
    # base_euler_list = np.stack(
    #     [logger['roll'],
    #     logger['pitch'],
    #     logger['yaw']], axis=-1)


    # # command_list = np.stack(
    # #     [logger['command_x'],
    # #     logger['command_y'],
    # #     logger['command_yaw']], axis=-1
    # # )


    # # base_height_list = np.array(logger['base_height'])



    print(dof_pos_list.shape)
    

    ## 1 step = 0.02, 1s = 50 step
    start_step = 0  ### 4s start
    end_step = 1000 if len(dof_pos_list) > 1000 else len(dof_pos_list)


    plot_dof_pos(dof_pos_list[start_step:end_step])
    
    # plot_dof_vel(dof_vel_list[start_step:end_step])

    # plot_dof_trq(dof_trq_list[start_step:end_step])

    # # plot_base_lin_vel(base_lin_vel_list[start_step:end_step])

    # plot_base_ang_vel(base_ang_vel_list[start_step:end_step])

    # plot_base_euler(base_euler_list[start_step:end_step])

    # # plot_command(command_list[start_step:end_step], base_lin_vel_list[start_step:end_step], base_ang_vel_list[start_step:end_step])

    # # plot_base_height(base_height_list[start_step:end_step])




if __name__ == '__main__':
    pass