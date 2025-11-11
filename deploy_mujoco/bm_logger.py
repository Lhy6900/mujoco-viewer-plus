"""
Beyond Mimic 简化日志记录类
用于记录状态和奖励信息（与原 logger.py 不冲突的简化版本）
"""

import os
import numpy as np
from collections import defaultdict


class BMLogger:
    """Beyond Mimic 日志记录类"""
    
    def __init__(self, dt):
        """
        初始化日志记录器
        
        Args:
            dt: 时间步长
        """
        self.state_log = defaultdict(list)  # 用于存储状态日志
        self.rew_log = defaultdict(list)    # 用于存储奖励日志
        self.dt = dt                        # 时间步长
        self.num_episodes = 0               # 记录episode数量

    def log_state(self, key, value):
        """记录单个状态"""
        self.state_log[key].append(value)

    def log_states(self, data_dict):
        """批量记录状态"""
        for key, value in data_dict.items():
            self.log_state(key, value)

    def log_rewards(self, data_dict, num_episodes):
        """记录奖励"""
        for key, value in data_dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes
        print("num_episodes:", self.num_episodes)
        print("============================================")

    def reset(self):
        """清空日志"""
        self.state_log.clear()
        self.rew_log.clear()

    def print_rewards(self):
        """打印每个episode的平均奖励"""
        print("Average rewards per episode:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def save_logs(self, save_path):
        """保存日志数据到文件"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **self.state_log)
        print(f"日志已保存到: {save_path}")
