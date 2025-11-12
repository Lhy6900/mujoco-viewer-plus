"""
Visualization module for Ghost, Force, and Rewards
可视化模块：Ghost渲染、外力可视化、奖励图表
"""

from .ghost import GhostRenderer
from .force_applicator import ForceApplicator
from .force_visualizer import ForceVisualizer
from .reward_calculator import compute_rewards, compute_rewards_simple
from .reward_visualizer import RewardPlotter

# 兼容性别名
RewardVisualizer = RewardPlotter

__all__ = [
    'GhostRenderer',
    'ForceApplicator',
    'ForceVisualizer',
    'compute_rewards',
    'compute_rewards_simple',
    'RewardPlotter',
    'RewardVisualizer',  # 别名
]
