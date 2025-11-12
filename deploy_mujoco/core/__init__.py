"""
Core module for multi-environment MuJoCo simulation
核心仿真模块
"""

from .environment import EnvironmentManager
from .policy import PolicyRunner
from .observation import ObservationBuilder
from .controller import pd_control
from .coordinator import SimulationCoordinator

__all__ = [
    'EnvironmentManager',
    'PolicyRunner',
    'ObservationBuilder',
    'pd_control',
    'SimulationCoordinator',
]
