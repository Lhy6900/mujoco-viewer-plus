"""Visualization utilities for robot simulation.

This module provides extensible visualization tools including:
- Ghost rendering (reference trajectory visualization)
- Force visualization (planned)
- Contact visualization (planned)
- Reward plotting
- Policy visualization utilities

Design Philosophy:
- Plugin-based architecture for easy extension
- Support for multiple backends (MuJoCo native, Viser, etc.)
- Clean separation of concerns
"""

from .ghost_renderer import GhostRenderer, build_ghost_qpos_from_policy
from .force_renderer import ForceRenderer, ContactRenderer
from .base_renderer import BaseVisualElement, BaseRenderer
from .policy_viz_utils import update_ghost_from_policy, compute_tracking_rewards

__all__ = [
    "GhostRenderer",
    "ForceRenderer",
    "ContactRenderer",
    "BaseVisualElement",
    "BaseRenderer",
    "build_ghost_qpos_from_policy",
    "update_ghost_from_policy",
    "compute_tracking_rewards",
]
