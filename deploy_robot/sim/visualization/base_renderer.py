"""Base classes for visualization elements.

This module defines the interface for all visualization elements,
making it easy to add new types of visualizations in the future.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import mujoco


class BaseVisualElement(ABC):
    """Base class for all visual elements (ghost, forces, contacts, etc.)."""
    
    def __init__(self, model: mujoco.MjModel, config: Optional[dict] = None):
        """Initialize the visual element.
        
        Args:
            model: MuJoCo model
            config: Optional configuration dictionary
        """
        self.model = model
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    def update(self, data: Any) -> None:
        """Update the visual element with new data.
        
        Args:
            data: Data to update the visualization (type depends on element)
        """
        pass
    
    @abstractmethod
    def render(self, viewer_scene: Any, viewer_option: Any) -> None:
        """Render the visual element.
        
        Args:
            viewer_scene: MuJoCo viewer scene (mjvScene)
            viewer_option: MuJoCo viewer options (mjvOption)
        """
        pass
    
    def toggle(self) -> None:
        """Toggle visibility of this element."""
        self.enabled = not self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Set visibility state."""
        self.enabled = enabled


class BaseRenderer(ABC):
    """Base class for different rendering backends (MuJoCo native, Viser, etc.)."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Optional[dict] = None):
        self.model = model
        self.data = data
        self.config = config or {}
        self.visual_elements = {}
    
    def register_element(self, name: str, element: BaseVisualElement) -> None:
        """Register a visual element."""
        self.visual_elements[name] = element
    
    def unregister_element(self, name: str) -> None:
        """Unregister a visual element."""
        if name in self.visual_elements:
            del self.visual_elements[name]
    
    def get_element(self, name: str) -> Optional[BaseVisualElement]:
        """Get a visual element by name."""
        return self.visual_elements.get(name)
    
    @abstractmethod
    def sync(self) -> None:
        """Synchronize and render all visual elements."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if renderer is still running."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the renderer."""
        pass
