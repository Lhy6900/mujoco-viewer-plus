import mujoco
import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class RewardPlotter:
    """Minimal reward plotter using MjvFigure. Displays reward curves on the side of viewer."""

    def __init__(self, history_length: int = 300):
        self.history_length = history_length
        self._histories: Dict[str, deque] = {}
        self._figures: Dict[str, mujoco.MjvFigure] = {}

    def register_terms(self, term_names: List[str]) -> None:
        """Register reward terms to track and plot."""
        for name in term_names:
            if name in self._histories:
                continue
            self._histories[name] = deque(maxlen=self.history_length)
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            # Set figure properties
            fig.title = name[:40]  # Truncate long names
            fig.xlabel = "Steps"
            # Set initial range: fig.range is 2x2 [[xmin, xmax], [ymin, ymax]]
            fig.range[0][0] = -self.history_length  # x min
            fig.range[0][1] = 0  # x max
            fig.range[1][0] = -0.01  # y min
            fig.range[1][1] = 0.01  # y max
            # Configure line style and colors
            fig.linepnt[0] = 0
            fig.figurergba[:] = [1.0, 1.0, 1.0, 0.3]  # White semi-transparent background (30% opacity)
            fig.panergba[:] = [1.0, 1.0, 1.0, 0.5]    # White semi-transparent panel (curve area, 50% opacity)
            
            # Set line colors - use different colors for different terms
            color_map = {
                "smoothness": [1.0, 0.0, 0.0],              # Red
                "pos_tracking_global": [0.0, 1.0, 0.0],     # Green
                "pos_tracking_local": [0.0, 0.0, 1.0],      # Blue
                "quat_tracking_global": [1.0, 1.0, 0.0],    # Yellow
                "quat_tracking_local": [1.0, 0.0, 1.0],     # Magenta
            }
            line_color = color_map.get(name, [0.5, 0.5, 1.0])  # Default: light blue
            fig.linergb[0][:] = line_color  # Set line color (RGB)
            
            # Optional: set grid and text colors
            fig.gridrgb[:] = [0.7, 0.7, 0.7]  # Gray grid
            fig.textrgb[:] = [0.0, 0.0, 0.0]  # White text
            
            self._figures[name] = fig

    def update(self, rewards: Dict[str, float]) -> None:
        """Update reward histories and figure data."""
        for k, v in rewards.items():
            if k not in self._histories:
                self.register_terms([k])
            
            self._histories[k].append(float(v))
            
            # Update figure data
            hist = list(self._histories[k])
            fig = self._figures[k]
            n = len(hist)
            
            if n == 0:
                continue
            
            # Write data to linedata array (format: x0, y0, x1, y1, ...)
            for i, val in enumerate(hist):
                fig.linedata[0][2 * i] = -n + i  # x: relative to current
                fig.linedata[0][2 * i + 1] = val  # y: value
            
            fig.linepnt[0] = n
            
            # Auto-scale y-axis with some padding
            if n > 0:
                vals = np.array(hist)
                ymin = float(np.min(vals))
                ymax = float(np.max(vals))
                yspan = max(ymax - ymin, 1e-6)
                padding = yspan * 0.1
                fig.range[1][0] = ymin - padding  # y min
                fig.range[1][1] = ymax + padding  # y max

    def get_figures_for_viewer(self, viewport: mujoco.MjrRect) -> List[Tuple]:
        """Compute viewports and return list of (viewport, figure) pairs for viewer.set_figures().
        
        Layout strategy: stack figures vertically on the right side of the screen.
        """
        if not self._figures:
            return []
        
        term_names = list(self._figures.keys())
        num_figs = len(term_names)
        
        # Reserve right 1/3 of screen for plots
        plot_width = viewport.width // 3
        plot_left = viewport.left + (viewport.width - plot_width)
        
        # Divide vertical space evenly (with small gaps)
        gap = 5
        fig_height = (viewport.height - gap * (num_figs + 1)) // num_figs
        fig_height = max(fig_height, 50)  # Minimum height
        
        result = []
        for i, name in enumerate(term_names[:12]):  # Limit to 12 plots max
            vp = mujoco.MjrRect(
                left=plot_left,
                bottom=viewport.bottom + gap + i * (fig_height + gap),
                width=plot_width,
                height=fig_height
            )
            result.append((vp, self._figures[name]))
        
        return result
