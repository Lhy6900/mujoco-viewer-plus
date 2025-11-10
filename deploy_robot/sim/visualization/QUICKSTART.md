# Visualization System - Quick Start Guide

## Overview

A modular, extensible visualization system for robot simulation with clean APIs and support for multiple rendering backends.

## Installation

No additional installation needed - the visualization module is part of `deploy_robot`.

## Quick Examples

### 1. Basic Ghost Rendering

```python
from deploy_robot.sim.visualization import update_ghost_from_policy

# In your control loop (that's it!)
while True:
    run_policy(policy, robot, time_step, logger)
    update_ghost_from_policy(env.simulator, policy, robot)
    time.sleep(control_dt)
```

### 2. Tracking Rewards

```python
from deploy_robot.sim.visualization import compute_tracking_rewards

# Compute and visualize tracking errors
rewards = compute_tracking_rewards(policy, robot)
env.simulator.update_reward_plot(rewards)
```

### 3. Custom Visualization (Advanced)

```python
from deploy_robot.sim.visualization import GhostRenderer

# Create custom ghost with different color
ghost_config = {"ghost_color": [1.0, 0.5, 0.5, 0.6]}  # Reddish ghost
ghost = GhostRenderer(model, ghost_config)

# Update and render
ghost.update(reference_qpos)
ghost.render(viewer.user_scn, viewer_option)
```

## Features

### Current
- ✅ Ghost rendering (reference trajectory)
- ✅ Reward plotting
- ✅ Multi-environment support
- ✅ Keyboard shortcuts (Ctrl+G, Ctrl+R)
- ✅ Modular architecture

### Planned (Easy to Add)
- ⏳ Force visualization (xfrc_applied)
- ⏳ Contact point rendering
- ⏳ Viser web viewer backend
- ⏳ Trajectory playback

## Architecture

```
User Script (run_multirobot_sim.py)
    ↓ uses
Visualization Utilities (policy_viz_utils.py)
    ↓ uses
Visual Elements (ghost_renderer.py, force_renderer.py, ...)
    ↓ renders to
Viewer Backend (MuJoCo Native, Viser, ...)
```

## Keyboard Shortcuts

- **g** or **Ctrl+G**: Toggle ghost visibility
- **r** or **Ctrl+R**: Toggle reward plots
- **m**: Toggle multi-environment rendering
- **Arrow Keys**: Switch between environments (multi-env mode)

## Adding Your Own Visualization

### Step 1: Create Renderer

```python
# my_viz.py
from deploy_robot.sim.visualization import BaseVisualElement
import mujoco

class MyRenderer(BaseVisualElement):
    def __init__(self, model, config=None):
        super().__init__(model, config)
        self.my_data = None
    
    def update(self, data):
        """Update with new data to visualize"""
        self.my_data = data
    
    def render(self, viewer_scene, viewer_option):
        """Render to MuJoCo viewer"""
        if not self.enabled or self.my_data is None:
            return
        
        # Use mujoco.mjv_addGeoms() or custom geometry
        # to add visual elements to viewer_scene
        pass
```

### Step 2: Register with Viewer

```python
# In your script
from my_viz import MyRenderer

my_renderer = MyRenderer(env.simulator.model)
env.simulator.viewer._visual_elements["my_viz"] = my_renderer

# Update in control loop
my_renderer.update(my_data)
```

### Step 3: Add Helper Function (Optional)

```python
# my_viz_utils.py
def update_my_viz_from_policy(simulator, policy, robot):
    """High-level helper for common use case"""
    my_data = extract_data_from_policy(policy, robot)
    if hasattr(simulator, 'viewer') and 'my_viz' in simulator.viewer._visual_elements:
        simulator.viewer._visual_elements['my_viz'].update(my_data)
```

## Files Reference

| File | Purpose |
|------|---------|
| `base_renderer.py` | Abstract base classes |
| `ghost_renderer.py` | Ghost/reference rendering |
| `force_renderer.py` | Force arrows (TODO) |
| `policy_viz_utils.py` | Policy-specific helpers |
| `README.md` | Detailed documentation |
| `REFACTORING.md` | Migration guide |
| `QUICKSTART.md` | This file |

## Examples in Codebase

See `scripts/run_multirobot_sim.py` for a complete example using:
- `update_ghost_from_policy()`
- `compute_tracking_rewards()`
- Multi-environment support

## Tips

1. **Keep it simple**: Use high-level functions (`update_ghost_from_policy`) for common tasks
2. **Toggle features**: All visual elements can be toggled on/off
3. **Performance**: Visual elements only render when enabled
4. **Debugging**: Add print/logging in `update()` to debug data flow

## Support

- See `deploy_robot/sim/visualization/README.md` for full API documentation
- Check examples in `scripts/run_multirobot_sim.py`
- All modules have detailed docstrings

## What's Next?

After getting comfortable with ghost rendering and rewards:

1. Try adding force visualization (follow pattern in `force_renderer.py`)
2. Experiment with custom colors and render options
3. Contribute your visual elements back to the project!

Happy visualizing! 🎨
