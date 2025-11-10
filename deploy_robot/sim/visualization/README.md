# Visualization Architecture

## Overview

The visualization system is designed to be modular, extensible, and support multiple rendering backends (MuJoCo native viewer, Viser, etc.).

## Directory Structure

```
deploy_robot/sim/visualization/
├── __init__.py                  # Public API exports
├── base_renderer.py             # Base classes for renderers and visual elements
├── ghost_renderer.py            # Ghost/reference trajectory rendering
├── force_renderer.py            # Force visualization (TODO)
├── policy_viz_utils.py          # Policy-specific visualization utilities
└── README.md                    # This file
```

## Design Philosophy

### 1. Separation of Concerns
- **Visual Elements**: Independent components (ghost, forces, contacts) that can be toggled on/off
- **Renderers**: Backend-specific implementations (MuJoCo native, Viser)
- **Utilities**: Helper functions to bridge policy outputs with visualization

### 2. Extensibility
- Easy to add new visual elements by extending `BaseVisualElement`
- Easy to add new renderers by extending `BaseRenderer`
- Minimal changes to existing code when adding features

### 3. Clean API
- Simple high-level functions for common tasks
- Encapsulated logic keeps user code clean
- Optional features don't clutter the main control loop

## Usage Examples

### Basic Ghost Rendering

```python
from deploy_robot.sim.visualization import update_ghost_from_policy

# In your control loop
update_ghost_from_policy(simulator, policy, robot)
```

### Computing Tracking Rewards

```python
from deploy_robot.sim.visualization import compute_tracking_rewards

rewards = compute_tracking_rewards(policy, robot)
simulator.update_reward_plot(rewards)
```

### Direct Ghost Construction (Advanced)

```python
from deploy_robot.sim.visualization import build_ghost_qpos_from_policy

ghost_qpos = build_ghost_qpos_from_policy(
    policy_output=policy.output_tensors,
    robot=robot,
    current_qpos=simulator.data.qpos
)
simulator.add_ghost(ghost_qpos)
```

## Adding New Visual Elements

To add a new type of visualization (e.g., force arrows, contact points):

1. Create a new renderer class extending `BaseVisualElement`:

```python
from deploy_robot.sim.visualization.base_renderer import BaseVisualElement

class MyRenderer(BaseVisualElement):
    def __init__(self, model, config=None):
        super().__init__(model, config)
        # Your initialization
    
    def update(self, data):
        # Update visualization data
        pass
    
    def render(self, viewer_scene, viewer_option):
        # Render to viewer scene
        pass
```

2. Register it in ViewerPlus or your custom renderer:

```python
my_renderer = MyRenderer(model, config)
viewer._visual_elements["my_feature"] = my_renderer
```

3. Add utility functions in a separate module if needed

## Future Extensions

### Planned Features
- **Force Visualization**: Display xfrc_applied as arrows
- **Contact Visualization**: Show contact points and normal forces
- **Viser Backend**: Web-based 3D visualization
- **Trajectory Playback**: Replay recorded trajectories with scrubbing

### Backend Support
The architecture is designed to support multiple backends:
- **MuJoCo Native** (current): Fast, local rendering
- **Viser** (planned): Web-based, remote viewing
- **Custom** (future): Your own rendering solution

## Migration Guide

If you have existing code with embedded visualization logic:

### Before
```python
# Embedded in control loop
policy_joint_pos = np.asarray(policy.output_tensors["joint_pos"])
policy_joint_pos = policy_joint_pos.squeeze(0)
policy_body_pos_w = np.asarray(policy.output_tensors["body_pos_w"])
policy_body_quat_w = np.asarray(policy.output_tensors["body_quat_w"])
base_pos_policy = policy_body_pos_w[0, 0, :]
base_quat_policy = policy_body_quat_w[0, 0, :]
ghost_qpos = np.zeros_like(current_qpos)
ghost_qpos[0:3] = base_pos_policy
ghost_qpos[3:7] = base_quat_policy
ghost_joint_pos_dof = robot.joint2dof(policy_joint_pos)
ghost_qpos[7:7+len(ghost_joint_pos_dof)] = ghost_joint_pos_dof
simulator.add_ghost(ghost_qpos)
```

### After
```python
# Clean one-liner
from deploy_robot.sim.visualization import update_ghost_from_policy
update_ghost_from_policy(simulator, policy, robot)
```

## API Reference

### High-Level Functions

- `update_ghost_from_policy(simulator, policy, robot)`: Update ghost from policy output
- `compute_tracking_rewards(policy, robot)`: Compute tracking error rewards

### Low-Level Classes

- `BaseVisualElement`: Base class for visual elements
- `BaseRenderer`: Base class for rendering backends
- `GhostRenderer`: Ghost rendering implementation
- `ForceRenderer`: Force visualization (TODO)
- `ContactRenderer`: Contact visualization (TODO)

### Utility Functions

- `build_ghost_qpos_from_policy(policy_output, robot, current_qpos)`: Construct ghost qpos

## Contributing

When adding new visualization features:

1. Keep visual elements independent and toggleable
2. Follow the BaseVisualElement interface
3. Add utility functions for common use cases
4. Document with docstrings and examples
5. Consider multi-backend support from the start
