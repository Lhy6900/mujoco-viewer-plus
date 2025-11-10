# Project Refactoring Summary

## Date: 2025-11-10

## Changes Made

### 1. New Visualization Module Structure

Created a modular, extensible visualization system:

```
deploy_robot/sim/visualization/
├── __init__.py                  # Public API
├── base_renderer.py             # Abstract base classes
├── ghost_renderer.py            # Ghost rendering
├── force_renderer.py            # Force visualization (placeholder)
├── policy_viz_utils.py          # Policy visualization utilities
└── README.md                    # Documentation
```

### 2. Key Improvements

#### Modularity
- Separated visualization logic from control code
- Each visual element is an independent, toggleable component
- Easy to add new visualization types (forces, contacts, etc.)

#### Clean API
**Before**: 30+ lines of ghost construction code in control loop
```python
# Extract policy outputs
policy_joint_pos = np.asarray(policy.output_tensors["joint_pos"])
# ... many more lines ...
simulator.add_ghost(ghost_qpos)
```

**After**: Single function call
```python
from deploy_robot.sim.visualization import update_ghost_from_policy
update_ghost_from_policy(simulator, policy, robot)
```

#### Extensibility
- Base classes (`BaseVisualElement`, `BaseRenderer`) for easy extension
- Plugin architecture: register new visual elements dynamically
- Prepared for multiple backends (MuJoCo native, Viser, custom)

### 3. Files Modified

#### Core Changes
- `scripts/run_multirobot_sim.py`: Refactored to use new visualization utilities
- `deploy_robot/sim/viewer_plus/viewer_plus.py`: Uses modular renderers (partial migration)

#### New Files
- `deploy_robot/sim/visualization/__init__.py`
- `deploy_robot/sim/visualization/base_renderer.py`
- `deploy_robot/sim/visualization/ghost_renderer.py`
- `deploy_robot/sim/visualization/force_renderer.py`
- `deploy_robot/sim/visualization/policy_viz_utils.py`
- `deploy_robot/sim/visualization/README.md`
- `deploy_robot/sim/visualization/REFACTORING.md` (this file)

### 4. DDS Initialization Fix

Fixed the `'NoneType' object has no attribute 'CreateChannel'` error:

**Problem**: Multiple calls to `ChannelFactoryInitialize()` in the same process returned `None`

**Solution**: Implemented singleton pattern with class-level factory cache

**Files Fixed**:
- `deploy_robot-main-v1.0.0/deploy_robot/g1/g1.py`
- `deploy_robot-main-v1.0.0/deploy_robot/sim/dds/g1_robot_dds.py`

## Future Work

### Short Term
1. Complete ViewerPlus refactoring to fully use GhostRenderer class
2. Remove legacy ghost code from ViewerPlus once migration is complete
3. Add unit tests for visualization components

### Medium Term
1. Implement ForceRenderer for xfrc_applied visualization
2. Implement ContactRenderer for contact point visualization
3. Add trajectory playback with scrubbing controls

### Long Term
1. Add Viser backend support for web-based visualization
2. Support recording and replay of visualization sessions
3. Add interactive debugging tools (pause, step, inspect)

## Migration Guide for Developers

### Adding Ghost Visualization to Your Script

**Old Way**:
```python
# Embedded in control loop - hard to maintain
if hasattr(policy, "output_tensors"):
    policy_joint_pos = np.asarray(policy.output_tensors["joint_pos"])
    # ... 25+ more lines ...
    if hasattr(env.simulator, "add_ghost"):
        env.simulator.add_ghost(ghost_qpos)
```

**New Way**:
```python
# At the top
from deploy_robot.sim.visualization import update_ghost_from_policy

# In control loop
update_ghost_from_policy(env.simulator, policy, robot)
```

### Adding Reward Tracking

**Old Way**:
```python
# Manual reward computation
current_joint_pos = robot.joint_pos
pos_error = np.linalg.norm(current_joint_pos - ref_joint_pos)
# ... more calculations ...
rewards = {"pos_tracking": -pos_error, ...}
env.simulator.update_reward_plot(rewards)
```

**New Way**:
```python
from deploy_robot.sim.visualization import compute_tracking_rewards

# Automatic computation
rewards = compute_tracking_rewards(policy, robot)
env.simulator.update_reward_plot(rewards)
```

### Adding Custom Visual Elements

```python
from deploy_robot.sim.visualization import BaseVisualElement

class MyVisualElement(BaseVisualElement):
    def update(self, data):
        # Update visualization data
        pass
    
    def render(self, viewer_scene, viewer_option):
        # Render to MuJoCo scene
        pass

# Register with viewer
viewer._visual_elements["my_feature"] = MyVisualElement(model, config)
```

## Benefits

1. **Maintainability**: Visualization logic in one place, not scattered across control code
2. **Reusability**: Same utilities work across different scripts
3. **Extensibility**: Easy to add new visualizations without touching existing code
4. **Testability**: Isolated components can be unit tested
5. **Readability**: Control loops are cleaner and focus on control logic

## Breaking Changes

None - all changes are backward compatible. Existing code continues to work,
but new code should use the utilities in `deploy_robot.sim.visualization`.

## Documentation

- See `deploy_robot/sim/visualization/README.md` for detailed usage guide
- Each module has comprehensive docstrings
- Examples in run_multirobot_sim.py show best practices
