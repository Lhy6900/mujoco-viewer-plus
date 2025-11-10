# Bug Fixes

## External Force Persistence Bug (Multi-Environment)

### Issue
When applying external forces in the viewer (by dragging the robot), the forces would persist indefinitely instead of being applied for just one timestep.

### Root Cause
In `mujoco_multisimulator.py`, the code was copying `xfrc_applied` from the viewer to the environment data but never clearing it:

```python
# Old buggy code
if np.any(self.data.xfrc_applied != 0):
    self.datalist[self._current_env_idx].xfrc_applied[:] = self.data.xfrc_applied[:]
# xfrc_applied never cleared, so forces persist forever!
```

### Solution
Two-part fix:

1. **Clear environment forces at step start**: Each environment's `xfrc_applied` is reset to 0 at the beginning of each decimation step
2. **Clear viewer forces after copying**: After copying viewer forces to environment, clear `self.data.xfrc_applied` to prevent re-application

```python
# Fixed code
for i in range(self.num_envs):
    data = self.datalist[i]
    # Clear forces from previous step
    data.xfrc_applied[:] = 0
    # ... compute control and step physics ...

# After stepping, copy viewer forces if any
if np.any(self.data.xfrc_applied != 0):
    self.datalist[self._current_env_idx].xfrc_applied[:] = self.data.xfrc_applied[:]
    # Clear viewer forces after copying (instantaneous application)
    self.data.xfrc_applied[:] = 0
```

### Behavior After Fix
- External forces from viewer are applied for exactly one physics step
- Forces naturally dissipate based on physics (damping, friction, etc.)
- User can continuously apply forces by holding mouse button (viewer will set xfrc_applied each frame)
- Releasing mouse stops force application immediately

### Testing
1. Start multi-environment simulation: `python scripts/run_multirobot_sim.py --num_envs=2`
2. Click and drag robot in viewer to apply force
3. Release mouse
4. ✅ Robot should stop receiving force immediately (though momentum continues)

### Files Modified
- `deploy_robot-main-v1.0/deploy_robot/sim/mujoco_multisimulator.py`

### Impact
- **Severity**: High (broken force interaction in multi-env mode)
- **Scope**: Only affects multi-environment simulation (`num_envs > 1`)
- **Breaking Changes**: None (fixes broken behavior)
