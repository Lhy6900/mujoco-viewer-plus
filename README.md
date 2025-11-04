# Deploy Robot

## TODO
- [x] modify onnx policy implementation, forward() function, action scale
- [x] make robot class implicitly adjust its joint order according to configuration joint names' order
- [x] fix the mismatch of number of dof between sim and g1 hardware (29 vs 35)
- [x] fix the falling problem in mujoco sim when switching from default to policy

## Installation

- Create a new conda environment or activate an existing environment with `python>=3.8`. For robot learning configuration, we will asume your policy can run properly in this env.
- Install [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) in your conda env.
- Install ONNX and ONNX Runtime by
```bash
pip install onnx onnxruntime
```
- Install this repo by `pip install -e .`

## Usage
- Before actually run code in this repo, please test your communication following instructions [here](https://github.com/unitreerobotics/unitree_sdk2_python#usage)
- Test your onnx model can run properly by
```bash
python scripts/test_onnx_policy.py path/to/model.onnx [--runs 10] [--gpu]
```
- To run a provided model in simulation (mujoco), run
```bash
python scripts/run_robot_sim.py
```