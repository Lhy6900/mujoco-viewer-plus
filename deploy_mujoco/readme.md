logger是工具
legged_robot.py是啥我也不知道好像没用到
deploy_mujoco.py是最原始的，需要python deploy_mujoco.py g1.yaml来使用。可以选择是否施加阶段性外力
compliance_test 用法同上，需要加g1.yaml，可以选择指定body施加弹簧力。
还有一个pbhc.py,不需要考虑g1.yaml，不指定外力，actor_obs和pbhc完全对齐。网络也和那边完全对齐。
1. 先把obs搞定，再把输入onnx和模块的事情搞定。

