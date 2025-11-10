from deploy_robot.g1.g1 import G1JointNames
from deploy_robot.sim.mujoco_simulator import MujocoSimulatorConfig, JointConfig, HardwareSimConfig

G1MujocoConfig = MujocoSimulatorConfig(
    xml_path="deploy_robot/assets/mjcf/g1.xml",
    robot_type="g1",
    headless=False,
    decimation=10,
    num_envs=1,
    dt=0.002,
    root_body_name="pelvis",
    joint_config=JointConfig(
        joints_order = G1JointNames,     # To be same as real hardware
        # This version 's file have the right default position for G1
        # default_positions=[ -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        #                 -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        #                 0.0, 0.0, 0.0,
        #                 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        #                 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0 ]
    ),
    # hardware_sim=HardwareSimConfig(),
)
    