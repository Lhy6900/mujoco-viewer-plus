# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
G1 robot DDS communication class
Handle the state publishing and command receiving of the G1 robot
"""

import numpy as np
from typing import Any, Dict, Optional
from deploy_robot.sim.dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


class G1RobotDDS(DDSObject):
    """G1 robot DDS communication class - singleton pattern"""
    
    def __init__(self, node_name: str = "g1_robot", dds_domain_id: int = 0):
        """Initialize the G1 robot DDS node
        
        Args:
            node_name: Name of the DDS node
            dds_domain_id: DDS domain ID for multi-environment support
        """
        # avoid duplicate initialization
        # if hasattr(self, '_initialized'):
        #     return
            
        super().__init__()
        self.node_name = node_name
        self.dds_domain_id = dds_domain_id
        self.crc = CRC()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self._initialized = True

        # setup the shared memory with unique names per environment
        self.setup_shared_memory(
            input_shm_name=f"isaac_robot_state_{self.node_name}",  # unique per environment
            output_shm_name=f"dds_robot_cmd_{self.node_name}",  # unique per environment
            input_size=3072,
            output_size=3072
        )
        
        logger_mp.info("[%s] G1 robot DDS node initialized with unique shared memory", self.node_name)
    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the G1 robot"""
        try:
            from cyclonedds.domain import Domain, DomainParticipant
            from cyclonedds.topic import Topic
            from cyclonedds.pub import DataWriter
            from unitree_sdk2py.core.channel_config import ChannelConfigAutoDetermine
            
            # 使用共享的 domain 0，但用不同的 topic 名称区分环境
            # CycloneDDS 不允许同一进程多个 Domain 实例
            if not hasattr(G1RobotDDS, '_shared_domain'):
                G1RobotDDS._shared_domain = Domain(0, ChannelConfigAutoDetermine)
                G1RobotDDS._shared_participant = DomainParticipant(0)
            
            self.participant = G1RobotDDS._shared_participant
            
            # 使用带环境 ID 的 topic 名称实现隔离
            # 直接使用 dds_domain_id 来生成唯一的 topic 名称
            topic_name = f"rt/lowstate_env{self.dds_domain_id}"
            topic = Topic(self.participant, topic_name, LowState_)
            self.publisher = DataWriter(self.participant, topic)
            
            logger_mp.info("[%s] State publisher initialized (%s)", 
                          self.node_name, topic_name)
            return True
        except Exception:
            logger_mp.exception("[%s] State publisher initialization failed", self.node_name)
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the G1 robot"""
        try:
            from cyclonedds.topic import Topic
            from cyclonedds.sub import DataReader
            from cyclonedds.core import Listener
            
            # 使用已创建的共享 participant
            # 使用带环境 ID 的 topic 名称实现隔离
            # 直接使用 dds_domain_id 来生成唯一的 topic 名称
            topic_name = f"rt/lowcmd_env{self.dds_domain_id}"
            topic = Topic(self.participant, topic_name, LowCmd_)
            
            # 创建带回调的 DataReader
            self.subscriber = DataReader(
                self.participant, 
                topic,
                listener=Listener(on_data_available=self._on_command_received)
            )
            
            logger_mp.info("[%s] Command subscriber initialized (%s)", 
                          self.node_name, topic_name)
            return True
        except Exception:
            logger_mp.exception("[%s] Command subscriber initialization failed", self.node_name)
            return False
    
    def _on_command_received(self, reader):
        """Callback when command data is available"""
        try:
            # 读取所有可用的样本
            samples = reader.take(N=1)  # 每次读取1个最新样本
            if samples:
                for sample in samples:
                    # 简单检查：如果 sample 有 crc 属性，就认为是有效的
                    if hasattr(sample, 'crc'):
                        self.dds_subscriber(sample, "")
        except Exception:
            logger_mp.exception("[%s] Error in command callback", self.node_name)
    
    def dds_publisher(self) -> Any:
        """Convert Isaac Lab state to DDS message and publish."""
        try:
            data = self.input_shm.read_data()
            if data is None:
                return

            motor_state = self.low_state.motor_state
            imu_state = self.low_state.imu_state

            positions = data.get("joint_positions")
            velocities = data.get("joint_velocities")
            torques = data.get("joint_torques")

            if positions and velocities and torques:
                q_array = np.asarray(positions, dtype=np.float32)
                dq_array = np.asarray(velocities, dtype=np.float32)
                tau_array = np.asarray(torques, dtype=np.float32)
                for i in range(len(q_array)):
                    motor = motor_state[i]
                    motor.q = q_array[i]
                    motor.dq = dq_array[i]
                    motor.tau_est = tau_array[i]

            imu = data.get("imu_data")
            if imu and len(imu) >= 13:
                imu_array = np.asarray(imu, dtype=np.float32)
                imu_state.rpy[:] = imu_array[0:3]
                imu_state.quaternion[:] = imu_array[3:7]
                imu_state.accelerometer[:] = imu_array[7:10]
                imu_state.gyroscope[:] = imu_array[10:13]

            self.low_state.tick += 1
            self.low_state.crc = self.crc.Crc(self.low_state)
            # 使用底层 DDS API 的 write 方法（小写）
            self.publisher.write(self.low_state)

        except Exception:
            logger_mp.exception("[%s] Error processing publish data", self.node_name)

    
    def dds_subscriber(self, msg: LowCmd_, datatype: str = None) -> Dict[str, Any]:
        """Process the subscribe data: convert the DDS command to the Isaac Lab format
        
        Return data format:
        {
            "mode_pr": int,
            "mode_machine": int,
            "motor_cmd": {
                "positions": [29 joint position commands],
                "velocities": [29 joint velocity commands],
                "torques": [29 joint torque commands],
                "kp": [29 position gains],
                "kd": [29 speed gains]
            }
        }
        """
        try:
            # verify the CRC
            if self.crc.Crc(msg) != msg.crc:
                logger_mp.warning("[%s] CRC verification failed", self.node_name)
                return {}
            
            # extract the command data
            num_cmd_motors = len(msg.motor_cmd)
            cmd_data = {
                "mode_pr": int(msg.mode_pr),
                "mode_machine": int(msg.mode_machine),
                "motor_cmd": {
                    "positions": [float(msg.motor_cmd[i].q) for i in range(num_cmd_motors)],
                    "velocities": [float(msg.motor_cmd[i].dq) for i in range(num_cmd_motors)],
                    "torques": [float(msg.motor_cmd[i].tau) for i in range(num_cmd_motors)],
                    "kp": [float(msg.motor_cmd[i].kp) for i in range(num_cmd_motors)],
                    "kd": [float(msg.motor_cmd[i].kd) for i in range(num_cmd_motors)]
                }
            }
            self.output_shm.write_data(cmd_data)
        except Exception:
            logger_mp.exception("[%s] Error processing subscribe data", self.node_name)
            return {}
    
    def get_robot_command(self) -> Optional[Dict[str, Any]]:
        """Get the robot control command
        
        Returns:
            Dict: the robot control command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def write_robot_state(self, joint_positions, joint_velocities, joint_torques, imu_data):
        """Write the robot state to the shared memory
        
        Args:
            joint_positions: the joint position list or torch.Tensor
            joint_velocities: the joint velocity list or torch.Tensor
            joint_torques: the joint torque list or torch.Tensor
            imu_data: the IMU data list or torch.Tensor
        """
        if self.input_shm is None:
            return
        try:
            state_data = {
                "joint_positions": joint_positions.tolist() if hasattr(joint_positions, 'tolist') else joint_positions,
                "joint_velocities": joint_velocities.tolist() if hasattr(joint_velocities, 'tolist') else joint_velocities,
                "joint_torques": joint_torques.tolist() if hasattr(joint_torques, 'tolist') else joint_torques,
                "imu_data": imu_data.tolist() if hasattr(imu_data, 'tolist') else imu_data
            }
            # print(f'G1ROBOTDDS-write_robot_state- node_name={self.node_name}, self.dds_id={self.dds_domain_id},joint_positions={state_data["joint_positions"][:3]}, '
            #       f'joint_velocities={state_data["joint_velocities"][:3]}, joint_torques={state_data["joint_torques"][:3]}, '
            #       f'imu_data={state_data["imu_data"][:3]}')
            self.input_shm.write_data(state_data)
        except Exception:
            logger_mp.exception("[%s] Error writing robot state", self.node_name)