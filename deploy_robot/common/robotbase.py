from dataclasses import dataclass

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

@dataclass
class RobotBaseConfig:
    env: str  # 'sim'(mujoco) or 'real'

class RobotBase:
    """Interface with robot
    
    Functions include:
    1. configure robot (sim or real, kp kd, mode, default position, joint limits, ...)
    2. start/stop robot communication
    3. start/stop robot control loop
    4. provide computing interface for observation and action execution
    """
    enable_motor: bool = False      # whether to enable motor torque, allowing program to send motor commands
    enable_control: bool = False    # whether to accept control commands and execute it, allowing policy to control the robot

    def __init__(self, cfg: RobotBaseConfig) -> None:
        logger_mp.info(f"Initialize {type(self).__name__} Interface...")
        self.cfg = cfg

    def start_communication(self):
        """Start robot communication interface"""
        raise NotImplementedError
    
    def stop_communication(self):
        """Stop robot communication interface"""
        raise NotImplementedError
    
    # Originally these methods are designed for multithreading, for now, these are unused.
    # def start_control_loop(self):
    #     """Start robot control loop"""
    #     raise NotImplementedError
    
    # def stop_control_loop(self):
    #     """Stop robot control loop"""
    #     raise NotImplementedError
    