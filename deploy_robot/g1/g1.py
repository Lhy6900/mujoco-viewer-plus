import time
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from deploy_robot.common.robotbase import RobotBase, RobotBaseConfig
from deploy_robot.g1.constants import G1_NUM_MOTOR, G1JointIndex, Mode, G1JointNames

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.joystick import Joystick

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

@dataclass
class G1Config(RobotBaseConfig):
    joint_names: list[str]
    joint_kps: list[float]
    joint_kds: list[float]
    default_joint_pos: list[float]
    # default value
    network_interface: str | None = None
    mode: Mode = Mode.PR
    dds_domain_id: int = 1  # DDS domain ID for multi-environment support (sim only)

class G1(RobotBase):
    """G1 29 dof robot interface
    
    Functions include:
    1. configure robot (sim or real, kp kd, mode, default position, joint limits, ...)
    2. start/stop robot communication
    3. start/stop robot control loop
    4. provide computing interface for observation and action execution

    Notes:
    1. To handle the problems incurred by joint order, we define dof_*, q_* like variables to represent in hardware, absolute order.
       Whereas joint_* variables represent in a configured, relative order by G1Config.joint_names. (except for default_joint_pos, which is always in absolute order)
    """
    cfg: G1Config

    base_quat: np.ndarray   # x y z w
    projected_gravity: np.ndarray 
    base_ang_vel_body: np.ndarray 

    target_dof_pos:  np.ndarray     # absolute position
    dof_pos: np.ndarray = np.zeros(G1_NUM_MOTOR)             # absolute position
    dof_vel: np.ndarray = np.zeros(G1_NUM_MOTOR)
    dof_trq: np.ndarray = np.zeros(G1_NUM_MOTOR)
    dof_names: list[str] = G1JointNames
    dof_kp: np.ndarray = np.zeros(G1_NUM_MOTOR)
    dof_kd: np.ndarray = np.zeros(G1_NUM_MOTOR)

    def __init__(self, cfg: G1Config) -> None:
        super().__init__(cfg)

        if self.cfg.env == 'real':
            ChannelFactoryInitialize(0, self.cfg.network_interface)
        elif self.cfg.env == 'sim':
            # In sim mode, DDS is handled by g1_robot_dds.py
            # We don't need to initialize ChannelFactory here
            pass


        # prepare hardware interface components
        self.joystick = Joystick()

        # prepare communication components
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.update_mode_machine_ = False   # whether mode_machine_ is updated from lowstate
        self.mode_machine_ = 5  # for G1: 4 - 23-DoF, 5 - 29-DoF, 6 - 27-DoF (waist lock)   should not be changed arbitrarily
        self.crc = CRC()

        self.parse_config()

    def parse_config(self):
        # This should be parse first since `dof2joint` and `joint2dof` depend on it
        self.joint_names = self.cfg.joint_names
        assert len(self.joint_names) == G1_NUM_MOTOR, "[G1] joint names length must be equal to G1_NUM_MOTOR!"

        self.default_dof_pos = self.joint2dof(np.array(self.cfg.default_joint_pos))
        assert len(self.default_dof_pos) == G1_NUM_MOTOR, "[G1] default position length must be equal to G1_NUM_MOTOR!"

        # prepare control parameters
        self.dof_kp = self.joint2dof(np.array(self.cfg.joint_kps))
        self.dof_kd = self.joint2dof(np.array(self.cfg.joint_kds))
        assert len(self.dof_kp) == G1_NUM_MOTOR, "[G1] kp length must be equal to G1_NUM_MOTOR!"
        assert len(self.dof_kd) == G1_NUM_MOTOR, "[G1] kd length must be equal to G1_NUM_MOTOR!"

    def start_communication(self):
        """
        Start robot communication interface.

        This method performs the following steps:
        1. Closes the motion controller.
        2. Initializes the motion switcher client.
        3. Checks the mode of the motion switcher client and releases the mode if it is already set.
        4. Creates a publisher for the "rt/lowcmd" channel.
        5. Initializes the publisher.
        6. Creates a subscriber for the "rt/lowstate" channel.
        7. Initializes the subscriber with the LowStateHandler callback function.

        Note: The LowStateHandler callback function is responsible for handling the received low state messages.

        Returns:
            None
        """
        start_time = time.time()
        print(f"[{type(self).__name__}] Starting robot communication...")
        # close motion controller
        if self.cfg.env == 'real':
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(5.0)
            self.msc.Init()

            status, result = self.msc.CheckMode()
            while result["name"]:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)

        # create publisher and subscriber using底层 CycloneDDS API
        if self.cfg.env == 'sim':
            from cyclonedds.domain import Domain, DomainParticipant
            from cyclonedds.topic import Topic
            from cyclonedds.pub import DataWriter
            from cyclonedds.sub import DataReader
            from cyclonedds.core import Listener
            from unitree_sdk2py.core.channel_config import ChannelConfigAutoDetermine
            
            # 导入 G1RobotDDS 以共享其 Domain（避免重复创建）
            from deploy_robot.sim.dds.g1_robot_dds import G1RobotDDS
            
            # 使用 G1RobotDDS 已创建的共享 domain
            # 如果还没创建，则创建之
            if not hasattr(G1RobotDDS, '_shared_domain'):
                G1RobotDDS._shared_domain = Domain(0, ChannelConfigAutoDetermine)
                G1RobotDDS._shared_participant = DomainParticipant(0)
            
            self.participant = G1RobotDDS._shared_participant
            
            # 使用环境特定的 topic 名称
            # 根据 dds_domain_id 生成唯一的 topic 名称
            robot_suffix = f"env{self.cfg.dds_domain_id}"
            cmd_topic_name = f"rt/lowcmd_{robot_suffix}"
            state_topic_name = f"rt/lowstate_{robot_suffix}"
            
            # 创建 publisher（用于发送命令）
            cmd_topic = Topic(self.participant, cmd_topic_name, LowCmd_)
            self.lowcmd_publisher_ = DataWriter(self.participant, cmd_topic)
            
            # 创建 subscriber（用于接收状态）
            state_topic = Topic(self.participant, state_topic_name, LowState_)
            self.lowstate_subscriber = DataReader(
                self.participant,
                state_topic,
                listener=Listener(on_data_available=self._on_lowstate_available)
            )

        else:
            # 实体机器人使用原来的 ChannelFactory 方式
            self.lowcmd_publisher_ = self.factory.CreateChannel("rt/lowcmd", LowCmd_)
            self.lowcmd_publisher_.SetWriter()
            
            self.lowstate_subscriber = self.factory.CreateChannel("rt/lowstate", LowState_)
            self.lowstate_subscriber.SetReader(self._lowstate_handler, 10)

        while self.low_state is None:
            time.sleep(0.01)  # Reduced sleep time for faster startup
            logger_mp.warning(f"[{type(self).__name__}] Waiting to subscribe dds...")
        logger_mp.info(f"[{type(self).__name__}] Robot communication started.")

        self.enable_motor = True
        self.enable_control = True
        
        # Immediately send a holding command to prevent free-fall
        # This keeps the robot at its current position with moderate stiffness
        logger_mp.info(f"[{type(self).__name__}] Sending initial holding command...")
        self.update_state()  # Get current state first
        initial_hold_kp = np.full(G1_NUM_MOTOR, 100.0)  # Moderate stiffness
        initial_hold_kd = np.full(G1_NUM_MOTOR, 5.0)    # Moderate damping
        
        # Temporarily override kp/kd for initial hold
        orig_kp = self.dof_kp.copy()
        orig_kd = self.dof_kd.copy()
        self.dof_kp = initial_hold_kp
        self.dof_kd = initial_hold_kd
        
        # Send holding command at current position
        self._send_motor_cmd(target_q=self.dof_pos)
        
        # Restore original gains
        self.dof_kp = orig_kp
        self.dof_kd = orig_kd
        
        end_time = time.time()
        print(f"[{type(self).__name__}] Robot communication startup time: {end_time - start_time:.3f} seconds.")

    def stop_communication(self):
        if self.cfg.env == 'real':
            if self.lowstate_subscriber is not None:
                # self.lowstate_subscriber.Close()
                self.lowstate_subscriber.CloseReader()
            if self.lowcmd_publisher_ is not None:
                # self.lowcmd_publisher_.Close()
                self.lowcmd_publisher_.CloseWriter()
        else:
            # 仿真环境下，CycloneDDS 的 DataReader/DataWriter 会自动清理
            # 但建议显式删除以确保资源释放
            if hasattr(self, 'lowstate_subscriber'):
                del self.lowstate_subscriber
            if hasattr(self, 'lowcmd_publisher_'):
                del self.lowcmd_publisher_
            if hasattr(self, 'participant'):
                del self.participant
            if hasattr(self, 'domain'):
                del self.domain

        self.enable_motor = False
        self.enable_control = False
        logger_mp.info(f"[{type(self).__name__}] Robot communication stopped.")

    def _on_lowstate_available(self, reader):
        """Callback when lowstate data is available (for CycloneDDS API)"""
        try:
            samples = reader.take(N=1)
            if samples:
                for sample in samples:
                    # 简单检查：如果 sample 有 imu_state 属性，就认为是有效的
                    if hasattr(sample, 'imu_state'):
                        self._lowstate_handler(sample)
        except Exception:
            logger_mp.exception("Error in lowstate callback")

    def _lowstate_handler(self, msg: LowState_):
        """
        Handles the low state message received at 500 Hz.
        Args:
            msg (LowState_): The low state message.
        Returns:
            None
        """
        import copy
        if self.low_state is None:
            self.low_state = copy.deepcopy(msg)  # 首次初始化
        else:
            # 每次都深拷贝,确保完全隔离
            self.low_state = copy.deepcopy(msg)

        if not self.update_mode_machine_:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

    def _send_motor_cmd(self, target_q=np.zeros(G1_NUM_MOTOR), target_dq=np.zeros(G1_NUM_MOTOR), target_trq=np.zeros(G1_NUM_MOTOR)):
        """
        Sends motor commands manually to the G1 robot.
        Note: this target position is absolute position. For relative position control, you can call control() function instead.
        Args:
            target_q (numpy.ndarray, optional): Target joint positions. Defaults to np.zeros(G1_NUM_MOTOR).
            target_dq (numpy.ndarray, optional): Target joint velocities. Defaults to np.zeros(G1_NUM_MOTOR).
            target_trq (numpy.ndarray, optional): Target joint torques. Defaults to np.zeros(G1_NUM_MOTOR).
        """
        if self.update_mode_machine_ and self.enable_motor:
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = target_trq[i]
                self.low_cmd.motor_cmd[i].q = target_q[i]
                self.low_cmd.motor_cmd[i].dq = target_dq[i]
                self.low_cmd.motor_cmd[i].kp = self.dof_kp[i]
                self.low_cmd.motor_cmd[i].kd = self.dof_kd[i]
        else:
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode = 0  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0
                self.low_cmd.motor_cmd[i].q = 0
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].kd = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        # 使用小写 write() 方法（CycloneDDS API）
        if self.cfg.env == 'sim':
            self.lowcmd_publisher_.write(self.low_cmd)
        else:
            self.lowcmd_publisher_.Write(self.low_cmd)

    def update_joystick(self):
        """Update joystick state."""
        self.joystick.extract(self.low_state.wireless_remote)

    def update_state(self):
        """Update robot's state manually, includes:
        - IMU
        - Joint encoders
        - Joystick
        """
        ### base 
        q = self.low_state.imu_state.quaternion  # wxyz
        self.base_quat = np.array([q[1], q[2], q[3], q[0]])  # turn to xyzw
        self.base_rpy = np.array(self.low_state.imu_state.rpy)
        self.base_rpy_aligned = np.array([self.base_rpy[0],self.base_rpy[1],0])     # this assumes planar walking
        self.rot_mat_aligned = R.from_euler('xyz', self.base_rpy_aligned).as_matrix()
        self.base_gyro = np.array(self.low_state.imu_state.gyroscope)  # rad/s
        # self.base_acc = np.array(self.low_state.imu_state.accelerometer)  # m/s^2

        ### motor encoder
        self.dof_pos = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)])
        self.dof_vel = np.array([self.low_state.motor_state[i].dq for i in range(G1_NUM_MOTOR)])
        self.dof_trq = np.array([self.low_state.motor_state[i].tau_est for i in range(G1_NUM_MOTOR)])

        ### wireless remote
        self.joystick.extract(self.low_state.wireless_remote)

        ### calc  -----------
        self.rot_mat_aligned_world2base = np.linalg.inv(self.rot_mat_aligned)

        # self.base_ang_vel_body = self.rot_mat_aligned_world2base @ self.base_gyro
        self.base_ang_vel_body = self.base_gyro.copy()  ###  todo

        self.projected_gravity = self.rot_mat_aligned_world2base @ np.array([0, 0, -1.0])

    def set_default_posture(self, tar_dof_pos=None, duration=1.5):
        """
        Sets the default posture of the G1 robot to the specified target degree of freedom positions.
        Parameters:
        tar_dof_pos (numpy.ndarray, optional): The target degree of freedom positions. Defaults to positions set in configs.
        duration (float, optional): The duration over which to achieve the target positions. Defaults to 1.5 seconds.
        Returns:
        None
        """
        if tar_dof_pos is None:
            tar_dof_pos = self.default_dof_pos.copy()

        self.enable_control = False
        kp_backup = self.dof_kp.copy()
        kd_backup = self.dof_kd.copy()
        # standby control gains
        self.dof_kp = np.asarray([ 350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0, 200.0, 200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0 ])
        self.dof_kd = np.asarray([ 5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 ])

        self.update_state()
        control_dt = 0.02
        cur_dof_pos = self.dof_pos
        for i in range(int(duration / control_dt)):
            ratio = np.clip(i * control_dt / duration, 0.0, 1.0)
            dof_pos = cur_dof_pos * (1.0 - ratio) + tar_dof_pos * ratio
            self._send_motor_cmd(target_q=dof_pos)
            time.sleep(control_dt)

        self.dof_kp = kp_backup
        self.dof_kd = kd_backup
        self.enable_control = True

    def control(self, target_joint_pos: np.ndarray):
        """control the robot with relative value.

        Args:
            target_joint_pos (numpy.ndarray): relative position control signal.
        """
        self.target_dof_pos = self.joint2dof(target_joint_pos) + self.default_dof_pos
        self._send_motor_cmd(target_q=self.target_dof_pos)


    def joint2dof(self, joint_array: np.ndarray) -> np.ndarray:
        """Convert joint space array to dof space array.

        Args:
            joint_array (np.ndarray): joint space array.

        Returns:
            np.ndarray: dof space array.
        """
        return np.asarray([joint_array[self.joint_names.index(q)] for q in self.dof_names])
    
    def dof2joint(self, dof_array: np.ndarray) -> np.ndarray:
        """Convert dof space array to joint space array.

        Args:
            dof_array (np.ndarray): dof space array.

        Returns:
            np.ndarray: joint space array.
        """
        return np.asarray([dof_array[self.dof_names.index(joint)] for joint in self.joint_names])

    @property
    def joint_pos(self):
        return self.dof2joint(self.dof_pos - self.default_dof_pos)

    @property
    def joint_vel(self):
        return self.dof2joint(self.dof_vel)