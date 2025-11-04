import numpy as np
from dataclasses import dataclass

from deploy_robot.g1.g1 import G1, G1Config
from deploy_robot.g1.constants import G1JointIndex, G1JointNames, G1JointArmIndex, G1JointArmNames, Mode
from deploy_robot.g1.constants import is_weak_motor, is_wrist_motor

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

@dataclass
class G1BimanualConfig(G1Config):
    # the difference from G1Config is that only the two arms' joint names are included
    pass

class G1Bimanual(G1):
    """G1 robot interface, only controlling the two arms, 14 dof, with other joints locked programmatically.
    
    Functions include:
    1. configure robot (sim or real, kp kd, mode, default position, joint limits, ...)
    2. start/stop robot communication
    3. start/stop robot control loop
    4. provide computing interface for observation and action execution

    Notes:
    - In this class, dof_* still represent full dof (29), but joint_* are 14-dim arrays, only for the two arms.
    - The locked joints are set to position where the robot is at the begining of communication.
    """
    cfg: G1BimanualConfig
    dof_names = G1JointNames

    # new in this class
    dof_arm_names = G1JointArmNames
    default_dof_kp: np.ndarray = np.zeros(len(G1JointNames))
    default_dof_kd: np.ndarray = np.zeros(len(G1JointNames))

    arm_indices: set[int]

    def __init__(self, cfg: G1BimanualConfig) -> None:
        # construct default kp, kd for all dof, get from unitree code, will be updated by config of arms later
        kp_high = 300.0
        kd_high = 3.0
        kp_low = 80.0
        kd_low = 3.0
        kp_wrist = 40.0
        kd_wrist = 1.5
        self.arm_indices = set(member.value for member in G1JointArmIndex)
        for member in G1JointIndex:
            idx = member.value
            if idx in self.arm_indices:
                if is_wrist_motor(member):
                    self.default_dof_kp[idx] = kp_wrist
                    self.default_dof_kd[idx] = kd_wrist
                else:
                    self.default_dof_kp[idx] = kp_low
                    self.default_dof_kd[idx] = kd_low
            else:
                if is_weak_motor(member):
                    self.default_dof_kp[idx] = kp_low
                    self.default_dof_kd[idx] = kd_low
                else:
                    self.default_dof_kp[idx] = kp_high
                    self.default_dof_kd[idx] = kd_high

        super().__init__(cfg)

    def parse_config(self):
        self.joint_names = self.cfg.joint_names
        assert set(self.joint_names).issubset(set(self.dof_names)), "[G1] joint names must be subset of G1JointNames!"
        assert len(self.default_dof_pos) == len(G1JointArmNames), "[G1] default position length must be equal to number of arm joints!"

        self.default_dof_pos = self.joint2dof(np.array(self.cfg.default_joint_pos), self.dof_pos)
        assert len(self.default_dof_pos) == len(G1JointNames), "[G1] default position length must be equal to number of all joints!"

        # prepare control parameters
        self.dof_kp = self.joint2dof(np.array(self.cfg.joint_kps), self.default_dof_kp)
        self.dof_kd = self.joint2dof(np.array(self.cfg.joint_kds), self.default_dof_kd)
        assert len(self.dof_kp) == len(G1JointNames), "[G1] kp length must be equal to number of all joints!"
        assert len(self.dof_kd) == len(G1JointNames), "[G1] kd length must be equal to number of all joints!"

    def start_communication(self):
        super().start_communication()

        # since we need to lock other joints, we need to update default position after communication is started
        self.update_state()
        for member in G1JointIndex:
            idx = member.value
            if idx not in self.arm_indices:
                self.default_dof_pos[idx] = self.dof_pos[idx]

    def control(self, target_joint_pos: np.ndarray):
        # For now, it is the same as G1's control function, but the input is 14-dim joint pos for two arms
        # TODO: add target joint clip
        self.target_dof_pos = self.joint2dof(target_joint_pos) + self.default_dof_pos   # for other joints, it will send 0 + default_dof_pos
        self._send_motor_cmd(target_q=self.target_dof_pos)

    def ctrl_dual_arm_go_home(self):
        # The same as drive arms to 0
        self.target_dof_pos = self.joint2dof(np.zeros(len(self.dof_arm_names)), self.default_dof_pos)
        super().set_default_posture(self.target_dof_pos)

    def joint2dof(self, joint_array: np.ndarray, default_dof_array: np.ndarray | None = None) -> np.ndarray:
        # This is different from G1's joint2dof because it need to impaint all the dof
        return self.partial2full_dof(super().joint2dof(joint_array), default_dof_array)

    def partial2full_dof(self, partial_dof_array: np.ndarray, default_dof_array: np.ndarray | None = None) -> np.ndarray:
        # Convert 14-dim arm dof array to 29-dim full dof array
        # TODO: optimize it by reducing the call times of index()
        if default_dof_array is None:
            default_dof_array = np.zeros(len(self.dof_names))
        return np.asarray([(partial_dof_array[self.dof_arm_names.index(q)] if q in self.dof_arm_names else default_dof_array[i]) for i, q in enumerate(self.dof_names)])
