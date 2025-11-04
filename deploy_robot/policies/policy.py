from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Policy(ABC):
    def __init__(self):
        self._joint_names: List[str] = []
        self._joint_stiffness: np.ndarray = np.array([])
        self._joint_damping: np.ndarray = np.array([])
        self._default_joint_positions: np.ndarray = np.array([])

        self._command_names: List[str] = []
        self._observation_names: List[str] = []
        self._action_scale: np.ndarray = np.array([])

    @abstractmethod
    def get_observation_size(self) -> int:
        pass

    @abstractmethod
    def get_action_size(self) -> int:
        pass

    # @abstractmethod
    # def init(self):
    #     pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def forward(self, observations: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_last_action(self) -> np.ndarray:
        pass

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @joint_names.setter
    def joint_names(self, value: List[str]):
        self._joint_names = value

    @property
    def joint_stiffness(self) -> np.ndarray:
        return self._joint_stiffness

    @joint_stiffness.setter
    def joint_stiffness(self, value: np.ndarray):
        self._joint_stiffness = value

    @property
    def joint_damping(self) -> np.ndarray:
        return self._joint_damping

    @joint_damping.setter
    def joint_damping(self, value: np.ndarray):
        self._joint_damping = value

    @property
    def default_joint_positions(self) -> np.ndarray:
        return self._default_joint_positions

    @default_joint_positions.setter
    def default_joint_positions(self, value: np.ndarray):
        self._default_joint_positions = value

    @property
    def command_names(self) -> List[str]:
        return self._command_names

    @command_names.setter
    def command_names(self, value: List[str]):
        self._command_names = value

    @property
    def observation_names(self) -> List[str]:
        return self._observation_names

    @observation_names.setter
    def observation_names(self, value: List[str]):
        self._observation_names = value

    @property
    def action_scale(self) -> np.ndarray:
        return self._action_scale

    @action_scale.setter
    def action_scale(self, value: np.ndarray):
        self._action_scale = value