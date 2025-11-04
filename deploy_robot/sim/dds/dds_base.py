# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from abc import ABC, abstractmethod
from deploy_robot.sim.dds.sharedmemorymanager import SharedMemoryManager
from typing import Any

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class DDSObject(ABC):
    def __init__(self):
        self.publishing = False
        self.subscribing = False
        # ensure attributes exist so we can safely clean them up
        self.input_shm: SharedMemoryManager | None = None
        self.output_shm: SharedMemoryManager | None = None
    
    @abstractmethod
    def dds_publisher(self) -> None:
        pass
    @abstractmethod
    def dds_subscriber(self,msg:Any,datatype:str=None) -> None:
        """Process subscribe data"""
        pass
    @abstractmethod
    def setup_subscriber(): 
        """Process hand command"""
        pass
    @abstractmethod
    def setup_publisher():
        """Process hand command"""
        pass

    def setup_shared_memory(
        self,
        input_shm_name: str = None,
        output_shm_name: str = None, 
        input_size: int = 4096,
        output_size: int = 4096,
        inputshm_flag: bool = True,
        outputshm_flag: bool = True
    ):
        """Setup shared memory"""
        if inputshm_flag:
            if input_shm_name:
                self.input_shm = SharedMemoryManager(input_shm_name, input_size)
            else:
                self.input_shm = SharedMemoryManager(size=input_size)
            logger_mp.info("[%s] Input shared memory: %s", self.node_name, self.input_shm.get_name())

        if outputshm_flag:
            if output_shm_name:
                self.output_shm = SharedMemoryManager(output_shm_name, output_size)
            else:
                self.output_shm = SharedMemoryManager(size=output_size)
            logger_mp.info("[%s] Output shared memory: %s", self.node_name, self.output_shm.get_name())

    def stop_communication(self):
        self.publishing = False
        self.subscribing = False
        # proactively cleanup shared memory to avoid resource_tracker leaks
        for attr in ("input_shm", "output_shm"):
            shm = getattr(self, attr, None)
            if shm is not None:
                try:
                    shm.cleanup()
                except Exception:
                    logger_mp.debug("[%s] Cleanup error on %s", getattr(self, "node_name", "DDSObject"), attr)
                finally:
                    setattr(self, attr, None)