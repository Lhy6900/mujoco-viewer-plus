# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import time
import threading
from typing import Dict, List, Optional
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from deploy_robot.sim.dds.dds_base import DDSObject

import logging_mp
logger_mp = logging_mp.get_logger(__name__)


class DDSManager:    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DDSManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Init DDSManager"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.publishing_running = False
        self.subscribing_running = False
        
        self.objects: Dict[str, DDSObject] = {}
        
        self.publish_thread: Optional[threading.Thread] = None
        self.subscribe_thread: Optional[threading.Thread] = None
        
        # publish object cache and frequency control (Hz→interval)
        self._pub_list: List[str] = []
        self._pub_interval: Dict[str, float] = {}
        self._pub_next_ts: Dict[str, float] = {}
        self._default_pub_interval: float = 0.01  # 100Hz default

        self.dds_initialized = False
        self._init_dds()
        logger_mp.info("DDSManager initialized")
    
    def _parse_object_name(self, name: str) -> tuple[str, str]:
        """Parse object name"""
        if ':' in name:
            parts = name.split(':', 1)
            return parts[0], parts[1]
        else:
            return "", name
    
    def _init_dds(self) -> bool:
        """Init DDS system"""
        if self.dds_initialized:
            return True
        
        try:
            # NOTE: Do NOT call ChannelFactoryInitialize here!
            # In multi-environment mode, each G1RobotDDS instance will initialize
            # its own ChannelFactory with a specific domain_id.
            # Global initialization here would override those domain_id settings.
            # ChannelFactoryInitialize(1)  # DISABLED for multi-env support
            self.dds_initialized = True
            logger_mp.info("DDS system initialized (skipping global ChannelFactory init)")
            return True
        except Exception:
            logger_mp.exception("DDS system initialization failed")
            return False
    
    def register_object(self, name: str, obj: DDSObject) -> bool:
        """Register DDS object"""
        if name in self.objects:
            logger_mp.warning("Object '%s' already exists", name)
            return False
        
        try:
            category, obj_name = self._parse_object_name(name)
            self.objects[name] = obj
            logger_mp.info("Registered object '%s' (category: %s)", name, category or 'No category')
            self._pub_interval[name] = self._default_pub_interval
            self._pub_next_ts[name] = 0.0
            return True
        except Exception:
            logger_mp.exception("Register object '%s' failed", name)
            return False
    
    def unregister_object(self, name: str) -> bool:
        """Unregister DDS object"""
        if name not in self.objects:
            logger_mp.warning("Object '%s' not found", name)
            return False
        
        obj = self.objects[name]
        obj.publishing = False
        obj.subscribing = False
        
        del self.objects[name]
        self._pub_interval.pop(name, None)
        self._pub_next_ts.pop(name, None)
        if name in self._pub_list:
            self._pub_list.remove(name)
        logger_mp.info("Unregistered object '%s'", name)
        return True
    
    def get_object(self, name: str) -> Optional[DDSObject]:
        """Get specified object"""
        obj = self.objects.get(name)
        if obj is None:
            logger_mp.warning("Object '%s' not found; available: %s", name, list(self.objects.keys()))
            return None
        return obj
    
    def get_objects_by_category(self, category: str) -> Dict[str, DDSObject]:
        """Get all objects by category"""
        result = {}
        for full_name, obj in self.objects.items():
            cat, obj_name = self._parse_object_name(full_name)
            if cat == category:
                result[obj_name] = obj
        return result
    
    def set_publish_rate(self, name: str, hz: float) -> None:
        """Set publish rate (Hz) for a specific object"""
        if name in self.objects and hz > 0:
            self._pub_interval[name] = 1.0 / hz
            # make the next cycle take effect immediately
            self._pub_next_ts[name] = 0.0
            logger_mp.info("Set publish rate for '%s' to %.1f Hz", name, hz)
    
    def set_default_publish_rate(self, hz: float) -> None:
        if hz > 0:
            self._default_pub_interval = 1.0 / hz
            for name in self.objects.keys():
                if name not in self._pub_interval:
                    self._pub_interval[name] = self._default_pub_interval
            logger_mp.info("Default publish rate set to %.1f Hz", hz)

    def _publish_loop(self) -> None:
        """Publish loop thread"""
        logger_mp.debug("Publish loop thread started")
        
        while self.publishing_running:
            try:
                now = time.perf_counter()
                next_due = None
                for name in self._pub_list:
                    obj = self.objects.get(name)
                    if obj is None or not obj.publishing:
                        continue
                    interval = self._pub_interval.get(name, self._default_pub_interval)
                    due = self._pub_next_ts.get(name, 0.0)
                    if now >= due:
                        try:
                            obj.dds_publisher()
                        except Exception:
                            logger_mp.exception("Object '%s' publish failed", name)
                        # schedule next
                        self._pub_next_ts[name] = now + interval
                    # track earliest due
                    nd = self._pub_next_ts.get(name, now + interval)
                    if next_due is None or nd < next_due:
                        next_due = nd
                # dynamic sleep until the nearest due, minimum lower bound
                if next_due is not None:
                    sleep_time = max(0.0002, next_due - time.perf_counter())
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.001)
                
            except Exception:
                logger_mp.exception("Publish loop error")
                time.sleep(0.01)
        
        logger_mp.debug("Publish loop thread stopped")
    
    def start_publishing(self, enable_publish_names: List[str] = None):
        """Start publishing"""
        if self.publishing_running:
            logger_mp.debug("Publishing already running")
            return
        self._pub_list.clear()
        for name, obj in self.objects.items():
            if enable_publish_names is None or name in enable_publish_names:
                obj.setup_publisher()
                obj.publishing = True
                self._pub_list.append(name)
                print('DDS-MANAGER:PUBLISH NAME:', name)
        self.publishing_running = True
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()
        logger_mp.info("Manager started, managing %d publishing objects", len(self._pub_list))

    def stop_publishing(self):
        """Stop publishing"""
        for name, obj in self.objects.items():
            obj.publishing = False
        self.publishing_running = False
    
    def stop_subscribing(self):
        """Stop subscribing"""
        for name, obj in self.objects.items():
            obj.subscribing = False
        self.subscribing_running = False
    
    def start_subscribing(self, enable_subscribe_names: List[str] = None):
        """Start subscribing"""
        if self.subscribing_running:
            logger_mp.debug("Subscribing already running")
            return
        for name, obj in self.objects.items():
            if enable_subscribe_names is None or name in enable_subscribe_names:
                obj.setup_subscriber()
                obj.subscribing = True
        self.subscribing_running = True

    def stop_all_communication(self):
        for name, obj in self.objects.items():
            obj.stop_communication()
        self.publishing_running = False
        self.subscribing_running = False

# global singleton instance
dds_manager = DDSManager()
