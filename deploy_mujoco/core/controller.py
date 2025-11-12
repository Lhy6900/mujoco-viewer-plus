"""
控制器模块
PD 控制等低级控制逻辑
"""

import numpy as np


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """
    PD 控制器
    
    Args:
        target_q: 目标位置
        q: 当前位置
        kp: 比例增益（位置）
        target_dq: 目标速度
        dq: 当前速度
        kd: 微分增益（速度）
        
    Returns:
        tau: 控制力矩
    """
    return (target_q - q) * kp + (target_dq - dq) * kd
