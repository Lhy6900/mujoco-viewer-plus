"""
数据加载工具
加载运动参考数据、ONNX 模型等
"""

import numpy as np
import onnx
import onnxruntime


def load_motion_data(npz_path):
    """
    加载运动参考数据，优雅处理文件不存在的情况
    
    Args:
        npz_path: NPZ 文件路径
        
    Returns:
        tuple: (motion_data_dict, is_loaded)
            - motion_data_dict: 包含运动数据的字典，如果加载失败则为 None
            - is_loaded: bool, 是否成功加载
    """
    try:
        data = np.load(npz_path)
        motion_data = {
            'body_pos_w': data['body_pos_w'],
            'body_quat_w': data['body_quat_w'],
            'joint_pos': data['joint_pos'],
            'joint_vel': data['joint_vel'],
        }
        print(f"[数据加载] ✓ 成功加载参考运动数据：{npz_path}")
        return motion_data, True
    except FileNotFoundError:
        print(f"[数据加载] ✗ 未找到参考运动文件：{npz_path}")
        print(f"[数据加载] → Ghost 渲染功能已禁用（适用于非模仿学习策略）")
        return None, False
    except Exception as e:
        print(f"[数据加载] ✗ 加载参考运动数据失败：{e}")
        print(f"[数据加载] → Ghost 渲染功能已禁用")
        return None, False


def load_onnx_model(model_path):
    """
    加载 ONNX 模型
    
    Args:
        model_path: ONNX 模型文件路径
        
    Returns:
        tuple: (onnx_model, onnx_session)
    """
    try:
        onnx_model = onnx.load(model_path)
        onnx_session = onnxruntime.InferenceSession(model_path)
        print(f"[模型加载] ✓ 成功加载 ONNX 模型：{model_path}")
        return onnx_model, onnx_session
    except Exception as e:
        print(f"[模型加载] ✗ 加载 ONNX 模型失败：{e}")
        raise
