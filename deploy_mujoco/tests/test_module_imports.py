#!/usr/bin/env python3
"""
模块导入测试脚本
验证新模块结构是否正确
"""

import sys
import os

# 添加父目录到 Python 路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("模块导入测试")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    # 测试核心模块
    print("\n【1. 核心模块】")
    modules_core = [
        ('core.environment', 'EnvironmentManager'),
        ('core.policy', 'PolicyRunner'),
        ('core.observation', 'ObservationBuilder'),
        ('core.controller', 'pd_control'),
    ]
    
    for module_name, class_name in modules_core:
        try:
            module = __import__(module_name, fromlist=[class_name])
            obj = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name}: {e}")
            fail_count += 1
    
    # 测试可视化模块
    print("\n【2. 可视化模块】")
    modules_vis = [
        ('visualization.ghost', 'GhostRenderer'),
        ('visualization.force_applicator', 'ForceApplicator'),
        ('visualization.force_visualizer', 'ForceVisualizer'),
        ('visualization.reward_calculator', 'compute_rewards'),
        ('visualization.reward_visualizer', 'RewardPlotter'),
    ]
    
    for module_name, class_name in modules_vis:
        try:
            module = __import__(module_name, fromlist=[class_name])
            obj = getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name}: {e}")
            fail_count += 1
    
    # 测试工具模块
    print("\n【3. 工具模块】")
    modules_utils = [
        ('utils.math_utils', 'quat_mul'),
        ('utils.math_utils', 'quat_invmul'),
        ('utils.math_utils', 'get_orientation_2d_from_quat'),
        ('utils.data_loader', 'load_motion_data'),
        ('utils.data_loader', 'load_onnx_model'),
        ('utils.logger', 'BMLogger'),
    ]
    
    for module_name, func_name in modules_utils:
        try:
            module = __import__(module_name, fromlist=[func_name])
            obj = getattr(module, func_name)
            print(f"  ✓ {module_name}.{func_name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {module_name}.{func_name}: {e}")
            fail_count += 1
    
    # 测试配置文件
    print("\n【4. 配置文件】")
    config_items = [
        'MODEL_PATH',
        'POLICY_CONFIG',
        'OBS_CONFIG',
        'REWARD_CONFIG',
        'FORCE_CONFIG',
        'CONTROLLER_CONFIG',
        'VISUALIZATION_CONFIG',
    ]
    
    try:
        import bmconfig
        for item in config_items:
            if hasattr(bmconfig, item):
                print(f"  ✓ bmconfig.{item}")
                success_count += 1
            else:
                print(f"  ✗ bmconfig.{item}: 不存在")
                fail_count += 1
    except Exception as e:
        print(f"  ✗ 导入 bmconfig 失败: {e}")
        fail_count += len(config_items)
    
    # 总结
    print("\n" + "=" * 60)
    print(f"测试完成: {success_count} 成功, {fail_count} 失败")
    print("=" * 60)
    
    return fail_count == 0


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
