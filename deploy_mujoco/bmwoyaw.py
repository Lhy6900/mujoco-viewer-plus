#!/usr/bin/env python3
"""
多环境 MuJoCo 仿真 - 精简入口
使用模块化框架运行多环境仿真、策略推理和可视化

用法:
    python bmwoyaw_multienv.py --num_envs=4
"""

import argparse
from bmconfig import *
from core.coordinator import SimulationCoordinator


def main():
    """主入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多环境 MuJoCo 仿真')
    parser.add_argument('--num_envs', type=int, default=DEFAULT_NUM_ENVS, 
                        help=f'并行环境数量 (默认: {DEFAULT_NUM_ENVS})')
    args = parser.parse_args()
    
    print("=" * 60)
    print("多环境 MuJoCo 仿真系统")
    print("=" * 60)
    print(f"环境数量: {args.num_envs}")
    print("=" * 60)
    
    # 构造配置字典
    config = {
        # 模型路径
        'model_path': MODEL_PATH,
        'xml_path': XML_PATH,
        'motion_ref_path': MOTION_REF_PATH,
        
        # 仿真参数
        'num_envs': args.num_envs,
        'env_spacing': 3.0,
        'control_decimation': CONTROL_DECIMATION,
        'simulation_dt': SIMULATION_DT,
        
        # 策略配置
        'policy_type': 'imitation',  # 'imitation' | 'custom'
        
        # 观测配置
        'obs_config': {
            'type': 'imitation',
            'dim': NUM_OBS,
        },
        'num_obs': NUM_OBS,
        
        # 关节配置
        'joint_xml': JOINT_XML,
        
        # 外力配置
        'force_config': {
            'anchor_body': FORCE_CONFIG['anchor_body'],
            'mode': {
                'stop': 'keeping',    # 'fixtime' | 'keeping'
                'style': 'spring',    # 'constant' | 'spring'
                'select': 0           # None | 0 | 1 | [0, 2, 3]
            }
        },
        
        # 日志配置
        'logger_dt': LOGGER_DT,
    }
    
    # 创建并运行仿真协调器
    try:
        coordinator = SimulationCoordinator(config)
        coordinator.run()
    except KeyboardInterrupt:
        print("\n[主程序] 用户中断仿真")
    except Exception as e:
        print(f"\n[错误] 仿真异常: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("[主程序] 仿真正常结束")
    return 0


if __name__ == "__main__":
    exit(main())
