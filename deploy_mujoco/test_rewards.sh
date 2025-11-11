#!/bin/bash
# 测试奖励计算集成
# 运行3个环境，观察奖励输出

cd /home/ubuntu/deploy_mini/deploy_mujoco
source ~/anaconda3/bin/activate mujoco

echo "================================"
echo "测试奖励计算集成"
echo "================================"
echo ""
echo "运行参数: --num_envs=2"
echo "预期效果: 每50步打印一次奖励值"
echo ""
echo "按 Ctrl+C 退出"
echo "================================"
echo ""

python bmwoyaw_multienv.py --num_envs=2
