"""
测试 ForceApplicator 的两种模式
"""
import numpy as np
import mujoco
from force_applicator import ForceApplicator

# 简单的测试 XML
xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 .5">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
dlist = [data]

print("=== 测试 1: fixtime 模式 ===")
force_mode_1 = {'stop': 'fixtime'}
force_app_1 = ForceApplicator(model, body_name="box", force_mode=force_mode_1)

# 触发外力
force_app_1.trigger(duration=2.0, force_magnitude=10.0, data_list=dlist)
info = force_app_1.get_force_info()
print(f"激活状态: {info['is_active']}")
print(f"模式: {info['mode']}")
print(f"持续时间: {info['duration']}s")
print(f"剩余时间: {info['remaining_time']:.2f}s")

print("\n=== 测试 2: keeping 模式 ===")
force_mode_2 = {'stop': 'keeping'}
force_app_2 = ForceApplicator(model, body_name="box", force_mode=force_mode_2)

# 第一次触发 - 开启外力
print("\n第一次按 Ctrl+F - 开启外力")
force_app_2.trigger(duration=2.0, force_magnitude=10.0, data_list=dlist)
info = force_app_2.get_force_info()
print(f"激活状态: {info['is_active']}")
print(f"模式: {info['mode']}")
print(f"持续时间: {info['duration']}")
print(f"剩余时间: {info['remaining_time']}")

# 第二次触发 - 关闭外力
print("\n第二次按 Ctrl+F - 关闭外力")
force_app_2.trigger(duration=2.0, force_magnitude=10.0, data_list=dlist)
info = force_app_2.get_force_info()
print(f"激活状态: {info}")

print("\n=== 测试 3: 无效模式 ===")
try:
    force_mode_3 = {'stop': 'invalid'}
    force_app_3 = ForceApplicator(model, body_name="box", force_mode=force_mode_3)
except ValueError as e:
    print(f"捕获到预期的错误: {e}")

print("\n✅ 所有测试通过！")
