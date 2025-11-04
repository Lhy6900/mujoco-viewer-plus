# log_file_path = "logs/experiment_2025-10-30-13-31-09.npz"
log_file_path = "logs/experiment_2025-10-30-14-32-52.npz"

import numpy as np

data = np.load(log_file_path)

# print all keys in the log file
print("Keys in the log file:")
# for key in data.keys():
#     print(f" - {key}")

obs = data['obs']
print(f"Observation shape: {obs.shape}")
desired_joint_pos = obs[:, 0:29]
desired_joint_vel = obs[:, 29:58]
projected_gravity = obs[:, 58:61]
base_ang_vel = obs[:, 61:64]
joint_pos = obs[:, 64:93]
joint_vel = obs[:, 93:122]
last_action = obs[:, 122:151]

dof_pos = data['dof_pos']
roll, pitch, yaw = data["roll"], data["pitch"], data["yaw"]

t = 0
print(f"desired_joint_pos at time {t}:", desired_joint_pos[t])
print(f"joint_pos at time {t}:", joint_pos[t])
print(f"position error at time {t}:", desired_joint_pos[t] - joint_pos[t])

print(f"projected_gravity at time:", projected_gravity)



