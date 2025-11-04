# test_quat_to_euler.py
import numpy as np
from scipy.spatial.transform import Rotation as R

# copy of your implementation (expects q = [w, x, y, z])
def quat_to_euler_wxyz(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def angle_diff(a, b):
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d

def compare_quat(q_wxyz):
    # scipy expects [x,y,z,w]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    r = R.from_quat(q_xyzw)
    euler_scipy = r.as_euler('xyz', degrees=False)
    euler_impl = quat_to_euler_wxyz(q_wxyz)
    diff = angle_diff(euler_impl, euler_scipy)
    return np.max(np.abs(diff)), euler_impl, euler_scipy

# Tests
tests = []

# identity
tests.append(np.array([1.0, 0.0, 0.0, 0.0]))

# 90 deg around X (roll)
angle = np.pi / 2
qx = np.array([np.cos(angle/2), np.sin(angle/2), 0.0, 0.0])  # w,x,y,z
tests.append(qx)

# 90 deg around Y (pitch)
qy = np.array([np.cos(angle/2), 0.0, np.sin(angle/2), 0.0])
tests.append(qy)

# 90 deg around Z (yaw)
qz = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
tests.append(qz)

# gimbal lock pitch = +90
# create from euler to ensure exact gimbal lock case
r_gl = R.from_euler('xyz', [0.0, np.pi/2, 0.0])
q_xyzw = r_gl.as_quat()                     # [x,y,z,w]
q_gl = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
tests.append(q_gl)

# randoms
for _ in range(10):
    rrand = R.random()
    q_xyzw = rrand.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    tests.append(q_wxyz)

# run
for i, q in enumerate(tests):
    max_err, e_impl, e_scipy = compare_quat(q)
    print(f"Test {i}: max angle diff = {max_err:.3e}")
    print(f"  quat(wxyz) = {q}")
    print(f"  impl euler = {e_impl}")
    print(f"  scipy euler = {e_scipy}")
    if max_err > 1e-6:
        print("  WARNING: difference > 1e-6\n")
    else:
        print("  OK\n")

