import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def quat_conjugate(q):
    """
    Conjugate of quaternion(s) in (x, y, z, w) format.
    Supports NumPy arrays or PyTorch tensors with shape (..., 4).
    """
    if isinstance(q, torch.Tensor):
        result = q.clone()
        result[..., :3] = -result[..., :3]
        return result
    else:
        q_np = np.asarray(q)
        result = q_np.copy()
        result[..., :3] = -result[..., :3]
        return result


def quat_mul(q, r):
    """
    Hamilton product of two quaternion(s) in (x, y, z, w) format.
    Supports broadcasting and shape (..., 4).
    Returns same type (NumPy or Torch) as inputs.
    """
    if isinstance(q, torch.Tensor) and isinstance(r, torch.Tensor):
        x1, y1, z1, w1 = q.unbind(-1)
        x2, y2, z2, w2 = r.unbind(-1)

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2

        return torch.stack([x, y, z, w], dim=-1)
    else:
        q_np = np.asarray(q)
        r_np = np.asarray(r)

        x1, y1, z1, w1 = q_np[..., 0], q_np[..., 1], q_np[..., 2], q_np[..., 3]
        x2, y2, z2, w2 = r_np[..., 0], r_np[..., 1], r_np[..., 2], r_np[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2

        return np.stack([x, y, z, w], axis=-1)


def quat_invmul(quat_a, quat_b):
    """
    Relative rotation: quat_B_in_A = conjugate(quat_a) ⊗ quat_b
    All quaternions are (x, y, z, w).
    """
    return quat_mul(quat_conjugate(quat_a), quat_b)


def get_orientation_2d_from_quat(quat):
    """
    From quaternion(s) in (x, y, z, w), get the first two columns of the
    rotation matrix. Returns NumPy array with shape (..., 6) where the last
    dimension is column-0 followed by column-1 flattened.
    """
    quat_np = quat.detach().cpu().numpy() if isinstance(quat, torch.Tensor) else np.asarray(quat)

    # Ensure shape (..., 4)
    if quat_np.shape[-1] != 4:
        raise ValueError("Quaternion must have last dimension 4: (x, y, z, w)")

    rot = R.from_quat(quat_np)
    mat = rot.as_matrix()               # (..., 3, 3)
    cols2 = mat[..., :, :2]             # (..., 3, 2)
    out = cols2.reshape(cols2.shape[:-2] + (6,))  # (..., 6)
    return out