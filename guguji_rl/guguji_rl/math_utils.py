from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def quaternion_xyzw_to_euler(quaternion: Iterable[float]) -> tuple[float, float, float]:
    x, y, z, w = quaternion

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def safe_array(values: Iterable[float], dtype=np.float32) -> np.ndarray:
    return np.asarray(list(values), dtype=dtype)
