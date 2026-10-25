"""
ekf_tracker.py  -  Group Tracking với EXTENDED KALMAN FILTER (EKF)
-------------------------------------------------------------------
Đo trong tọa độ cực: u = [r, φ, θ, ṙ]^T (tài liệu TI mục 1.7-1.8, công thức 1-27):
    r  = sqrt(x²+y²+z²)
    φ  = atan2(x, y)                 (azimuth, tính từ trục Y)
    θ  = atan2(z, sqrt(x²+y²))       (elevation)
    ṙ  = (x·ẋ + y·ẏ + z·ż) / r
Tuyến tính hóa bằng Jacobian J_H (1-31/1-32, Phụ lục 7.2); các cột gia tốc bằng 0.

Chạy:  python ekf_tracker.py --data pointcloud.npz --out ket_qua
"""
import numpy as np
from Gtrack_core import BaseFilter, run_cli


def h_polar(x):
    p, v = x[:3], x[3:6]
    r = np.linalg.norm(p)
    rho = np.hypot(p[0], p[1])
    return np.array([r, np.arctan2(p[0], p[1]), np.arctan2(p[2], rho), p @ v / r])


def jacobian_polar(x):
    px, py, pz = x[:3]
    v = x[3:6]
    p = x[:3]
    r2 = px * px + py * py + pz * pz
    r = np.sqrt(r2)
    rho2 = px * px + py * py
    rho = np.sqrt(rho2)
    rdot = p @ v / r
    J = np.zeros((4, len(x)))
    J[0, :3] = p / r                                          # ∂r/∂pos
    J[1, 0], J[1, 1] = py / rho2, -px / rho2                  # ∂φ/∂x, ∂φ/∂y
    J[2, 0] = -px * pz / (r2 * rho)                           # ∂θ/∂x
    J[2, 1] = -py * pz / (r2 * rho)                           # ∂θ/∂y
    J[2, 2] = rho / r2                                        # ∂θ/∂z
    J[3, :3] = (v - rdot * p / r) / r                         # ∂ṙ/∂pos  (== (1-31) hàng 4)
    J[3, 3:6] = p / r                                         # ∂ṙ/∂vel
    return J


class EKF(BaseFilter):
    name = "EKF"
    meas = "polar"

    def measurement_model(self, x, P):
        J = jacobian_polar(x)
        return h_polar(x), J @ P @ J.T, P @ J.T


if __name__ == "__main__":
    run_cli(EKF)