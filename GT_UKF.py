"""
ukf_tracker.py  -  Group Tracking với UNSCENTED KALMAN FILTER (UKF)
-------------------------------------------------------------------
Cùng mô hình đo cực u = [r, φ, θ, ṙ]^T như EKF nhưng KHÔNG dùng Jacobian:
phân bố trạng thái được đại diện bằng 2n+1 sigma points (scaled unscented transform),
đẩy qua h(·) rồi lấy trung bình/hiệp phương sai có trọng số.
Bước dự đoán là tuyến tính nên dùng F trực tiếp (kết quả tương đương UT).
Phần dư của góc azimuth được wrap về [-π, π].

Chạy:  python ukf_tracker.py --data pointcloud.npz --out ket_qua
"""
import numpy as np
from gtrack_core import BaseFilter, run_cli, wrap
from ekf_tracker import h_polar


class UKF(BaseFilter):
    name = "UKF"
    meas = "polar"
    alpha, beta, kappa = 0.3, 2.0, 0.0

    def _weights(self, n):
        lam = self.alpha ** 2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 0.5 / (n + lam))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1 - self.alpha ** 2 + self.beta)
        return lam, wm, wc

    def measurement_model(self, x, P):
        n = len(x)
        lam, wm, wc = self._weights(n)
        P = 0.5 * (P + P.T) + 1e-9 * np.eye(n)
        L = np.linalg.cholesky((n + lam) * P)
        X = np.vstack([x, x + L.T, x - L.T])                  # (2n+1, n)
        Z = np.array([h_polar(s) for s in X])                 # (2n+1, 4)
        # trung bình có xử lý góc: tính quanh sigma point trung tâm
        d0 = Z - Z[0]
        d0[:, 1] = wrap(d0[:, 1])
        zbar = Z[0] + wm @ d0
        dz = Z - zbar
        dz[:, 1] = wrap(dz[:, 1])
        dx = X - x
        Pzz = (wc[:, None] * dz).T @ dz
        Pxz = (wc[:, None] * dx).T @ dz
        return zbar, Pzz, Pxz


if __name__ == "__main__":
    run_cli(UKF)