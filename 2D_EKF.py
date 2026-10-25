"""
PHƯƠNG ÁN B — EKF theo tài liệu TI, vector đo u = [r, phi, r_dot]
==================================================================
    H(s) = [ sqrt(x^2+y^2),  atan2(x, y),  (x*vx + y*vy)/sqrt(x^2+y^2) ]   (công thức 1-6 / 1-16)
    J_H  = đạo hàm riêng theo công thức (1-18) cho mô hình CA (6 trạng thái);
           với CV chỉ lấy 4 cột đầu (công thức 1-10).

Đo lường (trên mặt phẳng ngang):
    r      = sqrt(x^2 + y^2)               (khoảng cách ngang, không dùng cột 'range' 3D đã lượng tử hoá)
    phi    = atan2(x, y)                   (= cột 'az')
    r_dot  = vr / cos(el)                  (hiệu chỉnh Doppler 3D -> mặt phẳng ngang)

Chạy:  python tracker_B_ekf.py [đường_dẫn/pointcloud.npz]
"""
import os
import sys
import numpy as np
from Radar_base import (BASE_CFG, run_pipeline, save_csv, plot_trajectory, print_summary)

CFG = dict(BASE_CFG)
CFG.update(
    npz_path="pointcloud.npz", out_dir=".",   # có thể truyền đường dẫn npz qua dòng lệnh
    motion_model="CA",         # "CA" (mặc định) hoặc "CV"
    sigma_a=0.05,              # m/s^2 (đã tinh chỉnh theo NIS)
    sigma_r=0.08,              # m      : sai số đo r của một điểm
    sigma_phi_deg=2.0,         # độ     : sai số đo góc của một điểm
    sigma_rdot=0.40,           # m/s    : Doppler của centroid dao động mạnh do tay chân đung đưa (micro-Doppler)
    floor_r=0.05, floor_phi_deg=2.0, floor_rdot=0.40,   # sàn độ lệch chuẩn của centroid
)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PolarEKF:
    """Mô hình đo phi tuyến [r, phi, r_dot] + Jacobian theo tài liệu."""
    name, dim = "B", 3

    def __init__(self, cfg):
        self.R_D = np.diag([cfg["sigma_r"] ** 2,
                            np.deg2rad(cfg["sigma_phi_deg"]) ** 2,
                            cfg["sigma_rdot"] ** 2])
        self.R_floor = np.diag([cfg["floor_r"] ** 2,
                                np.deg2rad(cfg["floor_phi_deg"]) ** 2,
                                cfg["floor_rdot"] ** 2])

    def point_vec(self, P, sel):
        return np.c_[P["r2d"][sel], P["phi"][sel], P["rdot2d"][sel]]

    def h(self, s):
        x, y, vx, vy = s[:4]
        r = max(np.hypot(x, y), 1e-3)
        return np.array([r, np.arctan2(x, y), (x * vx + y * vy) / r])

    def jac(self, s, n):
        x, y, vx, vy = s[:4]
        r2 = max(x * x + y * y, 1e-6)
        r = np.sqrt(r2)
        r3 = r2 * r
        J = np.zeros((3, n))
        J[0, 0], J[0, 1] = x / r, y / r
        J[1, 0], J[1, 1] = y / r2, -x / r2
        J[2, 0] = y * (vx * y - vy * x) / r3
        J[2, 1] = x * (vy * x - vx * y) / r3
        J[2, 2], J[2, 3] = x / r, y / r
        return J

    def resid(self, z, zp):
        d = z - zp
        d[1] = wrap(d[1])
        return d


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CFG["npz_path"] = sys.argv[1]
    out = CFG["out_dir"]
    os.makedirs(out, exist_ok=True)

    res = run_pipeline(PolarEKF(CFG), CFG)
    print_summary(res, f"PHƯƠNG ÁN B (EKF [r,phi,r_dot], {CFG['motion_model']})")

    n = save_csv(res, os.path.join(out, "trajectory_B.csv"))
    plot_trajectory(res, os.path.join(out, "trajectory_B.png"),
                    f"Phương án B — EKF [r, φ, ṙ], mô hình {CFG['motion_model']} + DBSCAN + Hungarian")
    print(f"\nĐã lưu: trajectory_B.png, trajectory_B.csv ({n} dòng) vào '{out}'")