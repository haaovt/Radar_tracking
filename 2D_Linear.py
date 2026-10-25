"""
PHƯƠNG ÁN A — Kalman TUYẾN TÍNH, vector đo u = [x, y]
=====================================================
Centroid của cụm DBSCAN được đo trực tiếp trong toạ độ Descartes.
    H(s) = [x, y]   ->   Jacobian cố định: J = [I2 | 0]
Không dùng Doppler trong bước update (chỉ dùng để khởi tạo vận tốc ban đầu).

Chạy:  python tracker_A_linear.py [đường_dẫn/pointcloud.npz]
"""
import os
import sys
import numpy as np
from radar_common import (BASE_CFG, run_pipeline, save_csv, plot_trajectory, print_summary)

CFG = dict(BASE_CFG)
CFG.update(
    npz_path="pointcloud.npz", out_dir=".",   # có thể truyền đường dẫn npz qua dòng lệnh
    motion_model="CA",     # "CA" (mặc định) hoặc "CV"
    sigma_a=0.05,          # m/s^2: biến thiên gia tốc mỗi frame (đã tinh chỉnh theo NIS)
    sigma_xy=0.18,         # m: độ lệch chuẩn đo của MỘT điểm (tài liệu gợi ý ≈ 1/3.46 = 0.29 m; hạ xuống theo NIS)
    meas_floor=0.05,       # m: sàn độ lệch chuẩn của centroid (tránh lọc quá tự tin)
)


class LinearXY:
    """Mô hình đo tuyến tính [x, y]."""
    name, dim = "A", 2

    def __init__(self, cfg):
        self.R_D = np.diag([cfg["sigma_xy"] ** 2] * 2)
        self.R_floor = np.diag([cfg["meas_floor"] ** 2] * 2)

    def point_vec(self, P, sel):
        return np.c_[P["x"][sel], P["y"][sel]]

    def h(self, s):
        return s[:2].copy()

    def jac(self, s, n):
        J = np.zeros((2, n))
        J[0, 0] = J[1, 1] = 1.0
        return J

    def resid(self, z, zp):
        return z - zp


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CFG["npz_path"] = sys.argv[1]
    out = CFG["out_dir"]
    os.makedirs(out, exist_ok=True)

    res = run_pipeline(LinearXY(CFG), CFG)
    print_summary(res, f"PHƯƠNG ÁN A (tuyến tính [x,y], {CFG['motion_model']})")

    n = save_csv(res, os.path.join(out, "trajectory_A.csv"))
    plot_trajectory(res, os.path.join(out, "trajectory_A.png"),
                    f"Phương án A — Kalman tuyến tính [x, y], mô hình {CFG['motion_model']} + DBSCAN + Hungarian")
    print(f"\nĐã lưu: trajectory_A.png, trajectory_A.csv ({n} dòng) vào '{out}'")