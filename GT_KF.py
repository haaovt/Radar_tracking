"""
kf_tracker.py  -  Group Tracking với KALMAN FILTER tuyến tính (KF)
------------------------------------------------------------------
Đo trực tiếp trong tọa độ Cartesian: u = [x, y, z]^T (tâm cụm DBSCAN),
nên H là ma trận hằng:  u = H s + v,  H = [I3 | 0].
KF không dùng Doppler trong bước update (Doppler chỉ dùng để khởi tạo vận tốc
và làm đặc trưng cho DBSCAN), vì ṙ là hàm phi tuyến của trạng thái.

Chạy:  python kf_tracker.py --data pointcloud.npz --out ket_qua
"""
import numpy as np
from gtrack_core import BaseFilter, run_cli


class KF(BaseFilter):
    name = "KF"
    meas = "cart"

    def measurement_model(self, x, P):
        # H = [I3 0]  ->  z = x[:3], Pzz = H P H^T, Pxz = P H^T
        return x[:3].copy(), P[:3, :3].copy(), P[:, :3].copy()


if __name__ == "__main__":
    run_cli(KF)