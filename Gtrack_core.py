"""
gtrack_core.py
==============
Phần lõi dùng chung cho 3 bộ lọc (KF / EKF / UKF) của Group Tracking
(theo TI "Tracking radar targets with multiple reflection points").

Luồng mỗi frame:
    Predict (tuyến tính, F,Q)  ->  DBSCAN lấy tâm cụm  ->  Association (gating + Hungarian)
    ->  Allocation (cụm chưa ghép)  ->  Update (KF/EKF/UKF)  ->  Maintenance  ->  Report

Mỗi file bộ lọc chỉ cần định nghĩa lớp con của BaseFilter (phần đo/cập nhật).
"""
import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BIG = 1e9


# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------
@dataclass
class Config:
    model: str = "ca"            # cv = {x,y,z,vx,vy,vz}; ca = thêm {ax,ay,az}
    sensor_height: float = 1.2   # m, lấy từ meta.pose.height (z trong file = z_sensor + 1.2)
    # --- DBSCAN (không gian x,y,z + vr*vr_weight) ---
    eps: float = 0.6
    min_samples: int = 3
    vr_weight: float = 0.25      # m trên (m/s): trọng số của Doppler trong khoảng cách DBSCAN
    # --- nhiễu đo từng điểm (R_D) ---
    sig_r: float = 0.05
    sig_az: float = np.radians(2.0)
    sig_el: float = np.radians(2.0)
    sig_vr: float = 0.10
    sig_xyz: float = 0.10        # dùng cho KF (đo Cartesian)
    # --- nhiễu quá trình (Q) ---
    sigma_a: float = 1.5         # m/s^2 cho CV (piecewise white noise)
    sigma_j: float = 16.0        # m/s^3 cho CA (nhiễu jerk)
    acc_tau: float = 0.5         # s, hằng số thời gian suy giảm gia tốc (Singer) cho CA; 0 = không suy giảm
    # --- Group tracking (mục 3.6 tài liệu) ---
    n_hat: float = 20.0          # số điểm kỳ vọng của một mục tiêu
    alpha_cd: float = 0.1        # hệ số làm mượt C_D (3-12)
    gate_prob: float = 0.99      # chi-square gate
    # --- Vòng đời track (bảng 5 tài liệu, đổi cho 25 Hz) ---
    det2active: int = 5
    det2free: int = 2
    active2free: int = 25
    alloc_min_points: int = 5     # số điểm tối thiểu của cụm để cấp track mới (setPointsThre)
    alloc_min_snr: float = 300.0  # tổng SNR tuyến tính tối thiểu của cụm (setSNRThre)
    report_min_hits: int = 40     # chỉ báo cáo track có >= số lần HIT này
    report_min_ratio: float = 0.4 # và tỉ lệ HIT/frame tối thiểu (loại ghost đa đường chập chờn)
    alloc_exclusion: float = 0.8  # m: không cấp track mới nếu cụm nằm gần track cũ
    merge_dist: float = 0.5       # m: gộp hai track trùng nhau


# ----------------------------------------------------------------------------
# Mô hình chuyển động (dùng chung, tuyến tính)
# ----------------------------------------------------------------------------
def build_FQ(dt, cfg):
    """Trạng thái sắp theo [pos(3), vel(3), (acc(3))]."""
    if cfg.model == "cv":
        Fb = np.array([[1, dt], [0, 1.0]])
        G = np.array([[dt ** 2 / 2], [dt]])
        Qb = (G @ G.T) * cfg.sigma_a ** 2          # công thức (2-17)
        order = 2
    else:
        rho = np.exp(-dt / cfg.acc_tau) if cfg.acc_tau > 0 else 1.0
        Fb = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, rho]])
        G = np.array([[dt ** 3 / 6], [dt ** 2 / 2], [dt]])
        Qb = (G @ G.T) * cfg.sigma_j ** 2
        order = 3
    n = 3 * order
    F = np.zeros((n, n))
    Q = np.zeros((n, n))
    for a in range(3):
        idx = [k * 3 + a for k in range(order)]
        F[np.ix_(idx, idx)] = Fb
        Q[np.ix_(idx, idx)] = Qb
    return F, Q


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------
# Lớp cơ sở cho bộ lọc: KF / EKF / UKF kế thừa và cài measurement_model
# ----------------------------------------------------------------------------
class BaseFilter:
    name = "base"
    meas = "polar"        # "cart": u=[x,y,z]; "polar": u=[r,az,el,rdot]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.nx = 6 if cfg.model == "cv" else 9
        self.nz = 3 if self.meas == "cart" else 4

    # -- nhiễu đo từng điểm R_D
    def R_point(self):
        c = self.cfg
        if self.meas == "cart":
            return np.eye(3) * c.sig_xyz ** 2
        return np.diag([c.sig_r ** 2, c.sig_az ** 2, c.sig_el ** 2, c.sig_vr ** 2])

    def residual(self, a, b):
        d = np.asarray(a, float) - np.asarray(b, float)
        if self.meas == "polar":
            d[..., 1] = wrap(d[..., 1])
        return d

    def measurement_model(self, x, P):
        """Trả về (z_pred, Pzz, Pxz) với Pzz = Cov(z) chưa cộng R, Pxz = Cov(x,z)."""
        raise NotImplementedError

    # -- cập nhật chung: S = Pzz + R,  K = Pxz S^-1
    def update(self, x, P, u, R):
        z, Pzz, Pxz = self.measurement_model(x, P)
        S = Pzz + R
        y = self.residual(u, z)
        K = Pxz @ np.linalg.inv(S)
        x_new = x + K @ y
        P_new = P - K @ S @ K.T
        P_new = 0.5 * (P_new + P_new.T)
        d2 = float(y @ np.linalg.solve(S, y))
        return x_new, P_new, d2


# ----------------------------------------------------------------------------
# Track
# ----------------------------------------------------------------------------
class Track:
    _next_id = 1

    def __init__(self, x, P, CD, frame):
        self.id = Track._next_id
        Track._next_id += 1
        self.x, self.P, self.CD = x, P, CD
        self.state = "DETECT"
        self.hits = 1
        self.consec_hit = 1
        self.consec_miss = 0
        self.born = frame
        self.ever_active = False
        self.hist = []      # (frame, t, state_vec, is_hit, n_pts)
        self.nis = []

    @property
    def pos(self):
        return self.x[:3]


# ----------------------------------------------------------------------------
# Nạp dữ liệu + DBSCAN
# ----------------------------------------------------------------------------
def load_points(path):
    d = np.load(path, allow_pickle=True)
    return d["points"]


def frame_arrays(p, cfg):
    az, el = np.radians(p["az"]).astype(float), np.radians(p["el"]).astype(float)
    r = p["range"].astype(float)
    xyz = np.c_[p["x"], p["y"], p["z"] - cfg.sensor_height].astype(float)   # hệ tọa độ cảm biến
    pol = np.c_[r, az, el, p["vr"]].astype(float)
    return xyz, pol, p["vr"].astype(float), p["snr_db"].astype(float)


def cluster_frame(p, filt, cfg):
    """DBSCAN -> danh sách cụm {u, D, N, xyz, vr, snr}. u là tâm cụm trong không gian đo của bộ lọc."""
    if len(p) < cfg.min_samples:
        return []
    xyz, pol, vr, snr = frame_arrays(p, cfg)
    feat = np.c_[xyz, cfg.vr_weight * vr]
    lab = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples).fit_predict(feat)
    U = xyz if filt.meas == "cart" else pol
    out = []
    for k in sorted(set(lab) - {-1}):
        m = lab == k
        Uk = U[m]
        N = int(m.sum())
        dev = Uk - Uk.mean(0)
        if filt.meas == "polar":
            dev[:, 1] = wrap(dev[:, 1])
        D = dev.T @ dev / N                               # (3-11), chuẩn hóa 1/N
        out.append(dict(u=Uk.mean(0), D=D, N=N, xyz=xyz[m].mean(0), vr=vr[m].mean(),
                        snr=float((10 ** (snr[m] / 10)).sum())))
    return out


# ----------------------------------------------------------------------------
# Group tracker
# ----------------------------------------------------------------------------
class GroupTracker:
    def __init__(self, filt: BaseFilter, cfg: Config):
        self.f, self.cfg = filt, cfg
        self.tracks = []
        self.gate = chi2.ppf(cfg.gate_prob, filt.nz)
        Track._next_id = 1

    # R_c = R_D/N_A + f(N_A, N^) C_D     (3-9, 3-10)
    def _Rc(self, CD, N):
        n_hat = self.cfg.n_hat
        f = (n_hat - N) / ((n_hat - 1) * N) if N <= n_hat else 0.0
        return self.f.R_point() / N + f * CD

    def _new_track(self, c, frame):
        cfg = self.cfg
        xyz = c["xyz"]
        rng = np.linalg.norm(xyz)
        v0 = c["vr"] * xyz / max(rng, 1e-6)                # chỉ biết thành phần hướng tâm
        x = np.zeros(self.f.nx)
        x[:3], x[3:6] = xyz, v0
        P = np.eye(self.f.nx)
        P[:3, :3] *= 0.2 ** 2
        P[3:6, 3:6] *= 1.5 ** 2
        if self.f.nx == 9:
            P[6:, 6:] *= 1.0
        return Track(x, P, c["D"].copy() + 1e-6 * np.eye(self.f.nz), frame)

    def step(self, frame, t, pts, dt):
        cfg, f = self.cfg, self.f
        F, Q = build_FQ(dt, cfg)
        # 1) Predict
        for tr in self.tracks:
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + Q
        # 2) Tâm cụm bằng DBSCAN
        clusters = cluster_frame(pts, f, cfg)
        nT, nC = len(self.tracks), len(clusters)
        # 3) Ma trận chi phí cho Hungarian: D^2 = ln|S| + d^2 (3-6), có gate chi-square
        cost = np.full((nT, nC), BIG)
        d2m = np.full((nT, nC), np.inf)
        for i, tr in enumerate(self.tracks):
            z, Pzz, _ = f.measurement_model(tr.x, tr.P)
            for j, c in enumerate(clusters):
                S = Pzz + self._Rc(tr.CD, c["N"])
                y = f.residual(c["u"], z)
                d2 = float(y @ np.linalg.solve(S, y))
                d2m[i, j] = d2
                if d2 < self.gate:
                    cost[i, j] = np.linalg.slogdet(S)[1] + d2
        # 4) Hungarian
        matched_t, matched_c = {}, set()
        if nT and nC:
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < BIG:
                    matched_t[i] = j
                    matched_c.add(j)
        # 5) Update / Miss
        for i, tr in enumerate(self.tracks):
            if i in matched_t:
                c = clusters[matched_t[i]]
                tr.CD = (1 - cfg.alpha_cd) * tr.CD + cfg.alpha_cd * c["D"]       # (3-12)
                R = self._Rc(tr.CD, c["N"])
                tr.x, tr.P, d2 = f.update(tr.x, tr.P, c["u"], R)
                tr.nis.append(d2)
                tr.hits += 1
                tr.consec_hit += 1
                tr.consec_miss = 0
                if tr.state == "DETECT" and tr.consec_hit >= cfg.det2active:
                    tr.state, tr.ever_active = "ACTIVE", True
                tr.hist.append((frame, t, tr.x.copy(), True, c["N"]))
            else:
                tr.consec_miss += 1
                tr.consec_hit = 0
                tr.hist.append((frame, t, tr.x.copy(), False, 0))
        # 6) Allocation: cụm chưa ghép và không nằm gần track hiện có
        for j, c in enumerate(clusters):
            if j in matched_c:
                continue
            if c["N"] < cfg.alloc_min_points or c["snr"] < cfg.alloc_min_snr:
                continue
            if any(np.linalg.norm(c["xyz"] - tr.pos) < cfg.alloc_exclusion for tr in self.tracks):
                continue
            tr = self._new_track(c, frame)
            tr.hist.append((frame, t, tr.x.copy(), True, c["N"]))
            self.tracks.append(tr)
        # 7) Maintenance
        keep = []
        for tr in self.tracks:
            if tr.state == "DETECT" and tr.consec_miss >= cfg.det2free:
                continue
            if tr.state == "ACTIVE" and tr.consec_miss >= cfg.active2free:
                continue
            keep.append(tr)
        self.tracks = self._merge(keep)

    def _merge(self, tracks):
        tracks = sorted(tracks, key=lambda k: -k.hits)
        out = []
        for tr in tracks:
            if any(np.linalg.norm(tr.pos - o.pos) < self.cfg.merge_dist for o in out):
                continue
            out.append(tr)
        return out


# ----------------------------------------------------------------------------
# Chạy toàn bộ + xuất kết quả
# ----------------------------------------------------------------------------
def run_tracking(filt_cls, cfg, points):
    filt = filt_cls(cfg)
    gt = GroupTracker(filt, cfg)
    frames = np.unique(points["frame"])
    t_of = {int(k): float(points["t"][points["frame"] == k][0]) for k in frames}
    all_tracks = {}
    prev_t = None
    for k in range(int(frames.min()), int(frames.max()) + 1):
        pts = points[points["frame"] == k]
        t = t_of.get(k, (prev_t or 0) + 0.04)
        dt = 0.04 if prev_t is None else max(t - prev_t, 1e-3)
        gt.step(k, t, pts, dt)
        for tr in gt.tracks:
            all_tracks[tr.id] = tr
        prev_t = t
    final = [tr for tr in all_tracks.values() if tr.ever_active]
    # cắt phần đuôi chỉ-dự-đoán (trước khi track bị xóa) vì nó trôi theo quán tính, không có đo đạc
    for tr in final:
        while tr.hist and not tr.hist[-1][3]:
            tr.hist.pop()
    # lọc theo chất lượng: đủ số HIT và tỉ lệ HIT/frame
    final = [tr for tr in final if sum(h[3] for h in tr.hist) >= cfg.report_min_hits
             and sum(h[3] for h in tr.hist) / len(tr.hist) >= cfg.report_min_ratio]
    return final, filt


def save_csv(tracks, cfg, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "frame", "t", "x", "y", "z_world", "vx", "vy", "vz", "hit", "n_points"])
        for tr in tracks:
            for fr, t, x, hit, n in tr.hist:
                w.writerow([tr.id, fr, f"{t:.3f}", *(f"{v:.4f}" for v in (x[0], x[1], x[2] + cfg.sensor_height,
                                                                         x[3], x[4], x[5])), int(hit), n])


def plot_2d(tracks, points, cfg, title, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points["x"], points["y"], s=2, c="0.8", label="point cloud (raw)", zorder=1)
    cm = plt.get_cmap("tab10")
    for i, tr in enumerate(tracks):
        X = np.array([h[2][:3] for h in tr.hist])
        hit = np.array([h[3] for h in tr.hist])
        col = cm(i % 10)
        ax.plot(X[:, 0], X[:, 1], "-", c=col, lw=1.8, zorder=3, label=f"Track {tr.id} ({len(X)} frames)")
        ax.scatter(X[~hit, 0], X[~hit, 1], s=8, c=[col], marker="x", zorder=3)
        ax.scatter(*X[0, :2], s=70, c=[col], marker="o", edgecolors="k", zorder=4)
        ax.scatter(*X[-1, :2], s=70, c=[col], marker="s", edgecolors="k", zorder=4)
    ax.scatter(0, 0, marker="*", s=200, c="red", zorder=5, label="Radar")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(title + "\n(o = bắt đầu, ■ = kết thúc, x = frame chỉ dự đoán)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_3d(tracks, points, cfg, title, path):
    """Trái: quỹ đạo 3D (màu = thời gian). Phải: độ cao Z theo thời gian."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    allX = np.vstack([np.array([h[2][:3] for h in tr.hist]) for tr in tracks])
    allX[:, 2] += cfg.sensor_height
    lo, hi = allX.min(0) - 0.7, allX.max(0) + 0.7
    m = ((points["x"] > lo[0]) & (points["x"] < hi[0]) & (points["y"] > lo[1]) & (points["y"] < hi[1])
         & (points["z"] > lo[2]) & (points["z"] < hi[2]))
    tmax = float(points["t"].max())
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    ax.scatter(points["x"][m], points["y"][m], points["z"][m], s=1.5, c="0.85", alpha=0.4)
    cmap = plt.get_cmap("viridis")
    for tr in tracks:
        X = np.array([h[2][:3] for h in tr.hist])
        t = np.array([h[1] for h in tr.hist])
        X[:, 2] += cfg.sensor_height
        seg = np.stack([X[:-1], X[1:]], axis=1)
        lc = Line3DCollection(seg, cmap=cmap, norm=plt.Normalize(0, tmax), linewidths=2.2)
        lc.set_array(t[:-1])
        ax.add_collection3d(lc)
        ax.scatter(*X[0], s=70, c="lime", marker="o", edgecolors="k", zorder=5)
        ax.scatter(*X[-1], s=70, c="red", marker="s", edgecolors="k", zorder=5)
        ax2.plot(t, X[:, 2], lw=1.2, label=f"Track {tr.id}")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], 0.8 * (hi[1] - lo[1])))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m) – độ cao so với sàn")
    ax.set_title(title + "\nMàu đường = thời gian; xanh lá = bắt đầu, đỏ = kết thúc")
    ax.view_init(elev=28, azim=-65)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, tmax))
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label="t (s)")
    ax2.axhline(cfg.sensor_height, c="red", ls="--", lw=1, label="độ cao radar")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("Z ước lượng (m)"); ax2.grid(alpha=0.3)
    ax2.set_title("Độ cao tâm cụm theo thời gian"); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def summarize(tracks, filt):
    print(f"[{filt.name}] số track ACTIVE cuối cùng: {len(tracks)}")
    for tr in tracks:
        n = len(tr.hist)
        hits = sum(h[3] for h in tr.hist)
        nis = np.mean(tr.nis) if tr.nis else float("nan")
        print(f"   Track {tr.id:3d}: frame {tr.hist[0][0]:4d}-{tr.hist[-1][0]:4d} | {n:4d} frame, "
              f"{hits:4d} hit | NIS trung bình = {nis:5.2f} (kỳ vọng ≈ {filt.nz})")


def run_cli(filt_cls):
    ap = argparse.ArgumentParser(description=f"Group Tracking với {filt_cls.name}")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "pointcloud.npz"))
    ap.add_argument("--out", default=here)
    ap.add_argument("--model", choices=["cv", "ca"], default="ca")
    ap.add_argument("--eps", type=float, default=Config.eps)
    ap.add_argument("--min-samples", type=int, default=Config.min_samples)
    ap.add_argument("--sigma-a", type=float, default=Config.sigma_a, help="nhiễu gia tốc (CV)")
    ap.add_argument("--sigma-j", type=float, default=Config.sigma_j, help="nhiễu jerk (CA)")
    ap.add_argument("--acc-tau", type=float, default=Config.acc_tau, help="hằng số suy giảm gia tốc (CA), s")
    a = ap.parse_args()
    cfg = Config(model=a.model, eps=a.eps, min_samples=a.min_samples, sigma_a=a.sigma_a,
                 sigma_j=a.sigma_j, acc_tau=a.acc_tau)
    pts = load_points(a.data)
    tracks, filt = run_tracking(filt_cls, cfg, pts)
    summarize(tracks, filt)
    os.makedirs(a.out, exist_ok=True)
    tag = filt.name.lower()
    title = f"Group Tracking – DBSCAN + {filt.name} + Hungarian ({cfg.model.upper()})"
    plot_2d(tracks, pts, cfg, title, os.path.join(a.out, f"{tag}_tracking_2d.png"))
    plot_3d(tracks, pts, cfg, title, os.path.join(a.out, f"{tag}_tracking_3d.png"))
    save_csv(tracks, cfg, os.path.join(a.out, f"{tag}_tracks.csv"))
    print(f"Đã lưu: {tag}_tracking_2d.png, {tag}_tracking_3d.png, {tag}_tracks.csv trong {a.out}")