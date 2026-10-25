"""
radar_common.py
===============
Các khối dùng chung cho hai phương án:
    - tracker_A_linear.py : Kalman tuyến tính, đo [x, y]
    - tracker_B_ekf.py    : EKF theo tài liệu TI, đo [r, phi, r_dot]

Pipeline mỗi frame:
    lọc điểm -> DBSCAN -> centroid -> Predict -> Hungarian (gating + scoring)
    -> Update -> cấp phát track mới -> quản lý vòng đời (DETECT/ACTIVE/FREE)

Quy ước toạ độ (mục 1.4 tài liệu, alpha = 0):
    x = r*sin(phi) (ngang), y = r*cos(phi) (hướng ra phía trước radar),
    phi = atan2(x, y).
"""
import csv
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------------------------
# CẤU HÌNH CHUNG (mọi tham số chỉnh ở đây)
# ----------------------------------------------------------------------------
BASE_CFG = dict(
    npz_path="pointcloud.npz",
    out_dir=".",

    # --- Tiền xử lý ---
    z_min=0.0, z_max=2.2,          # giữ điểm có 0 < z < 2.2 m (z là độ cao so với sàn)
    snr_min_db=15.0,               # bỏ điểm SNR thấp
    el_cos_min=0.5,                # chặn cos(el) khi đổi vr -> r_dot trên mặt phẳng ngang

    # --- DBSCAN ---
    eps=0.6,                       # m
    min_samples=3,
    vel_weight=0.0,                # >0: đưa vr vào không gian DBSCAN (m <-> m/s)

    # --- Mô hình chuyển động ---
    motion_model="CA",             # "CA" (mặc định, theo tài liệu) hoặc "CV"
    sigma_a=0.25,                  # m/s^2: độ lệch chuẩn của biến thiên gia tốc mỗi frame (piecewise white noise)
    p0_pos=0.4, p0_vel=1.0, p0_acc=1.0,   # độ lệch chuẩn khởi tạo P0

    # --- Nhiễu đo (mục 3.6, công thức 3-9, 3-10) ---
    n_hat=20,                      # N^: số phần tử ước tính của mục tiêu
    cd_alpha=0.1,                  # hệ số làm mượt C_D (công thức 3-12)
    meas_floor=None,               # sàn độ lệch chuẩn của R (set trong từng phương án)

    # --- Association ---
    gate_prob=0.99,                # chi-bình-phương gating

    # --- Vòng đời track (Bảng 5) ---
    alloc_min_points=3,
    det2active=5,                  # số HIT liên tiếp để DETECT -> ACTIVE
    det2free=2,                    # số MISS liên tiếp để DETECT -> FREE
    active2free=25,                # số MISS liên tiếp để ACTIVE -> FREE (25 frame = 1 s)

    frame_dt=None,                 # None: lấy từ meta (0.04 s)
)


# ----------------------------------------------------------------------------
# ĐỌC DỮ LIỆU + TIỀN XỬ LÝ
# ----------------------------------------------------------------------------
def load_points(cfg):
    d = np.load(cfg["npz_path"], allow_pickle=True)
    raw = d["points"]
    meta = json.loads(str(d["meta"]))
    order = np.argsort(raw["frame"], kind="stable")
    raw = raw[order]

    P = {k: raw[k].astype(np.float64) for k in
         ["t", "x", "y", "z", "vr", "snr_db", "range", "az", "el"]}
    P["frame"] = raw["frame"].astype(np.int64)

    # Đại lượng trên mặt phẳng ngang (2D) cho EKF
    P["r2d"] = np.hypot(P["x"], P["y"])
    P["phi"] = np.arctan2(P["x"], P["y"])                       # = az (rad)
    cos_el = np.clip(np.cos(np.deg2rad(P["el"])), cfg["el_cos_min"], 1.0)
    P["rdot2d"] = P["vr"] / cos_el                              # hiệu chỉnh vr 3D -> 2D
    P["w"] = 10.0 ** (P["snr_db"] / 20.0)                       # trọng số theo biên độ SNR

    P["keep"] = ((P["z"] > cfg["z_min"]) & (P["z"] < cfg["z_max"]) &
                 (P["snr_db"] >= cfg["snr_min_db"]))
    return P, meta


# ----------------------------------------------------------------------------
# DBSCAN + CENTROID
# ----------------------------------------------------------------------------
def cluster_frame(P, idx, cfg):
    """DBSCAN trên các điểm đã lọc của một frame. Trả về list các mảng chỉ số."""
    if len(idx) < cfg["min_samples"]:
        return []
    X = np.c_[P["x"][idx], P["y"][idx], cfg["vel_weight"] * P["vr"][idx]]
    lab = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"]).fit_predict(X)
    return [idx[lab == l] for l in sorted(set(lab) - {-1})]


def make_measurement(mm, P, sel, cfg):
    """Tính centroid (trung bình có trọng số SNR) và ma trận phân tán C_D của một cụm."""
    w = P["w"][sel]
    w = w / w.sum()
    V = mm.point_vec(P, sel)                       # (N, dim)
    z = (w[:, None] * V).sum(0)
    dV = V - z
    CD = (dV * w[:, None]).T @ dV                  # hiệp phương sai có trọng số
    cx = float((w * P["x"][sel]).sum())
    cy = float((w * P["y"][sel]).sum())
    return dict(z=z, CD=CD, n=len(sel), cx=cx, cy=cy,
                rdot=float((w * P["rdot2d"][sel]).sum()), sel=sel)


# ----------------------------------------------------------------------------
# MA TRẬN F, Q (mục 1.5, 1.6, 2.3.2)
# ----------------------------------------------------------------------------
def build_F(T, model):
    if model == "CA":
        F = np.eye(6)
        F[0, 2] = F[1, 3] = T
        F[0, 4] = F[1, 5] = 0.5 * T * T
        F[2, 4] = F[3, 5] = T
        return F
    F = np.eye(4)
    F[0, 2] = F[1, 3] = T
    return F


def build_Q(T, sigma, model):
    """Piecewise white noise: Q = G G^T sigma^2, G = [T^2/2, T, 1] cho mỗi trục."""
    if model == "CA":
        g = np.array([0.5 * T * T, T, 1.0])
        ax_idx = [(0, 2, 4), (1, 3, 5)]
        n = 6
    else:
        g = np.array([0.5 * T * T, T])
        ax_idx = [(0, 2), (1, 3)]
        n = 4
    Q = np.zeros((n, n))
    for ids in ax_idx:
        Q[np.ix_(ids, ids)] = np.outer(g, g) * sigma ** 2
    return Q


# ----------------------------------------------------------------------------
# TRACK (Kalman / EKF)
# ----------------------------------------------------------------------------
class Track:
    def __init__(self, tid, mm, cfg, meas, frame, F, Q):
        self.id, self.mm, self.cfg, self.F, self.Q = tid, mm, cfg, F, Q
        self.n = F.shape[0]
        x, y = meas["cx"], meas["cy"]
        phi = np.arctan2(x, y)
        s = np.zeros(self.n)
        s[0], s[1] = x, y
        s[2], s[3] = meas["rdot"] * np.sin(phi), meas["rdot"] * np.cos(phi)  # từ Doppler
        self.s = s
        p0 = [cfg["p0_pos"]] * 2 + [cfg["p0_vel"]] * 2 + [cfg["p0_acc"]] * 2
        self.P = np.diag(np.array(p0[:self.n]) ** 2)
        self.CD = meas["CD"] + 1e-6 * np.eye(mm.dim)
        self.state = "DETECT"
        self.hits, self.misses, self.consec_hits = 1, 0, 1
        self.ever_active = False
        self.records, self.nis = [], []
        self.start_frame = frame
        self._record(frame, "hit", meas)

    # ---- Predict (2-1, 2-2) ----
    def predict(self):
        self.s = self.F @ self.s
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.zp = self.mm.h(self.s)
        self.J = self.mm.jac(self.s, self.n)

    # ---- R của centroid (3-9, 3-10) ----
    def meas_R(self, meas):
        N, Nh = meas["n"], self.cfg["n_hat"]
        f = 0.0 if N >= Nh else (Nh - N) / ((Nh - 1) * N)
        return self.mm.R_D / N + f * self.CD + self.mm.R_floor

    # ---- Innovation + gating/scoring (3-2, 3-6) ----
    def innovation(self, meas):
        y = self.mm.resid(meas["z"], self.zp)
        S = self.J @ self.P @ self.J.T + self.meas_R(meas)
        d2 = float(y @ np.linalg.solve(S, y))
        logdet = np.linalg.slogdet(S)[1]
        return y, S, d2, logdet

    # ---- Update (2-3 ... 2-7), dạng Joseph cho ổn định số ----
    def update(self, meas):
        y, S, d2, _ = self.innovation(meas)
        K = self.P @ self.J.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        I_KJ = np.eye(self.n) - K @ self.J
        R = self.meas_R(meas)
        self.P = I_KJ @ self.P @ I_KJ.T + K @ R @ K.T
        a = self.cfg["cd_alpha"]
        self.CD = (1 - a) * self.CD + a * (meas["CD"] + 1e-6 * np.eye(self.mm.dim))
        self.nis.append(d2)

    def _record(self, frame, status, meas=None):
        self.records.append(dict(
            frame=frame, status=status, state=self.state,
            x=self.s[0], y=self.s[1], vx=self.s[2], vy=self.s[3],
            ax=self.s[4] if self.n == 6 else np.nan,
            ay=self.s[5] if self.n == 6 else np.nan,
            n_points=meas["n"] if meas else 0,
            meas_x=meas["cx"] if meas else np.nan,
            meas_y=meas["cy"] if meas else np.nan))


# ----------------------------------------------------------------------------
# TRACK MANAGER: Hungarian + vòng đời
# ----------------------------------------------------------------------------
class TrackManager:
    BIG = 1e5

    def __init__(self, mm, cfg, T):
        self.mm, self.cfg, self.T = mm, cfg, T
        self.F = build_F(T, cfg["motion_model"])
        self.Q = build_Q(T, cfg["sigma_a"], cfg["motion_model"])
        self.gate = chi2.ppf(cfg["gate_prob"], mm.dim)
        self.tracks, self.finished, self.next_id = [], [], 1

    def step(self, frame, meas_list):
        cfg = self.cfg
        for tr in self.tracks:
            tr.predict()

        # --- Association: Hungarian trên cost = ln|S| + d^2, gating d^2 < G ---
        matched = {}
        nT, nM = len(self.tracks), len(meas_list)
        if nT and nM:
            cost = np.full((nT, nM), self.BIG)
            for i, tr in enumerate(self.tracks):
                for j, m in enumerate(meas_list):
                    _, _, d2, ld = tr.innovation(m)
                    if d2 < self.gate:
                        cost[i, j] = ld + d2
            ri, ci = linear_sum_assignment(cost)
            matched = {i: j for i, j in zip(ri, ci) if cost[i, j] < self.BIG}

        # --- Update / Miss ---
        used = set()
        survivors = []
        for i, tr in enumerate(self.tracks):
            if i in matched:
                m = meas_list[matched[i]]
                used.add(matched[i])
                tr.update(m)
                tr.hits += 1
                tr.consec_hits += 1
                tr.misses = 0
                if tr.state == "DETECT" and tr.consec_hits >= cfg["det2active"]:
                    tr.state = "ACTIVE"
                    tr.ever_active = True
                tr._record(frame, "hit", m)
                survivors.append(tr)
            else:
                tr.misses += 1
                tr.consec_hits = 0
                dead = ((tr.state == "DETECT" and tr.misses >= cfg["det2free"]) or
                        (tr.state == "ACTIVE" and tr.misses >= cfg["active2free"]))
                if dead:
                    # bỏ các bản ghi "coast" ở đuôi (chỉ là dự đoán, không có dữ liệu thật)
                    if tr.misses > 0:
                        tr.records = tr.records[:-tr.misses] if len(tr.records) > tr.misses else []
                    self.finished.append(tr)
                else:
                    tr._record(frame, "coast")
                    survivors.append(tr)
        self.tracks = survivors

        # --- Allocation: cụm chưa được gán -> track mới ---
        for j, m in enumerate(meas_list):
            if j in used or m["n"] < cfg["alloc_min_points"]:
                continue
            self.tracks.append(Track(self.next_id, self.mm, cfg, m, frame, self.F, self.Q))
            self.next_id += 1

    def close(self):
        self.finished.extend(self.tracks)
        self.tracks = []
        return [t for t in self.finished if len(t.records) > 0]


# ----------------------------------------------------------------------------
# PIPELINE CHÍNH
# ----------------------------------------------------------------------------
def run_pipeline(mm, cfg):
    P, meta = load_points(cfg)
    T = cfg["frame_dt"] or float(meta.get("frame_dt", 0.04))
    n_frames = int(P["frame"].max()) + 1
    starts = np.searchsorted(P["frame"], np.arange(n_frames + 1))

    mgr = TrackManager(mm, cfg, T)
    all_centroids = []
    for f in range(n_frames):
        idx = np.arange(starts[f], starts[f + 1])
        idx = idx[P["keep"][idx]]
        clusters = cluster_frame(P, idx, cfg)
        meas_list = [make_measurement(mm, P, s, cfg) for s in clusters]
        for m in meas_list:
            all_centroids.append((f, m["cx"], m["cy"]))
        mgr.step(f, meas_list)
    tracks = mgr.close()

    # --- Phân loại track ---
    confirmed = [t for t in tracks if t.ever_active]
    main = max(confirmed, key=lambda t: t.hits) if confirmed else None
    for t in tracks:
        t.cls = ("main" if t is main else "confirmed" if t.ever_active else "noise")

    return dict(P=P, meta=meta, T=T, tracks=tracks, main=main,
                centroids=np.array(all_centroids), cfg=cfg, mm=mm)


# ----------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------
def save_csv(res, path):
    T = res["T"]
    cols = ["track_id", "track_class", "is_main_track", "frame", "t", "x", "y",
            "vx", "vy", "ax", "ay", "state", "status", "n_points", "meas_x", "meas_y"]
    rows = []
    for t in res["tracks"]:
        for r in t.records:
            rows.append([t.id, t.cls, int(t.cls == "main"), r["frame"],
                         round(r["frame"] * T, 3),
                         *[None if np.isnan(r[k]) else round(float(r[k]), 4)
                           for k in ["x", "y", "vx", "vy", "ax", "ay"]],
                         r["state"], r["status"], r["n_points"],
                         None if np.isnan(r["meas_x"]) else round(float(r["meas_x"]), 4),
                         None if np.isnan(r["meas_y"]) else round(float(r["meas_y"]), 4)])
    rows.sort(key=lambda r: (r[0], r[3]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    return len(rows)


# ----------------------------------------------------------------------------
# ẢNH QUỸ ĐẠO
# ----------------------------------------------------------------------------
def plot_trajectory(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    P, T, cfg = res["P"], res["T"], res["cfg"]
    fig, ax = plt.subplots(figsize=(9, 9))

    # point cloud nền (chỉ điểm đã qua lọc) + centroid đo thô
    k = P["keep"]
    ax.scatter(P["x"][k], P["y"][k], s=2, c="#c8c8c8", alpha=0.35, linewidths=0,
               label="Point cloud (đã lọc)", zorder=1)
    c = res["centroids"]
    if len(c):
        ax.scatter(c[:, 1], c[:, 2], s=7, c="#7fb2e5", alpha=0.6, linewidths=0,
                   label="Centroid DBSCAN (đo thô)", zorder=2)

    # radar + FOV
    ax.plot(0, 0, marker="^", ms=12, c="k", zorder=6, label="Radar")
    for sgn in (-1, 1):
        a = np.deg2rad(60) * sgn
        ax.plot([0, 12 * np.sin(a)], [0, 12 * np.cos(a)], c="k", lw=0.6, ls=":", alpha=0.6)

    # track nhiễu / phụ
    first_noise = first_conf = True
    n_noise = sum(t.cls == "noise" for t in res["tracks"])
    n_conf = sum(t.cls == "confirmed" for t in res["tracks"])
    for t in res["tracks"]:
        if t.cls == "main" or len(t.records) < 1:
            continue
        xs = [r["x"] for r in t.records]
        ys = [r["y"] for r in t.records]
        if t.cls == "noise":
            ax.plot(xs, ys, c="#8a8a8a", lw=1.0, ls="--", zorder=3,
                    label=f"Track nhiễu / chưa xác nhận ({n_noise})" if first_noise else None)
            first_noise = False
        else:
            ax.plot(xs, ys, c="#2ca02c", lw=1.6, zorder=4,
                    label=f"Track phụ đã xác nhận ({n_conf})" if first_conf else None)
            first_conf = False
        if t.cls == "confirmed":
            ax.annotate(f"#{t.id}", (xs[0], ys[0]), fontsize=7, color="#1b6e1b", zorder=5,
                        xytext=(3, 3), textcoords="offset points")

    # track chính, tô màu theo thời gian
    main = res["main"]
    if main is not None:
        rec = main.records
        xs = np.array([r["x"] for r in rec])
        ys = np.array([r["y"] for r in rec])
        ts = np.array([r["frame"] for r in rec]) * T
        pts = np.c_[xs, ys].reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap="plasma", linewidths=3.2, zorder=7)
        lc.set_array(ts[:-1])
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("Thời gian (s)")
        ax.scatter(xs[0], ys[0], s=110, c="#00c853", edgecolors="k", zorder=8, label="Bắt đầu (track chính)")
        ax.scatter(xs[-1], ys[-1], s=110, c="#d50000", marker="s", edgecolors="k", zorder=8, label="Kết thúc (track chính)")
        ax.annotate(f"#{main.id} (người)", (xs[0], ys[0]), fontsize=9, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points", zorder=9)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)  — ngang")
    ax.set_ylabel("Y (m)  — hướng ra phía trước radar")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# THỐNG KÊ NHANH
# ----------------------------------------------------------------------------
def print_summary(res, tag):
    T = res["T"]
    tracks = res["tracks"]
    print(f"\n===== {tag} =====")
    print(f"Số track: {len(tracks)}  | main: {sum(t.cls=='main' for t in tracks)}"
          f"  confirmed: {sum(t.cls=='confirmed' for t in tracks)}"
          f"  noise: {sum(t.cls=='noise' for t in tracks)}")
    for t in sorted(tracks, key=lambda t: -t.hits)[:6]:
        nis = np.mean(t.nis) if t.nis else np.nan
        print(f"  #{t.id:<3} {t.cls:<9} frames {t.records[0]['frame']:>4}-{t.records[-1]['frame']:<4} "
              f"hits={t.hits:<4} mean NIS={nis:.2f}")
    m = res["main"]
    if m is not None:
        rec = [r for r in m.records]
        xy = np.array([[r["x"], r["y"]] for r in rec])
        hit = np.array([r["status"] == "hit" for r in rec])
        mx = np.array([[r["meas_x"], r["meas_y"]] for r in rec])
        d = np.linalg.norm(xy[hit] - mx[hit], axis=1)
        path_len = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        acc = np.linalg.norm(np.diff(xy, 2, axis=0), axis=1) / T ** 2
        v = np.array([np.hypot(r["vx"], r["vy"]) for r in rec])
        print(f"  Track chính #{m.id}: {len(rec)} frame ({len(rec)*T:.1f}s), hits={m.hits}, "
              f"coast={int((~hit).sum())}, độ dài đường đi={path_len:.1f} m")
        print(f"  RMS(lọc - centroid đo)={np.sqrt((d**2).mean()):.3f} m | "
              f"tốc độ TB={v.mean():.2f} m/s, max={v.max():.2f} m/s | "
              f"gia tốc sai phân bậc 2 (median)={np.median(acc):.2f} m/s^2")