import os
import cv2
import time
import csv
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Value
from tensorflow.keras.models import model_from_json

import axis_satuei_4houkou_2cam_concat as axis_satuei_4houkou  # あなたの2cam撮影util

# ----------------------------- Config (EDIT HERE) -----------------------------
SUBJECT_NAME = "ifuku"
RUN_MODES = ["rgb", "hs", "g"]  # 3本同時

# concat出力サイズ（撮影プログラムと同じ）
OUT_H = 150
OUT_W_LEFT = 150
OUT_W_RIGHT = 140
OUT_W = OUT_W_LEFT + OUT_W_RIGHT  # 290

# カメラindex（環境で0/1逆なら入れ替えて）
CAM_LEFT_INDEX = 0   # nail
CAM_RIGHT_INDEX = 1  # tip

# ---- ROI（生値）----
N_CX, N_CY = 499, 250
N_W0, N_H0 = 282, 409

T_CX, T_CY = 324, 550
T_W0, T_H0 = 182, 136

# ---- ROIスケール（撮影プログラムと同じ）----
N_W_SCALE, N_H_SCALE = 1.3, 0.9
T_W_SCALE, T_H_SCALE = 1.7, 1.0

# 学習結果フォルダ（modeごと）
MODEL_DIRS = {
    "rgb": r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_rgb_concat_150x290",
    "hs":  r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_hs_concat_150x290",
    "g":   r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_g_concat_150x290",
}
# -----------------------------------------------------------------------------

def crop_with_center_wh_safe(img, cx, cy, w, h):
    H, W = img.shape[:2]

    w = int(max(1, min(w, W)))
    h = int(max(1, min(h, H)))

    cx = int(max(w // 2, min(cx, W - w // 2)))
    cy = int(max(h // 2, min(cy, H - h // 2)))

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = x1 + w
    y2 = y1 + h

    return img[y1:y2, x1:x2], (x1, y1, w, h)

def _resize_no_pad_center_crop(img, out_w, out_h):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    target = out_w / out_h
    cur = w / h

    if cur > target:
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        cropped = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        cropped = img[y0:y0 + new_h, :]

    interp = cv2.INTER_AREA if (cropped.shape[0] > out_h or cropped.shape[1] > out_w) else cv2.INTER_LINEAR
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)

def load_trained_model(model_dir: str, subject_name: str):
    model_json_path = os.path.join(model_dir, "for0-10.json")
    weight_path = os.path.join(model_dir, "weight", f"weight_{subject_name}_for0-10.h5")

    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"model json not found: {model_json_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weight not found: {weight_path}")

    with open(model_json_path, "r", encoding="utf-8") as f:
        model_json_string = f.read()

    model = model_from_json(model_json_string, custom_objects={})
    model.load_weights(weight_path)
    return model


def _clip_roi(x: int, y: int, w: int, h: int, W: int, H: int):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def _roi_from_center(cx: int, cy: int, w: int, h: int, w_scale: float, h_scale: float, img_shape):
    H, W = img_shape[:2]
    ww = int(w * w_scale)
    hh = int(h * h_scale)
    x = int(cx - ww / 2)
    y = int(cy - hh / 2)
    return _clip_roi(x, y, ww, hh, W, H)


def make_concat_bgr(img_left_bgr, img_right_bgr):
    n_w = int(N_W0 * N_W_SCALE)
    n_h = int(N_H0 * N_H_SCALE)
    t_w = int(T_W0 * T_W_SCALE)
    t_h = int(T_H0 * T_H_SCALE)

    roi_l, rect_l = crop_with_center_wh_safe(img_left_bgr,  N_CX, N_CY, n_w, n_h)
    roi_r, rect_r = crop_with_center_wh_safe(img_right_bgr, T_CX, T_CY, t_w, t_h)

    # ★ 撮影と同じ整形（歪みゼロ）
    roi_l = _resize_no_pad_center_crop(roi_l, OUT_W_LEFT, OUT_H)
    roi_r = _resize_no_pad_center_crop(roi_r, OUT_W_RIGHT, OUT_H)

    return cv2.hconcat([roi_l, roi_r])

def preprocess_for_mode(concat_bgr, mode: str):
    if mode == "rgb":
        rgb = cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32)[None, ...] / 255.0

    if mode == "g":
        g = concat_bgr[:, :, 1]
        gau = cv2.GaussianBlur(g, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        hist = clahe.apply(gau)
        g3 = np.stack([hist, hist, hist], axis=2)
        return g3.astype(np.float32)[None, ...] / 255.0

    if mode == "hs":
        hsv = cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        Z = np.zeros_like(H, dtype=np.float32)
        hs3 = np.stack([H, S, Z], axis=2)
        return hs3.astype(np.float32)[None, ...] / 255.0

    raise ValueError(f"Unknown mode: {mode}")


# -------- sensors (from axis_satuei_4houkou_2cam_concat) --------
gf2000 = axis_satuei_4houkou.gf2000
SC800IM700_1 = axis_satuei_4houkou.SC800IM700_1
SC800IM700_2 = axis_satuei_4houkou.SC800IM700_2
SC800IM700_3 = axis_satuei_4houkou.SC800IM700_3
SC800IM700_4 = axis_satuei_4houkou.SC800IM700_4


class RealTime:
    def __init__(self):
        # models
        self.models = {m: load_trained_model(MODEL_DIRS[m], SUBJECT_NAME) for m in RUN_MODES}

        # camera
        self.cap_l = cv2.VideoCapture(CAM_LEFT_INDEX)
        self.cap_r = cv2.VideoCapture(CAM_RIGHT_INDEX)
        if (not self.cap_l.isOpened()) or (not self.cap_r.isOpened()):
            raise RuntimeError("Camera open failed. (index 0/1 reversed?)")

        # ---- OpenCV window setting (重要) ----
        cv2.namedWindow("concat", cv2.WINDOW_NORMAL)
        cv2.moveWindow("concat", 50, 50)        # 左上に固定
        cv2.setWindowProperty("concat", cv2.WND_PROP_TOPMOST, 1)

        # ---- FPS ----
        self.prev_time = time.perf_counter()
        self.fps_ema = 0.0
        self.fps_alpha = 0.1  # 0.05〜0.2くらいが安定


        # normalize (学習時と同じ)
        self.normal_force_normalize = 10.0
        self.thear_force_normalize = 5.0

        self.N2gf = 101.972  # g -> N

        # csv
        self.datalog_path = "./datalog_fr_3modes.csv"
        self.data_csv = open(self.datalog_path, "w", newline="")
        self.w = csv.writer(self.data_csv)

        header = ["Time", "Fx_True", "Fy_True", "Fz_True"]
        for m in RUN_MODES:
            header += [f"Fx_Pred({m})", f"Fy_Pred({m})", f"Fz_Pred({m})",
                       f"Fx_Err({m})", f"Fy_Err({m})", f"Fz_Err({m})"]
        self.w.writerow(header)

        # angle input
        print("-180~180の範囲で剪断方向角度を入力してください(例:-45)")
        deg = int(input())
        self.theta = math.radians(deg)

        # plot init
        plt.ion()
        self.force_names = ["Fz", "Fx", "Fy"]
        self.fig, self.axes = plt.subplots(len(RUN_MODES), 3, figsize=(11, 7), sharex=True)

        # history buffers
        self.show_num = 100
        self.hist_time = []
        self.hist_true = {k: [] for k in self.force_names}
        self.hist_pred = {m: {k: [] for k in self.force_names} for m in RUN_MODES}

        # lines: true/pred (set_data方式)
        self.lines_true = {k: [] for k in self.force_names}  # k -> list (row)
        self.lines_pred = {m: {k: None for k in self.force_names} for m in RUN_MODES}



        for r, m in enumerate(RUN_MODES):
            for c, k in enumerate(self.force_names):
                ax = self.axes[r, c]
                lt, = ax.plot([], [], ls=":")  # true
                lp, = ax.plot([], [], ls=":")  # pred
                self.lines_true[k].append(lt)
                self.lines_pred[m][k] = lp
                if r == 0:
                    ax.set_title(k)
                if c == 0:
                    ax.set_ylabel(m)
        self.fig.tight_layout()

    def data_unnormalize(self, Y):
        Y = Y.copy()
        Y[:, 0] *= self.normal_force_normalize
        Y[:, 1] *= (self.thear_force_normalize * 2)
        Y[:, 2] *= (self.thear_force_normalize * 2)
        Y[:, 1] -= self.thear_force_normalize
        Y[:, 2] -= self.thear_force_normalize
        return Y

    def _update_plot(self):
        if len(self.hist_time) < 2:
            return

        t0, t1 = self.hist_time[0], self.hist_time[-1]
        if t1 == t0:
            t1 = t0 + 1e-6

        for r, m in enumerate(RUN_MODES):
            for c, k in enumerate(self.force_names):
                ax = self.axes[r, c]
                # set_data only (NO cla/plot)
                self.lines_true[k][r].set_data(self.hist_time, self.hist_true[k])
                self.lines_pred[m][k].set_data(self.hist_time, self.hist_pred[m][k])

                ax.set_xlim(t0, t1)

                # y auto adjust (optional): 少し余白
                y_all = np.array(self.hist_true[k] + self.hist_pred[m][k], dtype=np.float32)
                if y_all.size > 0:
                    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
                    if ymin == ymax:
                        ymin -= 1.0
                        ymax += 1.0
                    pad = 0.1 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def loop(self):
        start = time.perf_counter()

        try:
            while True:
                # FPS
                now_fps_time = time.perf_counter()
                dt = now_fps_time - self.prev_time
                self.prev_time = now_fps_time
                if dt > 0:
                    fps_inst = 1.0 / dt
                    self.fps_ema = (1 - self.fps_alpha) * self.fps_ema + self.fps_alpha * fps_inst

                # capture
                ret_l, base_l = self.cap_l.read()
                ret_r, base_r = self.cap_r.read()
                if not ret_l or not ret_r:
                    continue

                concat_bgr = make_concat_bgr(base_l, base_r)

                cv2.putText(concat_bgr, f"FPS: {self.fps_ema:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

                cv2.imshow("concat", concat_bgr)

                # ---- time ----
                now = time.perf_counter()
                t = now - start

                # ---- true forces ----
                Fz_true = float(normal_force.value / self.N2gf)
                Fr = float(shear_force1.value - shear_force3.value)
                Ff = float(shear_force2.value - shear_force4.value)

                # ここは「真値」をどう定義するかで変わる
                # いまはあなたの貼ってた通り Fr/Ff をそのまま Fx/Fy にしてる
                Fx_true = float(Fr)
                Fy_true = float(Ff)

                # ---- predict ----
                preds = {}
                for m, model in self.models.items():
                    X = preprocess_for_mode(concat_bgr, m)
                    y_list = model.predict(X, verbose=0)
                    y = np.concatenate([y_list[0], y_list[1], y_list[2]], axis=1)  # (1,3)
                    y = self.data_unnormalize(y)
                    preds[m] = y

                # ---- history push ----
                self.hist_time.append(t)
                self.hist_true["Fz"].append(Fz_true)
                self.hist_true["Fx"].append(Fx_true)
                self.hist_true["Fy"].append(Fy_true)

                for m in RUN_MODES:
                    p = preds[m]
                    self.hist_pred[m]["Fz"].append(float(p[0, 0]))
                    self.hist_pred[m]["Fx"].append(float(p[0, 1]))
                    self.hist_pred[m]["Fy"].append(float(p[0, 2]))

                # keep last N
                if len(self.hist_time) > self.show_num:
                    self.hist_time.pop(0)
                    for k in self.force_names:
                        self.hist_true[k].pop(0)
                    for m in RUN_MODES:
                        for k in self.force_names:
                            self.hist_pred[m][k].pop(0)

                # ---- CSV ----
                row = [t, Fx_true, Fy_true, Fz_true]
                for m in RUN_MODES:
                    p = preds[m]
                    Fz_pred = float(p[0, 0])
                    Fx_pred = float(p[0, 1])
                    Fy_pred = float(p[0, 2])
                    row += [Fx_pred, Fy_pred, Fz_pred,
                            Fx_pred - Fx_true, Fy_pred - Fy_true, Fz_pred - Fz_true]
                self.w.writerow(row)

                # ---- plot update ----
                self._update_plot()

                # ESC to quit
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            # release
            try:
                self.data_csv.close()
            except:
                pass
            self.cap_l.release()
            self.cap_r.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.close("all")


if __name__ == "__main__":
    # -------- shared memory vars --------
    normal_force = Value("f", 0.00)
    shear_force1 = Value("f", 0.00)
    shear_force2 = Value("f", 0.00)
    shear_force3 = Value("f", 0.00)
    shear_force4 = Value("f", 0.00)
    ser_flag = Value("b", True)

    # --- ports (あなたの環境) ---
    xy_port_1 = "COM8"
    xy_port_2 = "COM9"
    xy_port_3 = "COM10"
    xy_port_4 = "COM12"
    xy_address = 0x2A

    z_port = "COM14"

    # --- sensor init ---
    shear_loadcell_1 = SC800IM700_1(xy_port_1, xy_address); shear_loadcell_1.power_on(); shear_loadcell_1.sub_ready()
    shear_loadcell_2 = SC800IM700_2(xy_port_2, xy_address); shear_loadcell_2.power_on(); shear_loadcell_2.sub_ready()
    shear_loadcell_3 = SC800IM700_3(xy_port_3, xy_address); shear_loadcell_3.power_on(); shear_loadcell_3.sub_ready()
    shear_loadcell_4 = SC800IM700_4(xy_port_4, xy_address); shear_loadcell_4.power_on(); shear_loadcell_4.sub_ready()

    normal_loadcell = gf2000(z_port); normal_loadcell.sub_ready()

    # --- subprocess start ---
    sub_z = Process(target=gf2000.sub_loop, args=[z_port, ser_flag, normal_force])
    sub_z.start()

    count1 = Value("i", 0); count2 = Value("i", 0); count3 = Value("i", 0); count4 = Value("i", 0)

    sub_xy1 = Process(target=SC800IM700_1.sub_loop, args=[xy_port_1, xy_address, ser_flag, shear_force1, count1])
    sub_xy2 = Process(target=SC800IM700_2.sub_loop, args=[xy_port_2, xy_address, ser_flag, shear_force2, count2])
    sub_xy3 = Process(target=SC800IM700_3.sub_loop, args=[xy_port_3, xy_address, ser_flag, shear_force3, count3])
    sub_xy4 = Process(target=SC800IM700_4.sub_loop, args=[xy_port_4, xy_address, ser_flag, shear_force4, count4])

    sub_xy1.start(); sub_xy2.start(); sub_xy3.start(); sub_xy4.start()

    RealTime().loop()
