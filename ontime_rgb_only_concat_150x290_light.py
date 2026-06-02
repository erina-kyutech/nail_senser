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

import axis_satuei_rgb_only_concat as axis_satuei_4houkou

# ----------------------------- Config (EDIT HERE) -----------------------------
SUBJECT_NAME = "ifuku"
RUN_MODES = ["rgb"]  # RGBのみ

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
}

# 軽量化用
PLOT_EVERY_N_FRAMES = 5      # グラフ更新は5フレームに1回
SHOW_SEC = 10.0                # 履歴を少し短く
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

def preprocess_for_mode(concat_bgr):
    rgb = cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[None, ...] / 255.0


# -------- sensors (from axis_satuei_4houkou_2cam_concat) --------
gf2000 = axis_satuei_4houkou.gf2000
SC800IM700_1 = axis_satuei_4houkou.SC800IM700_1
SC800IM700_2 = axis_satuei_4houkou.SC800IM700_2
SC800IM700_3 = axis_satuei_4houkou.SC800IM700_3
SC800IM700_4 = axis_satuei_4houkou.SC800IM700_4


class RealTime:
    def __init__(self):
        # model (RGB only)
        self.model = load_trained_model(MODEL_DIRS["rgb"], SUBJECT_NAME)

        # camera
        self.cap_l = cv2.VideoCapture(CAM_LEFT_INDEX)
        self.cap_r = cv2.VideoCapture(CAM_RIGHT_INDEX)
        if (not self.cap_l.isOpened()) or (not self.cap_r.isOpened()):
            raise RuntimeError("Camera open failed. (index 0/1 reversed?)")

        cv2.namedWindow("concat", cv2.WINDOW_NORMAL)
        cv2.moveWindow("concat", 50, 50)
        cv2.setWindowProperty("concat", cv2.WND_PROP_TOPMOST, 1)

        # FPS
        self.prev_time = time.perf_counter()
        self.fps_ema = 0.0
        self.fps_alpha = 0.1

        # normalize
        self.normal_force_normalize = 10.0
        self.thear_force_normalize = 5.0
        self.N2gf = 101.972  # g -> N

        # csv
        self.datalog_path = "./datalog_fr_rgb.csv"
        self.data_csv = open(self.datalog_path, "w", newline="")
        self.w = csv.writer(self.data_csv)
        header = [
            "Time", "Fx_True", "Fy_True", "Fz_True",
            "Fx_Pred(rgb)", "Fy_Pred(rgb)", "Fz_Pred(rgb)",
            "Fx_Err(rgb)", "Fy_Err(rgb)", "Fz_Err(rgb)"
        ]
        self.w.writerow(header)

        # plot init: 横長にする
        plt.ion()
        self.force_names = ["Fz", "Fx", "Fy"]
        self.fig, self.axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        self.axes = np.expand_dims(self.axes, axis=0)

        # history buffers
        self.show_sec = SHOW_SEC
        self.plot_counter = 0
        self.hist_time = []
        self.hist_true = {k: [] for k in self.force_names}
        self.hist_pred = {k: [] for k in self.force_names}

        # lines
        self.lines_true = {}
        self.lines_pred = {}
        for c, k in enumerate(self.force_names):
            ax = self.axes[0, c]
            lt, = ax.plot([], [], ls=":", label="true")
            lp, = ax.plot([], [], ls="-", label="pred")
            self.lines_true[k] = lt
            self.lines_pred[k] = lp
            ax.set_title(k)
            if c == 0:
                ax.set_ylabel("rgb")
        self.fig.tight_layout()

        # control
        self.is_recording = False
        self.start_time = None
        self.Fr0 = 0.0
        self.Ff0 = 0.0

        print("準備OK：指の位置を確認して 'r' を押すと測定開始")

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

        t1 = self.hist_time[-1]
        t0 = max(0.0, t1 - self.show_sec)

        if t1 == t0:
            t1 = t0 + 1e-6

        for c, k in enumerate(self.force_names):
            ax = self.axes[0, c]
            self.lines_true[k].set_data(self.hist_time, self.hist_true[k])
            self.lines_pred[k].set_data(self.hist_time, self.hist_pred[k])
            ax.set_xlim(t0, t1)

            y_all = np.array(self.hist_true[k] + self.hist_pred[k], dtype=np.float32)
            if y_all.size > 0:
                ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                pad = 0.12 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def loop(self):
        try:
            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r') and not self.is_recording:
                    print("=== 測定開始 ===")
                    time.sleep(1.0)
                    self.Fr0 = shear_force1.value - shear_force3.value
                    self.Ff0 = shear_force2.value - shear_force4.value
                    self.start_time = time.perf_counter()
                    self.is_recording = True

                if key == 27:
                    break

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

                # display
                disp_bgr = concat_bgr.copy()
                cv2.putText(disp_bgr, f"FPS: {self.fps_ema:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
                cv2.imshow("concat", disp_bgr)

                if self.is_recording:
                    t = time.perf_counter() - self.start_time

                    # true forces
                    Fz_true = float(normal_force.value / self.N2gf)
                    Fx_true = (shear_force1.value - shear_force3.value) - self.Fr0
                    Fy_true = (shear_force2.value - shear_force4.value) - self.Ff0

                    # predict (RGB only, direct call for lighter inference)
                    X = preprocess_for_mode(concat_bgr)
                    y_list = self.model(X, training=False)
                    y = np.concatenate([arr.numpy() for arr in y_list], axis=1)
                    y = self.data_unnormalize(y)

                    Fz_pred = float(y[0, 0])
                    Fx_pred = float(y[0, 1])
                    Fy_pred = float(y[0, 2])

                    # history
                    self.hist_time.append(t)
                    self.hist_true["Fz"].append(Fz_true)
                    self.hist_true["Fx"].append(Fx_true)
                    self.hist_true["Fy"].append(Fy_true)
                    self.hist_pred["Fz"].append(Fz_pred)
                    self.hist_pred["Fx"].append(Fx_pred)
                    self.hist_pred["Fy"].append(Fy_pred)


                    while len(self.hist_time) > 0 and (self.hist_time[-1] - self.hist_time[0]) > self.show_sec:
                        self.hist_time.pop(0)
                        for k in self.force_names:
                            self.hist_true[k].pop(0)
                            self.hist_pred[k].pop(0)


                    # csv
                    row = [
                        t, Fx_true, Fy_true, Fz_true,
                        Fx_pred, Fy_pred, Fz_pred,
                        Fx_pred - Fx_true, Fy_pred - Fy_true, Fz_pred - Fz_true
                    ]
                    self.w.writerow(row)

                    # plot update every N frames
                    self.plot_counter += 1
                    if self.plot_counter % PLOT_EVERY_N_FRAMES == 0:
                        self._update_plot()

        finally:
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

    z_port = "COM15"

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
