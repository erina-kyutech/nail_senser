import datetime

from matplotlib import animation

# import validation_VGG16  # (old) replaced by direct model loader for concat 150x290
import axis_satuei_4houkou_2cam_concat as axis_satuei_4houkou  # concat capture util
import cv2
import matplotlib.pylab as plt
import time
import numpy as np
import pandas as pd
import math
import os
from tensorflow.keras.models import model_from_json


import random

# ----------------------------- Config (EDIT HERE) -----------------------------
# 学習に使った被験者名（重みファイル名に使う）
SUBJECT_NAME = "ifuku"   # weight/weight_<name>_for0-10.h5 の <name>

# リアルタイム推定したいモード（学習済みがあるものだけ）
# 例: ["rgb","hs","g"] で3本同時推定（重いのでFPSは落ちる）
RUN_MODES = ["rgb", "hs", "g"]

# concat後の最終サイズ（axis_satuei_4houkou_2cam_concat と揃える）
OUT_H = 150
OUT_W_LEFT = 150
OUT_W_RIGHT = 140
OUT_W = OUT_W_LEFT + OUT_W_RIGHT  # 290

# 2台カメラ設定（OpenCVのindex。環境によって 0/1 が逆の場合あり）
CAM_LEFT_INDEX = 0   # 例：爪（Nail）側
CAM_RIGHT_INDEX = 1  # 例：指先（Tip）側

# ---- ROI（中心 cx, cy と w, h）----
# axis_satuei_4houkou_2cam_concat で使っている値をそのままコピペ
# Nail（爪カメラ）
N_CX, N_CY = 499, 250
N_W, N_H = int(282 * 1.3), int(409 * 0.9)

# Tip（指先カメラ）
T_CX, T_CY = 324, 550
T_W, T_H = int(182 * 1.7), int(136 * 1.0)


# ROIスケール（見切れ/倍率調整）
N_W_SCALE, N_H_SCALE = 1.0, 1.0
T_W_SCALE, T_H_SCALE = 1.0, 1.0

# 学習結果フォルダ（modeごとに指定）
# ※あなたの result/CNN_result 配下のフォルダ名に合わせて変更してOK
MODEL_DIRS = {
    "rgb": r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_rgb_concat_150x290",
    "hs":  r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_hs_concat_150x290",
    "g":   r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_g_concat_150x290",
}

# -----------------------------------------------------------------------------

def load_trained_model(model_dir: str, subject_name: str):
    """simple_learning_houkou.py の保存形式:
    - 構造: <model_dir>/for0-10.json
    - 重み: <model_dir>/weight/weight_<name>_for0-10.h5
    """
    model_json_path = os.path.join(model_dir, "for0-10.json")
    weight_path = os.path.join(model_dir, "weight", f"weight_{subject_name}_for0-10.h5")

    if not os.path.exists(model_json_path):
        raise FileNotFoundError(f"model json not found: {model_json_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weight not found: {weight_path}")

    model_json_string = open(model_json_path, "r", encoding="utf-8").read()
    model = model_from_json(model_json_string, custom_objects={})
    model.load_weights(weight_path)
    return model

def _clip_roi(x: int, y: int, w: int, h: int, W: int, H: int):
    """画像外にはみ出さないようにROIをクリップして返す（x,y,w,h）"""
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
    """2枚のBGR画像から ROI -> resize -> hconcat -> (OUT_H, OUT_W, 3) を作る"""
    x1, y1, w1, h1 = _roi_from_center(N_CX, N_CY, N_W, N_H, N_W_SCALE, N_H_SCALE, img_left_bgr.shape)
    x2, y2, w2, h2 = _roi_from_center(T_CX, T_CY, T_W, T_H, T_W_SCALE, T_H_SCALE, img_right_bgr.shape)

    roi_l = img_left_bgr[y1:y1 + h1, x1:x1 + w1]
    roi_r = img_right_bgr[y2:y2 + h2, x2:x2 + w2]

    roi_l = cv2.resize(roi_l, (OUT_W_LEFT, OUT_H), interpolation=cv2.INTER_AREA)
    roi_r = cv2.resize(roi_r, (OUT_W_RIGHT, OUT_H), interpolation=cv2.INTER_AREA)

    return cv2.hconcat([roi_l, roi_r])  # (OUT_H, OUT_W, 3)

def preprocess_for_mode(concat_bgr, mode: str):
    """concat画像(BGR) -> mode別の入力テンソル (1,H,W,3) float32"""
    if mode == "rgb":
        rgb = cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32)[None, ...] / 255.0

    if mode == "g":
        # Gチャネル + Gaussian + CLAHE → 3ch化（simple_learning側と同系統）
        g = concat_bgr[:, :, 1]
        gau = cv2.GaussianBlur(g, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        hist = clahe.apply(gau)
        g3 = np.stack([hist, hist, hist], axis=2)
        return g3.astype(np.float32)[None, ...] / 255.0

    if mode == "hs":
        hsv = cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        Z = np.zeros_like(H)
        hs3 = np.stack([H, S, Z], axis=2)
        return hs3.astype(np.float32)[None, ...] / 255.0

    raise ValueError(f"Unknown mode: {mode}. Use 'rgb', 'hs', or 'g'.")

from matplotlib.pyplot import MultipleLocator
import csv
from multiprocessing import Process,Value,Manager

gf2000 = axis_satuei_4houkou.gf2000
SC800IM700_1 = axis_satuei_4houkou.SC800IM700_1
SC800IM700_2 = axis_satuei_4houkou.SC800IM700_2
SC800IM700_3 = axis_satuei_4houkou.SC800IM700_3
SC800IM700_4 = axis_satuei_4houkou.SC800IM700_4

class RealTime:

    def __init__(self):
        self.gf2000 = axis_satuei_4houkou.gf2000
        self.SC800IM700_1 = axis_satuei_4houkou.SC800IM700_1
        self.SC800IM700_2 = axis_satuei_4houkou.SC800IM700_2
        self.SC800IM700_3 = axis_satuei_4houkou.SC800IM700_3
        self.SC800IM700_4 = axis_satuei_4houkou.SC800IM700_4
        # --- モード別モデルをロード（重いので必要なものだけ RUN_MODES に入れる）---
        self.models = {}
        for m in RUN_MODES:
            if m not in MODEL_DIRS:
                raise ValueError(f"MODEL_DIRS に mode='{m}' がありません")
            self.models[m] = load_trained_model(MODEL_DIRS[m], SUBJECT_NAME)

        self.capture_left = cv2.VideoCapture(CAM_LEFT_INDEX)
        self.capture_right = cv2.VideoCapture(CAM_RIGHT_INDEX)
        if self.capture_left.isOpened() is False or self.capture_right.isOpened() is False:
            raise ("IO Error")

        self.normal_force_normalize = 10.0
        self.thear_force_normalize = 5.0
        self.angle_normalize = 90

        self.N2gf = 101.972

        self.datalog_path = "./datalog_fr.csv"
        self.data_csv = open(self.datalog_path, "w", newline="")
        self.data_writing = csv.writer(self.data_csv)

        print("-180~180の範囲で剪断方向角度を入力してください(例:-45)")
        self.degree_str = input()
        self.theta = math.radians(int(self.degree_str)) #角度のラジアン,剪断力の分解に利用

        self.now = datetime.datetime.now()

    def graph(self):
        # ===== 3x3 plot init (rows=mode, cols=Fz/Fx/Fy) =====
        plt.ion()
        fig, axes = plt.subplots(len(RUN_MODES), 3, figsize=(12, 8), sharex=True)
        if len(RUN_MODES) == 1:
            axes = np.expand_dims(axes, axis=0)

        force_names = ["Fz", "Fx", "Fy"]

        hist_true = {"Fz": [], "Fx": [], "Fy": []}
        hist_pred = {m: {"Fz": [], "Fx": [], "Fy": []} for m in RUN_MODES}
        hist_time = []

        show_num = 100

        # --------CSV header-------------
        header = ['Time', 'Fx_True', 'Fy_True', 'Fz_True']
        for m in RUN_MODES:
            header += [f'Fx_Predict({m})', f'Fy_Predict({m})', f'Fz_Predict({m})',
                       f'Fx_Error({m})', f'Fy_Error({m})', f'Fz_Error({m})']
        self.data_writing.writerow(header)

        star_time = time.perf_counter()

        while True:
            # ----- true force -----
            Fz = normal_force.value / self.N2gf
            Fr = shear_force1.value - shear_force3.value
            Ff = shear_force2.value - shear_force4.value

            # 真値（スカラー）
            Fz_true = float(Fz)
            Fx_true = float(Fr)
            Fy_true = float(Ff)

            # ----- capture -----
            ret_l, base_l = self.capture_left.read()
            ret_r, base_r = self.capture_right.read()
            if (not ret_l) or (not ret_r):
                continue

            concat_bgr = make_concat_bgr(base_l, base_r)
            cv2.imshow('concat', concat_bgr)

            # ----- time -----
            end_time = time.perf_counter()
            delta_time = end_time - star_time

            # ----- predict all modes -----
            preds = {}
            for mode, model in self.models.items():
                X = preprocess_for_mode(concat_bgr, mode)
                y_list = model.predict(X, verbose=0)
                y = np.concatenate([y_list[0], y_list[1], y_list[2]], axis=1)
                y = self.data_unnormalize(y)  # (1,3) in N
                preds[mode] = y

            # ===== append history =====
            hist_time.append(delta_time)
            hist_true["Fz"].append(Fz_true)
            hist_true["Fx"].append(Fx_true)
            hist_true["Fy"].append(Fy_true)

            for m in RUN_MODES:
                p = preds[m]
                hist_pred[m]["Fz"].append(float(p[0, 0]))
                hist_pred[m]["Fx"].append(float(p[0, 1]))
                hist_pred[m]["Fy"].append(float(p[0, 2]))

            # keep last show_num
            if len(hist_time) > show_num:
                hist_time.pop(0)
                for k in ["Fz", "Fx", "Fy"]:
                    hist_true[k].pop(0)
                for m in RUN_MODES:
                    for k in ["Fz", "Fx", "Fy"]:
                        hist_pred[m][k].pop(0)

            # ===== write CSV row =====
            row = [delta_time, Fx_true, Fy_true, Fz_true]
            for m in RUN_MODES:
                p = preds[m]
                Fz_pred = float(p[0, 0])
                Fx_pred = float(p[0, 1])
                Fy_pred = float(p[0, 2])

                Fz_err = Fz_pred - Fz_true
                Fx_err = Fx_pred - Fx_true
                Fy_err = Fy_pred - Fy_true

                row += [Fx_pred, Fy_pred, Fz_pred, Fx_err, Fy_err, Fz_err]

            self.data_writing.writerow(row)

            # ===== redraw 3x3 =====
            for r, m in enumerate(RUN_MODES):
                for c, k in enumerate(force_names):
                    ax = axes[r, c]
                    ax.cla()
                    ax.plot(hist_time, hist_true[k], ls=":")
                    ax.plot(hist_time, hist_pred[m][k], ls=":")
                    if r == 0:
                        ax.set_title(k)
                    if c == 0:
                        ax.set_ylabel(m)

            fig.tight_layout()
            plt.pause(0.01)

            # ESCで抜けたいならこれ（任意）
            if cv2.waitKey(1) & 0xFF == 27:
                break

    def data_unnormalize(self, Y):
        #垂直力を戻す
        Y[:, 0] *= self.normal_force_normalize
        #せんだん力の正規化
        Y[:, 1] *= (self.thear_force_normalize*2)
        Y[:, 2] *= (self.thear_force_normalize*2)
        Y[:, 1] -= self.thear_force_normalize
        Y[:, 2] -= self.thear_force_normalize

        return Y

if __name__ == "__main__":
    # --------メモリ共有変数-------------
    normal_force = Value('f', 0.00)
    shear_force1 = Value('f', 0.00)
    shear_force2 = Value('f', 0.00)
    shear_force3 = Value('f', 0.00)
    shear_force4 = Value('f', 0.00)
    ser_flag = Value('b', True)  # シリアル通信フラグ(Trueで荷重計，MD共にループ開始)
    rec_flag = Value('b', False)  # 測定フラグ(これがTrueの間測定)

    # -----------------------------------

    # ロードセル測定準備
    xy_port_1 = "COM8"
    xy_address_1 = 0x2A
    shear_loadcell_1 = SC800IM700_1(xy_port_1, xy_address_1)  # クラスの定義
    shear_loadcell_1.power_on()  # ロードセルの通信開始
    shear_loadcell_1.sub_ready()  # サブプロセスの準備

    # ロードセル測定準備
    xy_port_2 = "COM9"
    xy_address_2 = 0x2A
    shear_loadcell_2 = SC800IM700_2(xy_port_2, xy_address_2)  # クラスの定義
    shear_loadcell_2.power_on()  # ロードセルの通信開始
    shear_loadcell_2.sub_ready()  # サブプロセスの準備

    # ロードセル測定準備
    xy_port_3 = "COM10"
    xy_address_3 = 0x2A
    shear_loadcell_3 = SC800IM700_3(xy_port_3, xy_address_3)  # クラスの定義
    shear_loadcell_3.power_on()  # ロードセルの通信開始
    shear_loadcell_3.sub_ready()  # サブプロセスの準備

    # ロードセル測定準備
    xy_port_4 = "COM12"
    xy_address_4 = 0x2A
    shear_loadcell_4 = SC800IM700_4(xy_port_4, xy_address_4)  # クラスの定義
    shear_loadcell_4.power_on()  # ロードセルの通信開始
    shear_loadcell_4.sub_ready()  # サブプロセスの準備

    # 荷重計測定準備
    z_port = "COM14"
    normal_loadcell = gf2000(z_port)
    normal_loadcell.sub_ready()

    """
    サブプロセス開始(各種通信)
    並列処理したい関数がクラス内の関数(メソッド)の場合エラーが起きる
    その場合，メソッドをクラスメソッドとして定義してやると動かせる
    """
    sub_z = Process(target=gf2000.sub_loop, args=[z_port, ser_flag, normal_force])
    sub_z.start()
    count1 = Value('i', 0)
    count2 = Value('i', 0)
    count3 = Value('i', 0)
    count4 = Value('i', 0)

    # せん断力計測プロセスの定義
    sub_xy1 = Process(target=SC800IM700_1.sub_loop,
                      args=[xy_port_1, xy_address_1, ser_flag, shear_force1, count1])
    sub_xy2 = Process(target=SC800IM700_2.sub_loop,
                      args=[xy_port_2, xy_address_2, ser_flag, shear_force2, count2])
    sub_xy3 = Process(target=SC800IM700_3.sub_loop,
                      args=[xy_port_3, xy_address_3, ser_flag, shear_force3, count3])
    sub_xy4 = Process(target=SC800IM700_4.sub_loop,
                      args=[xy_port_4, xy_address_4, ser_flag, shear_force4, count4])

    # せん断力計測プロセスの開始
    sub_xy1.start()
    sub_xy2.start()
    sub_xy3.start()
    sub_xy4.start()

    RealTime().graph()

# -----------------------------------