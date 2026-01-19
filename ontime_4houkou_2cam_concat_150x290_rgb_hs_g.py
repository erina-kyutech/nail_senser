import datetime
import cv2
import matplotlib.pylab as plt
import time
import numpy as np
import math
import os
import csv
from multiprocessing import Process, Value

from tensorflow.keras.models import model_from_json

# concat capture util（あなたの環境のファイル名に合わせてね）
import axis_satuei_4houkou_2cam_concat as axis_satuei_4houkou

# ----------------------------- Config (EDIT HERE) -----------------------------
SUBJECT_NAME = "ifuku"

# 推定したいモード（学習済みがあるものだけ）
RUN_MODES = ["rgb", "hs", "g"]

# グラフ/CSV/誤差計算に使う代表モード（3本推定しても、ここで指定した1本を使う）
PLOT_MODE = "rgb"   # ← "rgb" 推奨。 "hs" や "g" にしてもOK。

# concat後の最終サイズ
OUT_H = 150
OUT_W_LEFT = 150
OUT_W_RIGHT = 140
OUT_W = OUT_W_LEFT + OUT_W_RIGHT  # 290

# カメラ設定
CAM_LEFT_INDEX = 0   # 爪
CAM_RIGHT_INDEX = 1  # 指先

# ROI（中心 cx, cy と w, h）
N_CX, N_CY = 499, 250
N_W, N_H = 282, 409
T_CX, T_CY = 324, 550
T_W, T_H = 182, 136

# ROIスケール
N_W_SCALE, N_H_SCALE = 1.3, 0.9
T_W_SCALE, T_H_SCALE = 1.7, 1.0

# 学習結果フォルダ（必ず実際のフォルダ名に一致させる）
MODEL_DIRS = {
    "rgb": r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_rgb_concat_150x290",
    "hs":  r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_hs_concat_150x290",
    "g":   r"C:\Users\Owner\PycharmProjects\result\CNN_result\vgg16_g_concat_150x290",
}
# -----------------------------------------------------------------------------


def load_trained_model(model_dir: str, subject_name: str):
    """保存形式:
    - 構造: <model_dir>/for0-10.json
    - 重み: <model_dir>/weight/weight_<name>_for0-10.h5
    """
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

    # 念のため入力shape表示
    print(f"[LOAD] {model_dir}")
    print(f"       input_shape = {model.input_shape}")
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


gf2000 = axis_satuei_4houkou.gf2000
SC800IM700_1 = axis_satuei_4houkou.SC800IM700_1
SC800IM700_2 = axis_satuei_4houkou.SC800IM700_2
SC800IM700_3 = axis_satuei_4houkou.SC800IM700_3
SC800IM700_4 = axis_satuei_4houkou.SC800IM700_4


class RealTime:
    def __init__(self):
        # --- モデルロード ---
        self.models = {}
        for m in RUN_MODES:
            if m not in MODEL_DIRS:
                raise ValueError(f"MODEL_DIRS に mode='{m}' がありません")
            self.models[m] = load_trained_model(MODEL_DIRS[m], SUBJECT_NAME)

        if PLOT_MODE not in self.models:
            raise ValueError(f"PLOT_MODE='{PLOT_MODE}' が RUN_MODES に含まれていません")

        # --- カメラ ---
        self.capture_left = cv2.VideoCapture(CAM_LEFT_INDEX)
        self.capture_right = cv2.VideoCapture(CAM_RIGHT_INDEX)
        if (not self.capture_left.isOpened()) or (not self.capture_right.isOpened()):
            raise RuntimeError("IO Error: camera open failed")

        # 正規化定数（学習時と一致）
        self.normal_force_normalize = 10.0
        self.thear_force_normalize = 5.0

        self.N2gf = 101.972

        self.datalog_path = "./datalog_fr.csv"
        self.data_csv = open(self.datalog_path, "w", newline="")
        self.data_writing = csv.writer(self.data_csv)

        print("-180~180の範囲で剪断方向角度を入力してください(例:-45)")
        self.degree_str = input().strip()
        self.theta = math.radians(int(self.degree_str))

        self.now = datetime.datetime.now()

    def data_unnormalize(self, Y):
        Y[:, 0] *= self.normal_force_normalize
        Y[:, 1] *= (self.thear_force_normalize * 2)
        Y[:, 2] *= (self.thear_force_normalize * 2)
        Y[:, 1] -= self.thear_force_normalize
        Y[:, 2] -= self.thear_force_normalize
        return Y

    def graph(self):
        time_list = []
        temp_list_time = []

        temp_list_Fz_Predict = []
        temp_list_Fx_Predict = []
        temp_list_Fy_Predict = []

        temp_list_Fz_True = []
        temp_list_Fx_True = []
        temp_list_Fy_True = []

        show_num = 100
        num = 0

        plt.ion()

        header = [
            'Time',
            'Fx_True', 'Fy_True', 'Fz_True',
            f'Fx_Predict({PLOT_MODE})', f'Fy_Predict({PLOT_MODE})', f'Fz_Predict({PLOT_MODE})',
            'Fx_Error', 'Fy_Error', 'Fz_Error'
        ]
        self.data_writing.writerow(header)

        start_time = time.perf_counter()

        while True:
            # センサ値
            Fz = normal_force.value / self.N2gf
            Fr = shear_force1.value - shear_force3.value
            Ff = shear_force2.value - shear_force4.value

            ret_l, base_l = self.capture_left.read()
            ret_r, base_r = self.capture_right.read()
            if (not ret_l) or (not ret_r):
                continue

            concat_bgr = make_concat_bgr(base_l, base_r)
            cv2.imshow("concat", concat_bgr)

            # --- 推定（RUN_MODES全部）---
            preds = {}  # mode -> (1,3)
            for mode, model in self.models.items():
                X = preprocess_for_mode(concat_bgr, mode)           # (1,150,290,3)
                y_list = model.predict(X, verbose=0)               # [Fz,Fx,Fy] (each (1,1))
                y = np.concatenate([y_list[0], y_list[1], y_list[2]], axis=1)  # (1,3)
                y = self.data_unnormalize(y)
                preds[mode] = y

            # 表示（3本同時）
            for mode in preds:
                # (1,3) -> スカラー
                fz = float(preds[mode][0, 0])
                fx = float(preds[mode][0, 1])
                fy = float(preds[mode][0, 2])
                print(f"[{mode}] Fz={fz:.3f}, Fx={fx:.3f}, Fy={fy:.3f}")

            # --- ここが旧コードの predict 相当（代表モードのみを後続に渡す）---
            predict = preds[PLOT_MODE]  # (1,3)  ← NameError対策の本体

            # 予測（スカラー化）
            Fz_pred = float(predict[0, 0])
            Fx_pred = float(predict[0, 1])
            Fy_pred = float(predict[0, 2])

            # 真値
            Fz_true = float(Fz)
            Fx_true = float(Fr)
            Fy_true = float(Ff)

            # 誤差
            Fz_error = Fz_pred - Fz_true
            Fx_error = Fx_pred - Fx_true
            Fy_error = Fy_pred - Fy_true

            end_time = time.perf_counter()
            delta_time = end_time - start_time

            time_list.append(delta_time)
            temp_list_time.append(delta_time)

            self.data_writing.writerow([
                delta_time,
                Fx_true, Fy_true, Fz_true,
                Fx_pred, Fy_pred, Fz_pred,
                Fx_error, Fy_error, Fz_error
            ])

            # プロット用バッファ
            temp_list_Fz_Predict.append(Fz_pred)
            temp_list_Fx_Predict.append(Fx_pred)
            temp_list_Fy_Predict.append(Fy_pred)

            temp_list_Fz_True.append(Fz_true)
            temp_list_Fx_True.append(Fx_true)
            temp_list_Fy_True.append(Fy_true)

            if num > show_num:
                temp_list_time.pop(0)

                temp_list_Fz_True.pop(0)
                temp_list_Fz_Predict.pop(0)

                temp_list_Fx_True.pop(0)
                temp_list_Fx_Predict.pop(0)

                temp_list_Fy_True.pop(0)
                temp_list_Fy_Predict.pop(0)

                plt.cla()

            integer = math.floor(delta_time)

            plt.subplot(311, title=f"Fz ({PLOT_MODE})", xlabel="t [s]", ylabel="Fz [N]")
            plt.plot(temp_list_time, temp_list_Fz_True, ls=':')
            plt.plot(temp_list_time, temp_list_Fz_Predict, ls=':')
            if integer > 10:
                plt.xlim(integer - 10, integer + 10)
            else:
                plt.xlim(0, 20)

            plt.subplot(312, title=f"Fx ({PLOT_MODE})", xlabel="t [s]", ylabel="Fx [N]")
            plt.plot(temp_list_time, temp_list_Fx_True, ls=':')
            plt.plot(temp_list_time, temp_list_Fx_Predict, ls=':')
            if integer > 10:
                plt.xlim(integer - 10, integer + 10)
            else:
                plt.xlim(0, 20)

            plt.subplot(313, title=f"Fy ({PLOT_MODE})", xlabel="t [s]", ylabel="Fy [N]")
            plt.plot(temp_list_time, temp_list_Fy_True, ls=':')
            plt.plot(temp_list_time, temp_list_Fy_Predict, ls=':')
            if integer > 10:
                plt.xlim(integer - 10, integer + 10)
            else:
                plt.xlim(0, 20)

            plt.tight_layout()
            plt.pause(0.01)

            num += 1

            # 終了キー（q）対応
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.data_csv.close()
        self.capture_left.release()
        self.capture_right.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # --------メモリ共有変数-------------
    normal_force = Value('f', 0.00)
    shear_force1 = Value('f', 0.00)
    shear_force2 = Value('f', 0.00)
    shear_force3 = Value('f', 0.00)
    shear_force4 = Value('f', 0.00)
    ser_flag = Value('b', True)
    rec_flag = Value('b', False)

    # ロードセル準備（あなたのCOMに合わせて）
    xy_port_1 = "COM8"
    xy_address_1 = 0x2A
    shear_loadcell_1 = SC800IM700_1(xy_port_1, xy_address_1)
    shear_loadcell_1.power_on()
    shear_loadcell_1.sub_ready()

    xy_port_2 = "COM9"
    xy_address_2 = 0x2A
    shear_loadcell_2 = SC800IM700_2(xy_port_2, xy_address_2)
    shear_loadcell_2.power_on()
    shear_loadcell_2.sub_ready()

    xy_port_3 = "COM10"
    xy_address_3 = 0x2A
    shear_loadcell_3 = SC800IM700_3(xy_port_3, xy_address_3)
    shear_loadcell_3.power_on()
    shear_loadcell_3.sub_ready()

    xy_port_4 = "COM12"
    xy_address_4 = 0x2A
    shear_loadcell_4 = SC800IM700_4(xy_port_4, xy_address_4)
    shear_loadcell_4.power_on()
    shear_loadcell_4.sub_ready()

    # 荷重計準備
    z_port = "COM14"
    normal_loadcell = gf2000(z_port)
    normal_loadcell.sub_ready()

    # サブプロセス開始
    sub_z = Process(target=gf2000.sub_loop, args=[z_port, ser_flag, normal_force])
    sub_z.start()

    count1 = Value('i', 0)
    count2 = Value('i', 0)
    count3 = Value('i', 0)
    count4 = Value('i', 0)

    sub_xy1 = Process(target=SC800IM700_1.sub_loop, args=[xy_port_1, xy_address_1, ser_flag, shear_force1, count1])
    sub_xy2 = Process(target=SC800IM700_2.sub_loop, args=[xy_port_2, xy_address_2, ser_flag, shear_force2, count2])
    sub_xy3 = Process(target=SC800IM700_3.sub_loop, args=[xy_port_3, xy_address_3, ser_flag, shear_force3, count3])
    sub_xy4 = Process(target=SC800IM700_4.sub_loop, args=[xy_port_4, xy_address_4, ser_flag, shear_force4, count4])

    sub_xy1.start()
    sub_xy2.start()
    sub_xy3.start()
    sub_xy4.start()

    RealTime().graph()
