import datetime

from matplotlib import animation

import validation_VGG16
import axis_satuei_4houkou
import cv2
import matplotlib.pylab as plt
import time
import numpy as np
import pandas as pd
import math


import random
from matplotlib.pyplot import MultipleLocator
import csv
from multiprocessing import Process,Value,Manager

model_rgb = validation_VGG16.load_model_for_mode("rgb")
model_g   = validation_VGG16.load_model_for_mode("g")
model_hs  = validation_VGG16.load_model_for_mode("hs")

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



        self.roix, self.roiy = 260, 315  # 左上座標
        self.w, self.h = 140, 155  # 幅,高さ

        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened() is False:
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
        time_list = []  # 用于存放x轴数据
        temp_list_time = []  # 临时存放x轴数据

        Fz_Predict_list = []
        Fz_True_list = []

        Fx_Predict_list = []
        Fx_True_list = []

        Fy_Predict_list = []
        Fy_True_list = []


        temp_list_Fz_True = []  # 临时存放y轴数据
        temp_list_Fx_True = []  # 临时存放y轴数据
        temp_list_Fy_True = []  # 临时存放y轴数据

        pred = {
            "rgb": {"Fz": [], "Fx": [], "Fy": []},
            "g": {"Fz": [], "Fx": [], "Fy": []},
            "hs": {"Fz": [], "Fx": [], "Fy": []},
        }

        show_num = 100  # x轴显示的数据个数，例：show_num = 10表示x轴只显示10个数据
        num = 0

        # plt.ion()  # 打开交互模式
        # #fig1 = plt.figure(figsize=(100, 80))  # 设置图片大小
        # plt.xlim(0, show_num)  # 设置x轴的数值显示范围
        # plt.ylim(-10, 10)
        # #x_major_locator = MultipleLocator(100000)  # 把x轴的刻度间隔设置为2
        # #y_major_locator = MultipleLocator(10)  # 把y轴的刻度间隔设置为10
        # ax = plt.gca()  # ax为两条坐标轴的实例
        # #ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为2的倍数
        # #ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数

        plt.ion()

        modes = ["rgb", "g", "hs"]
        axes_names = ["Fz", "Fx", "Fy"]

        fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)

        #title, label
        for r, mode in enumerate(modes):
            axs[r, 0].set_ylabel(mode)
        for c, nm in enumerate(axes_names):
            axs[0, c].set_title(nm)

        #9枠それぞれに　True/Pred の「線」を１回だけ作る
        lines = {}
        for r, mode in enumerate(modes):
            for c, nm in enumerate(axes_names):
                l_true, = axs[r,c].plot([], [], ls=':') #True
                l_pred, = axs[r,c].plot([], [], ls=':') #Pred
                lines[(mode, nm, "true")] = l_true
                lines[(mode, nm, "pred")] = l_pred
                axs[r, c].grid(True)

        fig.tight_layout()

        # --------メモリ共有変数-------------
        header = [
            "Time",
            "Fx_true", "Fy_true", "Fz_true",

            "rgb_Fx_pred", "rgb_Fy_pred", "rgb_Fz_pred",
            "g_Fx_pred", "g_Fy_pred", "g_Fz_pred",
            "hs_Fx_pred", "hs_Fy_pred", "hs_Fz_pred",

            "rgb_Fx_err", "rgb_Fy_err", "rgb_Fz_err",
            "g_Fx_err", "g_Fy_err", "g_Fz_err",
            "hs_Fx_err", "hs_Fy_err", "hs_Fz_err",
        ]
        self.data_writing.writerow(header)

        star_time = time.perf_counter()

        # --------------------------------------------------------------------------------------------------------------------------------------------
        try:
            while True:
                #センサ読み取り
                Fz = normal_force.value / self.N2gf
                Fr = shear_force1.value - shear_force3.value
                Ff = shear_force2.value - shear_force4.value

                #画像取得・ROI
                ret, base = self.capture.read()
                roi = base[self.roiy:self.roiy + self.h, self.roix:self.roix + self.w]
                ret, base = self.capture.read()
                if not ret:
                    print("capture.read() failed")
                    continue

                roi = base[self.roiy:self.roiy + self.h, self.roix:self.roix + self.w]

                cv2.imshow("ROI", roi)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("q -> break")
                    break

                # --- 3モード入力生成 ---
                g = roi[:, :, 1]
                gau = cv2.GaussianBlur(g, (5, 5), 0)
                clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
                hist = clahe.apply(gau)
                g3 = np.stack([hist, hist, hist], axis=2)  # (155,140,3)
                X_g = g3.astype(np.float32).reshape(1, 155, 140, 3) / 255.0

                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                X_rgb = rgb.astype(np.float32).reshape(1, 155, 140, 3) / 255.0

                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                H = hsv[:, :, 0];
                S = hsv[:, :, 1];
                Z = np.zeros_like(H)
                hs3 = np.stack([H, S, Z], axis=2)
                X_hs = hs3.astype(np.float32).reshape(1, 155, 140, 3) / 255.0

                # --- 3モデル推論 ---
                pred_rgb_list = model_rgb.predict(X_rgb, verbose=0)
                pred_g_list = model_g.predict(X_g, verbose=0)
                pred_hs_list = model_hs.predict(X_hs, verbose=0)

                pred_rgb = np.concatenate([pred_rgb_list[0], pred_rgb_list[1], pred_rgb_list[2]], axis=1)
                pred_g = np.concatenate([pred_g_list[0], pred_g_list[1], pred_g_list[2]], axis=1)
                pred_hs = np.concatenate([pred_hs_list[0], pred_hs_list[1], pred_hs_list[2]], axis=1)

                pred_rgb = self.data_unnormalize(pred_rgb)
                pred_g = self.data_unnormalize(pred_g)
                pred_hs = self.data_unnormalize(pred_hs)

                # ---- スカラー化（CSV用）----
                rgb_Fz = float(pred_rgb[:, 0]);
                rgb_Fx = float(pred_rgb[:, 1]);
                rgb_Fy = float(pred_rgb[:, 2])
                g_Fz = float(pred_g[:, 0]);
                g_Fx = float(pred_g[:, 1]);
                g_Fy = float(pred_g[:, 2])
                hs_Fz = float(pred_hs[:, 0]);
                hs_Fx = float(pred_hs[:, 1]);
                hs_Fy = float(pred_hs[:, 2])

                Fx_true = float(Fr)
                Fy_true = float(Ff)
                Fz_true = float(Fz)

                # ---- 誤差（mode別）----
                rgb_Fx_err = rgb_Fx - Fx_true;
                rgb_Fy_err = rgb_Fy - Fy_true;
                rgb_Fz_err = rgb_Fz - Fz_true
                g_Fx_err = g_Fx - Fx_true;
                g_Fy_err = g_Fy - Fy_true;
                g_Fz_err = g_Fz - Fz_true
                hs_Fx_err = hs_Fx - Fx_true;
                hs_Fy_err = hs_Fy - Fy_true;
                hs_Fz_err = hs_Fz - Fz_true


                # --- 真値 ---
                temp_list_Fz_True.append(Fz_true)
                temp_list_Fx_True.append(Fx_true)
                temp_list_Fy_True.append(Fy_true)

                # --- 予測保存（9本） ---
                pred["rgb"]["Fz"].append(float(pred_rgb[:, 0]));
                pred["rgb"]["Fx"].append(float(pred_rgb[:, 1]));
                pred["rgb"]["Fy"].append(float(pred_rgb[:, 2]))
                pred["g"]["Fz"].append(float(pred_g[:, 0]));
                pred["g"]["Fx"].append(float(pred_g[:, 1]));
                pred["g"]["Fy"].append(float(pred_g[:, 2]))
                pred["hs"]["Fz"].append(float(pred_hs[:, 0]));
                pred["hs"]["Fx"].append(float(pred_hs[:, 1]));
                pred["hs"]["Fy"].append(float(pred_hs[:, 2]))

                end_time = time.perf_counter()
                delta_time = end_time - star_time
                temp_list_time.append(delta_time)

                self.data_writing.writerow([
                    delta_time,
                    Fx_true, Fy_true, Fz_true,

                    rgb_Fx, rgb_Fy, rgb_Fz,
                    g_Fx, g_Fy, g_Fz,
                    hs_Fx, hs_Fy, hs_Fz,

                    rgb_Fx_err, rgb_Fy_err, rgb_Fz_err,
                    g_Fx_err, g_Fy_err, g_Fz_err,
                    hs_Fx_err, hs_Fy_err, hs_Fz_err,
                ])

                if num >= show_num:
                    temp_list_time.pop(0)

                    temp_list_Fz_True.pop(0)
                    temp_list_Fx_True.pop(0)
                    temp_list_Fy_True.pop(0)

                    for m in ["rgb", "g", "hs"]:
                        for a in ["Fz", "Fx", "Fy"]:
                            pred[m][a].pop(0)

                # x軸
                t = temp_list_time

                for r, mode in enumerate(["rgb", "g", "hs"]):
                    lines[(mode, "Fz", "true")].set_data(t, temp_list_Fz_True)
                    lines[(mode, "Fz", "pred")].set_data(t, pred[mode]["Fz"])

                    lines[(mode, "Fx", "true")].set_data(t, temp_list_Fx_True)
                    lines[(mode, "Fx", "pred")].set_data(t, pred[mode]["Fx"])

                    lines[(mode, "Fy", "true")].set_data(t, temp_list_Fy_True)
                    lines[(mode, "Fy", "pred")].set_data(t, pred[mode]["Fy"])

                # 軸調整
                if len(t) >= 2:
                    xmin, xmax = t[0], t[-1]
                    for r, mode in enumerate(["rgb", "g", "hs"]):
                        for c, axis in enumerate(["Fz", "Fx", "Fy"]):
                            a = axs[r, c]
                            a.set_xlim(xmin, xmax)

                            ys = []
                            ys += list(lines[(mode, axis, "true")].get_ydata())
                            ys += list(lines[(mode, axis, "pred")].get_ydata())

                            if ys:
                                y_min, y_max = min(ys), max(ys)
                                if y_min == y_max:
                                    y_min -= 1
                                    y_max += 1
                                pad = 0.1 * (y_max - y_min)
                                a.set_ylim(y_min - pad, y_max + pad)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

                print(temp_list_time)
                num += 1

        finally:
            # 1) カメラ解放
            if hasattr(self, "capture") and self.capture is not None:
                self.capture.release()

            # 2) OpenCVウィンドウ破棄
            cv2.destroyAllWindows()

            # 3) CSVをフラッシュしてクローズ
            try:
                self.data_csv.flush()
                self.data_csv.close()
            except Exception:
                pass

            # 4) Matplotlibも閉じる（任意）
            try:
                plt.ioff()
                plt.close(fig)
            except Exception:
                pass

            print("clean shutdown done.")

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
