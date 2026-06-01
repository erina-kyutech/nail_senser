import datetime

from matplotlib import animation

import validation_VGG16
import axis_satsuei
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

gf2000 = axis_satsuei.gf2000
SC800IM700 = axis_satsuei.SC800IM700

class RealTime:

    def __init__(self):
        self.gf2000 = axis_satsuei.gf2000
        self.SC800IM700 = axis_satsuei.SC800IM700
        self.CNN = validation_VGG16.multitask_CNN()


        self.roix, self.roiy = 260, 165  # 左上座標
        self.w, self.h = 140, 155  # 幅,高さ

        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened() is False:
            raise ("IO Error")

        self.normal_force_normalize = 10.0
        self.thear_force_normalize = 5.0

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

        temp_list_Fz_Predict = []  # 临时存放y轴数据
        temp_list_Fx_Predict = []  # 临时存放y轴数据
        temp_list_Fy_Predict = []  # 临时存放y轴数据


        temp_list_Fz_True = []  # 临时存放y轴数据
        temp_list_Fx_True = []  # 临时存放y轴数据
        temp_list_Fy_True = []  # 临时存放y轴数据

        show_num = 100  # x轴显示的数据个数，例：show_num = 10表示x轴只显示10个数据
        num = 0

        plt.ion()  # 打开交互模式
        #fig1 = plt.figure(figsize=(100, 80))  # 设置图片大小
        #plt.xlim(0, show_num)  # 设置x轴的数值显示范围
        #plt.ylim(-5, 5)
        #x_major_locator = MultipleLocator(100000)  # 把x轴的刻度间隔设置为2
        #y_major_locator = MultipleLocator(10)  # 把y轴的刻度间隔设置为10
        #ax = plt.gca()  # ax为两条坐标轴的实例
        #ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为2的倍数
        #ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数

        plt.subplot(311, title="Fz", xlabel="t [s]", ylabel="Fz [N]")
        plt.subplot(312, title="Fx", xlabel="t [s]", ylabel="Fx [N]")
        plt.subplot(313, title="Fy", xlabel="t [s]", ylabel="Fy [N]")

        # --------メモリ共有変数-------------
        header = ['Time', 'Fx_True','Fy_True', 'Fz_True', 'Fx_Predict', 'Fy_Predict', 'Fz_Predict','Fx_Error','Fy_Error','Fz_Error']
        self.data_writing.writerow(header)

        star_time = time.perf_counter()

        # --------------------------------------------------------------------------------------------------------------------------------------------

        while True:
            # 荷重計の取った値[g]を[gf]として[N]に変換(ここ並列処理にすると40msec高速化)
            Fz = normal_force.value / self.N2gf
            Fr = shear_force.value / self.N2gf

            ret, base = self.capture.read()
            roi = base[self.roiy:self.roiy + self.h, self.roix:self.roix + self.w]
            img_blue_c1, img_green_c1, img_red_c1 = cv2.split(roi)
            gray = img_green_c1

            gau = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            hist = clahe.apply(gau)

            X_array = np.array(hist)
            X_array = X_array.reshape(-1, 155, 140, 1)

            X = X_array.astype("float64")
            X = X.reshape((-1, 155, 140, 1))

            X/= 255.0


            predict = self.CNN.model.predict(X)

            predict = np.concatenate([predict[0],
                                      predict[1],
                                      predict[2]],
                                      axis = 1)

            predict = self.data_unnormalize(predict)
            print(predict)

            #print("Fz=",Fz)
            #print("Fr=",Fr)
            # -----------------------------------
            #Fz_True_list.append(Fz)
            #temp_list_Fz_True.append(Fz)


            # -----------------------------------
            Fz_pred = predict[:,0]
            Fx_pred = predict[:,1]
            Fy_pred = predict[:,2]
            # -----------------------------------
            Fz_true = Fz
            Fx_true = Fr * np.cos(self.theta)
            Fy_true = Fr * np.sin(self.theta)
            # -----------------------------------
            Fz_error = Fz_pred - Fz_true
            Fx_error = Fx_pred - Fx_true
            Fy_error = Fy_pred - Fy_true
            # -----------------------------------
            Fz_pred = str("".join([str(x)for x in Fz_pred]))
            Fz_error = str("".join([str(x)for x in Fz_error]))

            Fx_pred = str("".join([str(x)for x in Fx_pred]))
            Fx_error = str("".join([str(x)for x in Fx_error]))

            Fy_pred = str("".join([str(x)for x in Fy_pred]))
            Fy_error = str("".join([str(x)for x in Fy_error]))
            # -----------------------------------
            Fz_Predict_list.append(predict[:, 0])
            Fx_Predict_list.append(predict[:, 1])
            Fy_Predict_list.append(predict[:, 2])

            temp_list_Fz_Predict.append(predict[:, 0])
            temp_list_Fx_Predict.append(predict[:, 1])
            temp_list_Fy_Predict.append(predict[:, 2])

            temp_list_Fz_True.append(Fz_true)
            temp_list_Fx_True.append(Fx_true)
            temp_list_Fy_True.append(Fy_true)

            end_time = time.perf_counter()
            delta_time = end_time - star_time
            #delta_time = time.strftime("%M:%S:%MS", time.localtime(time.time()))
            time_list.append(delta_time)
            temp_list_time.append(delta_time)

            self.data_writing.writerow([delta_time, Fx_true, Fy_true, Fz_true, Fx_pred, Fy_pred, Fz_pred, Fx_error, Fy_error, Fz_error])


            if num > show_num:
                temp_list_time.remove(temp_list_time[0])

                temp_list_Fz_True.remove(temp_list_Fz_True[0])
                temp_list_Fz_Predict.remove(temp_list_Fz_Predict[0])

                temp_list_Fx_True.remove(temp_list_Fx_True[0])
                temp_list_Fx_Predict.remove(temp_list_Fx_Predict[0])

                temp_list_Fy_True.remove(temp_list_Fy_True[0])
                temp_list_Fy_Predict.remove(temp_list_Fy_Predict[0])

                # plt.clf()
                plt.cla()
                # plt.close()

            #plt.xticks([])
            #integer = int(temp_list_time)
            integer = math.floor(delta_time)

            plt.subplot(311, title="Fz", xlabel="t [s]", ylabel="Fz [N]")
            plt.plot(temp_list_time, temp_list_Fz_True, c='b', ls=':')
            plt.plot(temp_list_time, temp_list_Fz_Predict, c='r', ls=':')
            if integer > 10:
                plt.xlim(integer-10, integer+10)
            else:
                plt.xlim(0, 20)

            plt.subplot(312, title="Fx", xlabel="t [s]", ylabel="Fx [N]")
            plt.plot(temp_list_time, temp_list_Fx_True, c='b', ls=':')
            plt.plot(temp_list_time, temp_list_Fx_Predict, c='g', ls=':')
            if integer > 10:
                plt.xlim(integer-10, integer+10)
            else:
                plt.xlim(0, 20)

            plt.subplot(313, title="Fy", xlabel="t [s]", ylabel="Fy [N]")
            plt.plot(temp_list_time, temp_list_Fy_True, c='b', ls=':')
            plt.plot(temp_list_time, temp_list_Fy_Predict, c='g', ls=':')
            if integer > 10:
                plt.xlim(integer-10, integer+10)
            else:
                plt.xlim(0, 20)

            plt.tight_layout()#文字の重なり解消
            plt.pause(0.1)
            print(temp_list_time)
            num += 1

            # plt.ioff()  # 关闭交互模式
            #plt.show()

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
    shear_force = Value('f', 0.00)
    ser_flag = Value('b', True)  # シリアル通信フラグ(Trueで荷重計，MD共にループ開始)
    rec_flag = Value('b', False)  # 測定フラグ(これがTrueの間測定)

    # -----------------------------------

    # ロードセル測定準備
    xy_port = "COM6"
    xy_address = 0x2A
    shear_loadcell = SC800IM700(xy_port, xy_address)
    shear_loadcell.connect_check()
    shear_loadcell.power_on()
    shear_loadcell.sub_ready()

    # 荷重計測定準備
    z_port = "COM5"
    normal_loadcell = gf2000(z_port)
    normal_loadcell.sub_ready()

    """
    サブプロセス開始(各種通信)
    並列処理したい関数がクラス内の関数(メソッド)の場合エラーが起きる
    その場合，メソッドをクラスメソッドとして定義してやると動かせる
    """
    sub_z = Process(target=gf2000.sub_loop, args=[z_port, ser_flag, normal_force])
    sub_z.start()
    sub_xy = Process(target=SC800IM700.sub_loop, args=[xy_port, xy_address, ser_flag, shear_force])
    sub_xy.start()

    RealTime().graph()

# -----------------------------------
