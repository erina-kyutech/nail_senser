# -*- coding: utf-8 -*-
"""
axis_satuei_coverage.py

【このファイルについて】
既存の撮影プログラム(axis_satuei_rgb_only_concat 系)をベースに、
8方向固定のせん断力教示を撤廃し、代わりに力空間(Fz, Fr, Ff)の
カバレッジ(どれだけ満遍なくデータが取れているか)をリアルタイムに
可視化する方式へ改修したものです。

変更点の要約は、このファイル末尾の docstring、もしくはチャット側の
説明を参照してください。コード中にも "★変更" というコメントを
付けてあります。
"""

import cv2
import numpy as np
import time
import math
import os
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import csv
from multiprocessing import Process, Value
import serial
import threading


# =========================================================
# ★変更なし: カメラ読み込み方式の切り替えフラグ
#   True  -> 別スレッドで常時カメラを読み続け、最新フレームをバッファから取得(FPSが出やすい)
#   False -> メインループ(updateframe)内で直接 cap.read() する(安定重視・低FPS)
# =========================================================
DIAG_USE_THREAD = True


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
    return img[y1:y2, x1:x2], (cx, cy, w, h)


def _resize_no_pad_center_crop(img, out_w, out_h):
    """黒帯なし・歪みなしで中心クロップして指定サイズへ"""
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


# =========================================================
# ★変更なし: ロードセル関連クラス群(校正式・通信方式そのまま)
# =========================================================
class SC800IM700_1:
    def __init__(self, ser_port, I2C_address):
        self.ser = serial.Serial(ser_port, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.01)
        self.read_address = ((I2C_address << 1) | 0x01).to_bytes(1, 'big')
        self.write_address = ((I2C_address << 1) & 0xfe).to_bytes(1, 'big')
        self.address = I2C_address.to_bytes(1, 'big')
        self.S = b'\x53'
        self.P = b'\x50'
        self.R = b'\x52'
        self.W = b'\x57'
        self.gain = 'x16'
        print('Loadcell UART BRIDGE Connected')

    def readRegister(self, address):
        self.ser.flushInput()
        self.ser.write(self.R + address + self.P)
        value = b''
        while value == b'':
            value = self.ser.read()
        return ord(value)

    def write_byte_data(self, address, reg, value):
        packet = self.S + self.write_address + b'\x02' + reg + value + self.P
        self.ser.write(packet)

    def read_byte(self, reg):
        packet = self.S + self.write_address + b'\x01' + reg + self.S + self.read_address + b'\x01' + self.P
        self.ser.flushInput()
        self.ser.write(packet)
        value = b''
        while value == b'':
            value = self.ser.read()
        return ord(value)

    def read_byte_data(self, reg):
        return self.read_byte(reg)

    def gain_make(self, base_gain):
        if base_gain == 'x16':
            self.write_byte_data(self.address, b'\x01', b'\x24')
            self.offset03, self.offset04, self.offset05 = 0x00, 0x03, 0xE8
            self.gain06, self.gain07, self.gain08, self.gain09 = 0x0F, 0x00, 0x00, 0x00
        else:
            raise Exception('Unknown Gain')

        self.write_byte_data(self.address, b'\x02', b'\x32')
        self.write_byte_data(self.address, b'\x03', self.offset03.to_bytes(1, 'big'))
        self.write_byte_data(self.address, b'\x04', self.offset04.to_bytes(1, 'big'))
        self.write_byte_data(self.address, b'\x05', self.offset05.to_bytes(1, 'big'))
        self.write_byte_data(self.address, b'\x02', b'\x33')
        self.write_byte_data(self.address, b'\x06', self.gain06.to_bytes(1, 'big'))
        self.write_byte_data(self.address, b'\x07', self.gain07.to_bytes(1, 'big'))
        self.write_byte_data(self.address, b'\x08', self.gain08.to_bytes(1, 'big'))
        self.write_byte_data(self.address, b'\x09', self.gain09.to_bytes(1, 'big'))

    def power_on(self):
        self.write_byte_data(self.address, b'\x00', b'\x01')
        print('Register reset')
        self.write_byte_data(self.address, b'\x00', b'\x0E')
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xAE')
        self.write_byte_data(self.address, b'\x15', b'\x30')
        self.gain_make(self.gain)
        self.write_byte_data(self.address, b'\x02', b'\x30')
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xBE')
        print('Loadcell Ready!!')

    def sub_ready(self):
        self.ser.close()

    @classmethod
    def sub_loop(cls, ser_port, I2C_address, ser_flag, shear_force1, count1):
        ser = serial.Serial(ser_port, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.01)
        read_address = ((I2C_address << 1) | 0x01).to_bytes(1, 'big')
        write_address = ((I2C_address << 1) & 0xfe).to_bytes(1, 'big')
        S, P = b'\x53', b'\x50'
        while ser_flag.value:
            packet = S + write_address + b'\x01' + b'\x12' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data1 = b''
            while data1 == b'':
                data1 = ser.read()
            packet = S + write_address + b'\x01' + b'\x13' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data2 = b''
            while data2 == b'':
                data2 = ser.read()
            data_row = int.from_bytes(data1 + data2, 'big', signed=False)
            gf_data = (data_row + 3601) / 1696
            count1.value += 1
            if count1.value == 1:
                gf_data_ini = gf_data
            gf_data = gf_data - gf_data_ini
            shear_force1.value = gf_data * (gf_data > 0)
        ser.close()


class SC800IM700_2(SC800IM700_1):
    def power_on(self):
        self.write_byte_data(self.address, b'\x00', b'\x01')
        print('Register reset')
        self.write_byte_data(self.address, b'\x00', b'\x0E')
        time.sleep(1)
        print(hex(self.read_byte_data(b'\x00')))
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xAE')
        self.write_byte_data(self.address, b'\x15', b'\x30')
        self.gain_make(self.gain)
        self.write_byte_data(self.address, b'\x02', b'\x30')
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xBE')
        print('Loadcell Ready!!')

    @classmethod
    def sub_loop(cls, ser_port, I2C_address, ser_flag, shear_force2, count2):
        ser = serial.Serial(ser_port, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.01)
        read_address = ((I2C_address << 1) | 0x01).to_bytes(1, 'big')
        write_address = ((I2C_address << 1) & 0xfe).to_bytes(1, 'big')
        S, P = b'\x53', b'\x50'
        while ser_flag.value:
            packet = S + write_address + b'\x01' + b'\x12' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data1 = b''
            while data1 == b'':
                data1 = ser.read()
            packet = S + write_address + b'\x01' + b'\x13' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data2 = b''
            while data2 == b'':
                data2 = ser.read()
            data_row = int.from_bytes(data1 + data2, 'big', signed=False)
            gf_data = (data_row + 2623) / 1655
            count2.value += 1
            if count2.value == 1:
                gf_data_ini = gf_data
            gf_data = gf_data - gf_data_ini
            shear_force2.value = gf_data * (gf_data > 0)
        ser.close()


class SC800IM700_3(SC800IM700_1):
    def power_on(self):
        self.write_byte_data(self.address, b'\x00', b'\x01')
        print('Register reset')
        self.write_byte_data(self.address, b'\x00', b'\x0E')
        time.sleep(1)
        print(hex(self.read_byte_data(b'\x00')))
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xAE')
        self.write_byte_data(self.address, b'\x15', b'\x30')
        self.gain_make(self.gain)
        self.write_byte_data(self.address, b'\x02', b'\x30')
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xBE')
        print('Loadcell Ready!!')

    @classmethod
    def sub_loop(cls, ser_port, I2C_address, ser_flag, shear_force3, count3):
        ser = serial.Serial(ser_port, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.01)
        read_address = ((I2C_address << 1) | 0x01).to_bytes(1, 'big')
        write_address = ((I2C_address << 1) & 0xfe).to_bytes(1, 'big')
        S, P = b'\x53', b'\x50'
        while ser_flag.value:
            packet = S + write_address + b'\x01' + b'\x12' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data1 = b''
            while data1 == b'':
                data1 = ser.read()
            packet = S + write_address + b'\x01' + b'\x13' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data2 = b''
            while data2 == b'':
                data2 = ser.read()
            data_row = int.from_bytes(data1 + data2, 'big', signed=False)
            gf_data = (data_row * 10 - 938) / 17420
            count3.value += 1
            if count3.value == 1:
                gf_data_ini = gf_data
            gf_data = gf_data - gf_data_ini
            shear_force3.value = gf_data * (gf_data > 0)
        ser.close()


class SC800IM700_4(SC800IM700_1):
    def power_on(self):
        self.write_byte_data(self.address, b'\x00', b'\x01')
        print('Register reset')
        self.write_byte_data(self.address, b'\x00', b'\x0E')
        time.sleep(1)
        print(hex(self.read_byte_data(b'\x00')))
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xAE')
        self.write_byte_data(self.address, b'\x15', b'\x30')
        self.gain_make(self.gain)
        self.write_byte_data(self.address, b'\x02', b'\x30')
        time.sleep(1)
        self.write_byte_data(self.address, b'\x00', b'\xBE')
        print('Loadcell Ready!!')

    @classmethod
    def sub_loop(cls, ser_port, I2C_address, ser_flag, shear_force4, count4):
        ser = serial.Serial(ser_port, baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=0.01)
        read_address = ((I2C_address << 1) | 0x01).to_bytes(1, 'big')
        write_address = ((I2C_address << 1) & 0xfe).to_bytes(1, 'big')
        S, P = b'\x53', b'\x50'
        while ser_flag.value:
            packet = S + write_address + b'\x01' + b'\x12' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data1 = b''
            while data1 == b'':
                data1 = ser.read()
            packet = S + write_address + b'\x01' + b'\x13' + S + read_address + b'\x01' + P
            ser.flushInput(); ser.write(packet)
            data2 = b''
            while data2 == b'':
                data2 = ser.read()
            data_row = int.from_bytes(data1 + data2, 'big', signed=False)
            gf_data = (data_row + 4184) / 1145
            count4.value += 1
            if count4.value == 1:
                gf_data_ini = gf_data
            gf_data = gf_data - gf_data_ini
            shear_force4.value = gf_data * (gf_data > 0)
        ser.close()


class gf2000:
    def __init__(self, ser_port):
        self.ser = serial.Serial(ser_port, baudrate=9600, bytesize=serial.SEVENBITS, parity=serial.PARITY_EVEN)
        print('Normal force loadcell Ready!!')

    def sub_ready(self):
        self.ser.close()

    @classmethod
    def sub_loop(cls, ser_port, ser_flag, normal_force):
        ser = serial.Serial(ser_port, baudrate=9600, bytesize=serial.SEVENBITS, parity=serial.PARITY_EVEN)
        while ser_flag.value:
            line = ser.readline()
            line = line.decode('utf-8')
            line = line[-9:-2]
            normal_force.value = float(line)
        ser.close()


# =========================================================
# ★変更: 素材選択 (1-4の番号 or 素材名で入力可能)
# =========================================================
def select_material():
    """
    素材を選ばせて force_path を決める。
    番号(1-4) でも素材名(felt/acrylic/paper/aluminum)でも入力可能。
    不正な入力の場合は再入力を促す。
    """
    name_by_key = {
        "1": "felt", "felt": "felt",
        "2": "acrylic", "acrylic": "acrylic",
        "3": "paper", "paper": "paper",
        "4": "aluminum", "aluminum": "aluminum",
    }
    path_by_name = {
        "felt": "felt_0-10xyz",
        "acrylic": "acrylic_0-10xyz",
        "paper": "paper_0-10xyz",
        "aluminum": "aluminum_0-10xyz",
    }

    print("=== 素材を選択してください ===")
    print("  1: felt")
    print("  2: acrylic")
    print("  3: paper")
    print("  4: aluminum")

    while True:
        raw = input("番号または素材名を入力してください: ").strip().lower()
        if raw in name_by_key:
            material_name = name_by_key[raw]
            force_path = path_by_name[material_name]
            print(f"-> 選択された素材: {material_name} (force_path={force_path})")
            return force_path
        print("入力が正しくありません。1〜4の番号か felt/acrylic/paper/aluminum を入力してください。")


class GraphMake:
    def __init__(self):
        nowdir = os.path.dirname(__file__)
        print('now_directry:', nowdir)
        os.chdir(nowdir)
        os.chdir('..')

        # ★変更: 実行時に素材を選ばせる(以前は force_path をコメントアウトで固定していた)
        self.force_path = select_material()
        self.degree_str = 360

        print('名前と測定回数を英字半角で入力してください(例:felt1)')
        self.dirname = input().strip()

        self.save_dir = './datas/' + self.force_path + '/' + self.dirname + '/360deg'
        os.makedirs(self.save_dir, exist_ok=False)
        self.datalog_path = self.save_dir + '/datalog.csv'
        self.namelist_path = './datas/' + self.force_path + '/namelist.csv'

        self.name_csv = open(self.namelist_path, 'a', newline='')
        self.name_writing = csv.writer(self.name_csv)
        record_path = self.force_path + '/' + self.dirname + '/360deg/datalog.csv'
        self.name_writing.writerow([record_path, '360'])
        self.name_csv.close()

        # ★変更なし: CSVの列名は既存学習プログラムとの互換性のため維持
        self.data_csv = open(self.datalog_path, 'w', newline='')
        self.data_writing = csv.writer(self.data_csv)
        self.data_writing.writerow(['path', 'Fz', 'Fr', 'Ff'])

        # ★削除: mu, fz, T_num, Fr, rectime, recstart, recfinish, graphtime, frontT, Frange
        #   -> 8方向固定教示の軌道生成に使っていたパラメータなので不要になった。
        self.N2gf = 101.972
        self.datanum = 0

        self.CAM_NAIL = 1
        self.CAM_TIP = 0
        self.OUT_H = 150
        self.OUT_W_LEFT = 150
        self.OUT_W_RIGHT = 140

        self.n_cx, self.n_cy = 499, 250
        self.n_w, self.n_h = int(282 * 1.3), int(409 * 0.9)
        self.t_cx, self.t_cy = 324, 550
        self.t_w, self.t_h = int(182 * 1.7), int(136 * 1.0)

        self.cap_nail = cv2.VideoCapture(self.CAM_NAIL, cv2.CAP_MSMF)
        time.sleep(0.8)
        self.cap_tip = cv2.VideoCapture(self.CAM_TIP, cv2.CAP_MSMF)
        for cap in (self.cap_nail, self.cap_tip):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        if (not self.cap_nail.isOpened()) or (not self.cap_tip.isOpened()):
            raise RuntimeError(
                f'IO Error (2 cameras). opened_nail={self.cap_nail.isOpened()} opened_tip={self.cap_tip.isOpened()}'
            )

        # ★変更なし(ただし DIAG_USE_THREAD で有効/無効を切替できるようにした)
        self._frame_nail = None
        self._frame_tip = None
        self._lock_nail = threading.Lock()
        self._lock_tip = threading.Lock()
        self._cam_running = True
        self.use_thread = DIAG_USE_THREAD

        if self.use_thread:
            self._thread_nail = threading.Thread(target=self._read_nail, daemon=True)
            self._thread_tip = threading.Thread(target=self._read_tip, daemon=True)
            self._thread_nail.start()
            self._thread_tip.start()
            time.sleep(0.5)  # 最初のフレームが入るまで待つ
        else:
            print('[INFO] DIAG_USE_THREAD=False: メインループで直接 cap.read() します')

        self.w = self.OUT_W_LEFT + self.OUT_W_RIGHT
        self.h = self.OUT_H

        # =========================================================
        # ★変更: 力空間カバレッジの設定
        #   ビン数や範囲を変えたい場合はここを編集する
        # =========================================================
        self.FZ_MIN, self.FZ_MAX, self.N_BIN_FZ = 0.0, 10.0, 5
        self.FR_MIN, self.FR_MAX, self.N_BIN_FR = -5.0, 5.0, 5
        self.FF_MIN, self.FF_MAX, self.N_BIN_FF = -5.0, 5.0, 5
        self.TOTAL_BINS = self.N_BIN_FZ * self.N_BIN_FR * self.N_BIN_FF
        # visited[iz, ir, iff] : そのビンにデータが入ったかどうか(bool)
        self.visited = np.zeros((self.N_BIN_FZ, self.N_BIN_FR, self.N_BIN_FF), dtype=bool)
        self.coverage_pct = 0.0

        # せん断力の履歴(表示用、薄く残す)
        self.hist_Fr = []
        self.hist_Ff = []
        self.MAX_HIST_POINTS = 20000  # 表示が重くなりすぎないよう上限を設ける

        # =========================================================
        # ★変更: 描画レイアウト
        #   axL   : カメラ結合画像
        #   axFz  : 現在のFz(バー表示)
        #   axShear: 現在のせん断力(Fr,Ff)平面 + 履歴
        #   axCov : カバレッジヒートマップ(Fr-Ff平面、色=踏んだFz段階数) + coverage[%]表示
        # =========================================================
        self.fig = plt.figure(figsize=(20, 6))
        self.fig.subplots_adjust(wspace=0.6)
        manager = plt.get_current_fig_manager()
        try:
            manager.window.wm_geometry('+20+50')  # Tk用
        except AttributeError:
            try:
                manager.windoe.setGeometry(20, 50, 1800, 600)  # Qt用
            except AttributeError:
                pass

        gs = GridSpec(1, 8)
        label_fontsize = 14
        title_fontsize = 15
        tick_fontsize = 11

        self.axL = self.fig.add_subplot(gs[:, 0:3])
        self.axFz = self.fig.add_subplot(gs[:, 3:4])
        self.axShear = self.fig.add_subplot(gs[:, 4:6])
        self.axCov = self.fig.add_subplot(gs[:, 6:8])

        # --- 画像パネル ---
        self.axL.set_title('Captured RGB image', fontsize=title_fontsize)
        self.axL.set_xlabel('image width [pixel]', fontsize=label_fontsize)
        self.axL.set_ylabel('image height [pixel]', fontsize=label_fontsize)
        self.image_init0 = np.zeros((self.h, self.w, 3), dtype='uint8')
        self.image_plt = self.axL.imshow(self.image_init0, animated=True)

        # --- Fzバー表示パネル(★変更: 従来の"教示線"表示をやめ、現在値のバーにした) ---
        self.axFz.set_title('Current Fz', fontsize=title_fontsize)
        self.axFz.set_ylabel('Fz [N]', fontsize=label_fontsize)
        self.axFz.set_xlim(-1, 1)
        self.axFz.set_ylim(0, self.FZ_MAX * 1.1)
        self.axFz.set_xticks([])
        self.axFz.tick_params(axis='both', labelsize=tick_fontsize)
        self.fz_bar = self.axFz.bar([0], [0], width=0.6, color='tab:blue')[0]

        # --- せん断力(Fr,Ff)パネル(★変更: 教示軌道は廃止し、現在値+履歴のみ) ---
        self.axShear.set_title('Current shear force (Fr, Ff)', fontsize=title_fontsize)
        self.axShear.set_xlabel('Fr [N]', fontsize=label_fontsize)
        self.axShear.set_ylabel('Ff [N]', fontsize=label_fontsize)
        self.axShear.set_xlim(self.FR_MIN - 1, self.FR_MAX + 1)
        self.axShear.set_ylim(self.FF_MIN - 1, self.FF_MAX + 1)
        self.axShear.tick_params(axis='both', labelsize=tick_fontsize)
        self.shear_hist_line, = self.axShear.plot([], [], '.', color='gray', alpha=0.25, markersize=3,
                                                    label='history')
        self.shear_now_line, = self.axShear.plot([], [], 'o', color='red', markersize=9, label='now')
        self.axShear.legend(loc='upper right', fontsize=9)

        # --- カバレッジヒートマップパネル(★新規追加) ---
        self.axCov.set_title('Coverage map (Fz bins reached)', fontsize=title_fontsize)
        self.axCov.set_xlabel('Fr [N]', fontsize=label_fontsize)
        self.axCov.set_ylabel('Ff [N]', fontsize=label_fontsize)
        init_grid = np.zeros((self.N_BIN_FF, self.N_BIN_FR))
        self.cov_im = self.axCov.imshow(
            init_grid, origin='lower',
            extent=[self.FR_MIN, self.FR_MAX, self.FF_MIN, self.FF_MAX],
            aspect='auto', cmap='viridis', vmin=0, vmax=self.N_BIN_FZ
        )
        self.cov_cbar = self.fig.colorbar(self.cov_im, ax=self.axCov, fraction=0.046, pad=0.04)
        self.cov_cbar.set_label('Fz bins visited', fontsize=10)
        self.cov_text = self.axCov.text(
            0.02, 0.96, 'coverage: 0.0%', transform=self.axCov.transAxes,
            color='white', fontsize=13, va='top',
            bbox=dict(facecolor='black', alpha=0.5, pad=3)
        )

        # ★変更: blitはOFFにした(新しいパネル・テキスト・カラーバーが増え、
        #   blit=Trueのままだと更新漏れが起きやすいため。画像更新の負荷は軽いので問題なし)
        self.use_blit = False

        # FPS計測用
        self._fps_prev_time = time.perf_counter()
        self._fps = 0.0

        # 撮影中フラグ管理用
        self.recording_started = False

    # ---------------------------------------------------------
    # ★変更なし: カメラ読み込みスレッド(DIAG_USE_THREAD=Trueのときのみ使用)
    # ---------------------------------------------------------
    def _read_nail(self):
        while self._cam_running:
            ret, frame = self.cap_nail.read()
            if ret and frame is not None:
                with self._lock_nail:
                    self._frame_nail = frame

    def _read_tip(self):
        while self._cam_running:
            ret, frame = self.cap_tip.read()
            if ret and frame is not None:
                with self._lock_tip:
                    self._frame_tip = frame

    # ---------------------------------------------------------
    # ★新規: カバレッジ計算まわり(軽量な実装)
    # ---------------------------------------------------------
    @staticmethod
    def _bin_index(val, vmin, vmax, nbins):
        """値を [0, nbins-1] のビン番号に変換(範囲外はクリップ)"""
        if nbins <= 1:
            return 0
        if val <= vmin:
            return 0
        if val >= vmax:
            return nbins - 1
        idx = int((val - vmin) / (vmax - vmin) * nbins)
        return min(max(idx, 0), nbins - 1)

    def update_coverage(self, Fz, Fr, Ff):
        """
        現在の(Fz, Fr, Ff)がどのビンに入るか計算し、visited を更新する。
        戻り値はカバレッジ[%]。
        計算量はO(1)(ビン番号を計算してbool配列に立てるだけ)なので軽量。
        """
        iz = self._bin_index(Fz, self.FZ_MIN, self.FZ_MAX, self.N_BIN_FZ)
        ir = self._bin_index(Fr, self.FR_MIN, self.FR_MAX, self.N_BIN_FR)
        iff = self._bin_index(Ff, self.FF_MIN, self.FF_MAX, self.N_BIN_FF)
        self.visited[iz, ir, iff] = True
        self.coverage_pct = float(self.visited.sum()) / self.TOTAL_BINS * 100.0
        return self.coverage_pct

    def coverage_grid(self):
        """
        Fr-Ff平面のヒートマップ用グリッドを作る。
        各マス = そのFr,Ff領域で何段階のFzビンを踏んだか(0〜N_BIN_FZ)。
        visited.shape = (N_BIN_FZ, N_BIN_FR, N_BIN_FF)
        -> Fz軸方向にsumして (N_BIN_FR, N_BIN_FF) にし、
           imshow用に (N_BIN_FF, N_BIN_FR) へ転置する。
        """
        counts_fr_ff = self.visited.sum(axis=0)  # shape: (N_BIN_FR, N_BIN_FF)
        return counts_fr_ff.T  # shape: (N_BIN_FF, N_BIN_FR)  imshowの行=Ff, 列=Fr

    # ---------------------------------------------------------
    def onkey(self, event):
        if event.key == 'escape':
            print('esc')
            ser_flag.value = False
            self._cam_running = False
            try:
                self.data_csv.close()
            except Exception:
                pass
            try:
                self.cap_nail.release()
            except Exception:
                pass
            try:
                self.cap_tip.release()
            except Exception:
                pass
            try:
                self.ani.event_source.stop()
            except Exception:
                pass
            plt.close(event.canvas.figure)

        if event.key == 'r' and not self.recording_started:
            print('r : 撮影開始')
            rec_flag.value = True
            self.recording_started = True
            self.graphstart = time.perf_counter()

    def updateframe(self, dum):
        # FPS計算
        now = time.perf_counter()
        dt = now - self._fps_prev_time
        self._fps_prev_time = now
        if dt > 0:
            self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1
        print(f'\rFPS: {self._fps:.1f}  coverage: {self.coverage_pct:.1f}%', end='')

        # ---- カメラ画像取得(スレッド方式 / 直接read方式を切替) ----
        if self.use_thread:
            with self._lock_nail:
                base_n = self._frame_nail
            with self._lock_tip:
                base_t = self._frame_tip
        else:
            ret1, base_n = self.cap_nail.read()
            ret2, base_t = self.cap_tip.read()
            if not ret1:
                base_n = None
            if not ret2:
                base_t = None

        if not hasattr(self, '_printed_shape'):
            self._printed_shape = True
            print('capture nail shape:', None if base_n is None else base_n.shape)
            print('capture tip  shape:', None if base_t is None else base_t.shape)

        artists = [self.image_plt, self.fz_bar, self.shear_hist_line, self.shear_now_line,
                   self.cov_im, self.cov_text]

        if (base_n is None) or (base_t is None):
            return artists

        roi_n, _ = crop_with_center_wh_safe(base_n, self.n_cx, self.n_cy, self.n_w, self.n_h)
        roi_t, _ = crop_with_center_wh_safe(base_t, self.t_cx, self.t_cy, self.t_w, self.t_h)
        nail_in = _resize_no_pad_center_crop(roi_n, self.OUT_W_LEFT, self.OUT_H)
        tip_in = _resize_no_pad_center_crop(roi_t, self.OUT_W_RIGHT, self.OUT_H)
        net_in = np.concatenate([nail_in, tip_in], axis=1)  # BGRのまま保存(既存互換)

        net_rgb = cv2.cvtColor(net_in, cv2.COLOR_BGR2RGB)
        self.image_plt.set_array(net_rgb)

        # ---- 現在の力を取得(★変更なし: 既存の計算式のまま) ----
        Fz = normal_force.value / self.N2gf
        Fr = shear_force1.value - shear_force3.value
        Ff = shear_force2.value - shear_force4.value

        # ---- 表示は撮影開始前後にかかわらず常に現在値を反映 ----
        self.fz_bar.set_height(max(0.0, float(Fz)))

        self.hist_Fr.append(float(Fr))
        self.hist_Ff.append(float(Ff))
        if len(self.hist_Fr) > self.MAX_HIST_POINTS:
            self.hist_Fr = self.hist_Fr[-self.MAX_HIST_POINTS:]
            self.hist_Ff = self.hist_Ff[-self.MAX_HIST_POINTS:]
        self.shear_hist_line.set_data(self.hist_Fr, self.hist_Ff)
        self.shear_now_line.set_data([Fr], [Ff])

        if rec_flag.value:
            # ---- 撮影中: 画像保存・CSV保存・カバレッジ更新 ----
            self.update_coverage(Fz, Fr, Ff)
            self.cov_im.set_data(self.coverage_grid())
            self.cov_text.set_text(f'coverage: {self.coverage_pct:.1f}%')

            t = time.perf_counter() - self.graphstart
            framename = self.save_dir + '/' + str(self.datanum) + '.png'
            cv2.imwrite(framename, net_in)
            self.data_writing.writerow([framename, Fz, Fr, Ff])
            self.datanum += 1

            if self.datanum % 50 == 0:
                print(f'\n[recording] t={t:.1f}s  n={self.datanum}  '
                      f'Fz={Fz:.2f} Fr={Fr:.2f} Ff={Ff:.2f}  coverage={self.coverage_pct:.1f}%')
        else:
            # ---- 待機中: coverageテキストだけ現状値を保持して表示 ----
            self.cov_text.set_text(f'coverage: {self.coverage_pct:.1f}% (waiting: press "r")')

        return artists

    def animation(self):
        self.ani = animation.FuncAnimation(self.fig, self.updateframe, interval=0, blit=self.use_blit)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    normal_force = Value('f', 0.00)
    shear_force1 = Value('f', 0.00)
    shear_force2 = Value('f', 0.00)
    shear_force3 = Value('f', 0.00)
    shear_force4 = Value('f', 0.00)
    ser_flag = Value('b', True)
    rec_flag = Value('b', False)

    xy_port_1 = 'COM8';  xy_address_1 = 0x2A
    xy_port_2 = 'COM9';  xy_address_2 = 0x2A
    xy_port_3 = 'COM10'; xy_address_3 = 0x2A
    xy_port_4 = 'COM12'; xy_address_4 = 0x2A
    z_port = 'COM21'

    shear_loadcell_1 = SC800IM700_1(xy_port_1, xy_address_1); shear_loadcell_1.power_on(); shear_loadcell_1.sub_ready()
    shear_loadcell_2 = SC800IM700_2(xy_port_2, xy_address_2); shear_loadcell_2.power_on(); shear_loadcell_2.sub_ready()
    shear_loadcell_3 = SC800IM700_3(xy_port_3, xy_address_3); shear_loadcell_3.power_on(); shear_loadcell_3.sub_ready()
    shear_loadcell_4 = SC800IM700_4(xy_port_4, xy_address_4); shear_loadcell_4.power_on(); shear_loadcell_4.sub_ready()
    normal_loadcell = gf2000(z_port); normal_loadcell.sub_ready()

    sub_z = Process(target=gf2000.sub_loop, args=[z_port, ser_flag, normal_force])
    sub_z.start()

    count1 = Value('i', 0); count2 = Value('i', 0); count3 = Value('i', 0); count4 = Value('i', 0)
    sub_xy1 = Process(target=SC800IM700_1.sub_loop, args=[xy_port_1, xy_address_1, ser_flag, shear_force1, count1])
    sub_xy2 = Process(target=SC800IM700_2.sub_loop, args=[xy_port_2, xy_address_2, ser_flag, shear_force2, count2])
    sub_xy3 = Process(target=SC800IM700_3.sub_loop, args=[xy_port_3, xy_address_3, ser_flag, shear_force3, count3])
    sub_xy4 = Process(target=SC800IM700_4.sub_loop, args=[xy_port_4, xy_address_4, ser_flag, shear_force4, count4])
    sub_xy1.start(); sub_xy2.start(); sub_xy3.start(); sub_xy4.start()

    try:
        graph = GraphMake()
        graph.animation()
    finally:
        # ★念のため: 異常終了時もセンサプロセスを止める
        ser_flag.value = False
        for p in (sub_z, sub_xy1, sub_xy2, sub_xy3, sub_xy4):
            try:
                p.join(timeout=2.0)
            except Exception:
                pass