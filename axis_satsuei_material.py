# -*- coding: utf-8 -*-
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
import threading  # ★ 追加


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


class GraphMake:
    # ====== 素材選択の定義（追加） ======
    MATERIAL_OPTIONS = {
        '1': ('felt', 'felt_0-10xyz'),
        '2': ('acrylic', 'acrylic_0-10xyz'),
        '3': ('paper', 'paper_0-10xyz'),
        '4': ('aluminum', 'aluminum_0-10xyz'),
    }
    MATERIAL_NAME_ALIASES = {
        'felt': 'felt', 'フェルト': 'felt',
        'acrylic': 'acrylic', 'アクリル': 'acrylic',
        'paper': 'paper', '紙': 'paper', 'かみ': 'paper',
        'aluminum': 'aluminum', 'aluminium': 'aluminum', 'アルミ': 'aluminum', 'アルミニウム': 'aluminum',
    }

    def _select_material(self):
        """
        実行時に素材を選択させる。
        番号(1-4)でも素材名でも入力できて、
        不正な入力の場合は再入力を促す。
        """
        prompt = (
            '素材を選んでください:\n'
            '  1: felt (フェルト)\n'
            '  2: acrylic (アクリル)\n'
            '  3: paper (紙)\n'
            '  4: aluminum (アルミ)\n'
            '番号 または 素材名で入力 > '
        )
        while True:
            raw = input(prompt).strip()

            if raw in self.MATERIAL_OPTIONS:
                name, force_path = self.MATERIAL_OPTIONS[raw]
                print(f'[INFO] 選択された素材: {name} -> {force_path}')
                return force_path

            key = raw.lower()
            if key in self.MATERIAL_NAME_ALIASES:
                canonical = self.MATERIAL_NAME_ALIASES[key]
                for _, (name, force_path) in self.MATERIAL_OPTIONS.items():
                    if name == canonical:
                        print(f'[INFO] 選択された素材: {name} -> {force_path}')
                        return force_path

            print(f'[WARN] "{raw}" は認識できませんでした。1-4の番号か素材名で入力し直してください。\n')
    # ====== ここまで追加 ======

    def __init__(self):
        nowdir = os.path.dirname(__file__)
        print('now_directry:', nowdir)
        os.chdir(nowdir)
        os.chdir('..')

        self.Fz = 10

        # ★素材はコメントアウトではなく実行時選択に変更（変更）
        self.force_path = self._select_material()

        self.degree_str = 360

        print('名前と測定回数を英字半角で入力してください(例:felt1)')
        self.dirname = input().strip()

        self.save_dir = './datas/' + self.force_path + '/' + self.dirname + '/360deg'
        os.makedirs(self.save_dir, exist_ok=False)
        self.datalog_path = self.save_dir + '/datalog.csv'
        self.namelist_path = './datas/' + self.force_path + '/namelist.csv'

        # ====== 回転オフセットの管理（追加） ======
        # 元の12方向パターンは30度おきなので、施行を重ねるたびに
        # ROTATION_STEP_DEG ずつ時計回りにパターン全体をずらすことで、
        # 複数回の撮影を合計した時に30度の隙間が埋まっていくようにする。
        # 素材フォルダ単位で「これまで何回撮ったか」をテキストファイルに記録し、
        # 起動ごとに自動で読み込み→次回用に+1して書き戻す。
        self.ROTATION_STEP_DEG = 5.0  # 6回で5度刻みに密になる
        self.rotation_state_path = './datas/' + self.force_path + '/rotation_state.txt'

        run_count = 0
        if os.path.exists(self.rotation_state_path):
            try:
                with open(self.rotation_state_path, 'r', encoding='utf-8') as f:
                    run_count = int(f.read().strip())
            except (ValueError, OSError):
                run_count = 0

        self.direction_offset_deg = (run_count * self.ROTATION_STEP_DEG) % 30.0
        self._rotation_rad = math.radians(self.direction_offset_deg)

        with open(self.rotation_state_path, 'w', encoding='utf-8') as f:
            f.write(str(run_count + 1))

        print(f'[INFO] この施行の回転オフセット: {self.direction_offset_deg:.1f}度'
              f'（この素材でこれまでの撮影回数: {run_count}）')
        # ====== ここまで追加 ======

        self.name_csv = open(self.namelist_path, 'a', newline='')
        self.name_writing = csv.writer(self.name_csv)
        record_path = self.force_path + '/' + self.dirname + '/360deg/datalog.csv'
        self.name_writing.writerow([record_path, '360'])
        self.name_csv.close()

        self.data_csv = open(self.datalog_path, 'w', newline='')
        self.data_writing = csv.writer(self.data_csv)
        self.data_writing.writerow(['path', 'Fz', 'Fr', 'Ff'])

        if self.Fz == 5:
            self.mu, self.fz, self.T_num = 0.9, 0.03, 6
        else:
            self.mu, self.fz, self.T_num = 0.5, 0.05 * 4, 12

        self.Fr = self.Fz * self.mu
        self.now_Fz = 0.0
        self.t_rest = 5.0
        self.T = 1 / self.fz
        self.rectime = self.T * self.T_num
        self.recstart = self.t_rest
        self.recfinish = self.t_rest + self.rectime
        self.graphtime = self.rectime + 2 * self.t_rest
        self.frontT = 3.0
        self.Frange = 0.2
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
            raise RuntimeError(f'IO Error (2 cameras). opened_nail={self.cap_nail.isOpened()} opened_tip={self.cap_tip.isOpened()}')

        # ★ バッファ初期化
        self._frame_nail = None
        self._frame_tip  = None
        self._lock_nail  = threading.Lock()
        self._lock_tip   = threading.Lock()
        self._cam_running = True

        # ★ カメラ読み込みスレッド起動（2台同時に読み続ける）
        self._thread_nail = threading.Thread(target=self._read_nail, daemon=True)
        self._thread_tip  = threading.Thread(target=self._read_tip,  daemon=True)
        self._thread_nail.start()
        self._thread_tip.start()
        time.sleep(0.5)  # 最初のフレームが入るまで待つ

        self.w = self.OUT_W_LEFT + self.OUT_W_RIGHT
        self.h = self.OUT_H

        self.fig = plt.figure(figsize=(15, 6))
        self.fig.subplots_adjust(wspace=0.5)
        manager = plt.get_current_fig_manager()
        try:
            manager.window.wm_geometry('+20+50')
        except AttributeError:
            try:
                manager.windoe.setGeometry(20, 50, 1500, 600)
            except AttributeError:
                pass
        gs = GridSpec(1, 5)

        label_fontsize = 16
        title_fontsize = 17
        tick_fontsize = 13

        self.axL = self.fig.add_subplot(gs[:, 0:2])
        self.axM = self.fig.add_subplot(gs[:, 2:3])
        self.axR = self.fig.add_subplot(gs[:, 3:5])

        self.axL.set_title('Captured RGB image', fontsize=title_fontsize)
        self.axL.set_xlabel('image width [pixel]', fontsize=label_fontsize)
        self.axL.set_ylabel('image height [pixel]', fontsize=label_fontsize)

        self.axM.set_title('Normal force command', fontsize=title_fontsize)
        self.axM.set_xlabel('Shear force x [N]', fontsize=label_fontsize)
        self.axM.set_ylabel('Normal force z [N]', fontsize=label_fontsize)

        # ====== タイトルに今回の回転オフセットを表示（変更） ======
        self.axR.set_title(f'Shear force command\n(rotation offset: {self.direction_offset_deg:.1f} deg)', fontsize=title_fontsize)
        # ====== ここまで変更 ======
        self.axR.set_xlabel('Shear force x [N]', fontsize=label_fontsize)
        self.axR.set_ylabel('Shear force y [N]', fontsize=label_fontsize)

        for ax in (self.axL, self.axM, self.axR):
            ax.tick_params(axis='both', labelsize=tick_fontsize)

        self.axM.set_xlim(-1, 1)
        self.axM.set_ylim(-0.5, 10.5)
        self.axR.set_xlim(-6, 6)
        self.axR.set_ylim(-6, 6)

        self.t = 0.00
        self.rec_t = 0.00
        self.line_M, = self.axM.plot([], [], color='green', label='example')
        self.now_R, = self.axM.plot([], [], color='red', marker='.', markersize=10, label='now_ex')
        self.rec1, = self.axM.plot([], [], color='blue', marker='.', markersize=10, label='before')
        self.rec2, = self.axM.plot([], [], color='blue', marker='.', markersize=10, label='recording')
        self.line, = self.axR.plot([], [], color='green', label='example')
        self.now_F, = self.axR.plot([], [], color='red', marker='.', markersize=10, label='now_ex')
        self.rec3, = self.axR.plot([], [], color='blue', marker='.', markersize=10, label='before')
        self.rec4, = self.axR.plot([], [], color='blue', marker='.', markersize=10, label='recording')
        self.image_init0 = np.zeros((self.h, self.w, 3), dtype='uint8')
        self.image_plt = self.axL.imshow(self.image_init0, animated=True)

        # ★ FPS計測用
        self._fps_prev_time = time.perf_counter()
        self._fps = 0.0

        self.t_line = np.arange(0, 5, 0.001)
        self.Fz_line = self.Fz / 2 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
        self.Fzz_line = 0
        self.Ff_line = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
        self.Fr_line = 0
        self.now_Ff = 0.0
        self.now_Fr = 0.0
        self.now_Fzz = 0.0

    # ★ 爪カメラを読み続けるスレッド
    def _read_nail(self):
        while self._cam_running:
            ret, frame = self.cap_nail.read()
            if ret and frame is not None:
                with self._lock_nail:
                    self._frame_nail = frame

    # ★ 指先カメラを読み続けるスレッド
    def _read_tip(self):
        while self._cam_running:
            ret, frame = self.cap_tip.read()
            if ret and frame is not None:
                with self._lock_tip:
                    self._frame_tip = frame

    # ====== 回転オフセット適用用ヘルパー（追加） ======
    def _rotate_fr_ff(self, fr, ff):
        """
        お手本の軌跡・目標点(Fr, Ff)を self._rotation_rad だけ時計回りに回転する。
        fr, ff はスカラーでもnumpy配列でもそのまま使える。
        実際にロードセルで測定した値は回転しない（あくまでガイド表示のみ回す）。
        """
        cos_t = math.cos(self._rotation_rad)
        sin_t = math.sin(self._rotation_rad)
        fr_rot = fr * cos_t + ff * sin_t
        ff_rot = -fr * sin_t + ff * cos_t
        return fr_rot, ff_rot
    # ====== ここまで追加 ======

    def onkey(self, event):
        if event.key == 'escape':
            print('esc')
            ser_flag.value = False
            self._cam_running = False  # ★ スレッド停止
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
            plt.close(event.canvas.figure)

        if event.key == 'r':
            print('r')
            rec_flag.value = True
            print(rec_flag.value)
            self.graphstart = time.perf_counter()

    def updateframe(self, dum):
        # ★ FPS計算
        now = time.perf_counter()
        dt = now - self._fps_prev_time
        self._fps_prev_time = now
        if dt > 0:
            self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1  # 指数移動平均
        print(f'\rFPS: {self._fps:.1f}', end='')

        # ★ カメラを直接読まずバッファから取得する
        with self._lock_nail:
            base_n = self._frame_nail
        with self._lock_tip:
            base_t = self._frame_tip

        if not hasattr(self, '_printed_shape'):
            self._printed_shape = True
            print('capture nail shape:', None if base_n is None else base_n.shape)
            print('capture tip  shape:', None if base_t is None else base_t.shape)

        if (base_n is None) or (base_t is None):
            return self.rec1, self.rec3, self.image_plt, self.line, self.line_M

        roi_n, _ = crop_with_center_wh_safe(base_n, self.n_cx, self.n_cy, self.n_w, self.n_h)
        roi_t, _ = crop_with_center_wh_safe(base_t, self.t_cx, self.t_cy, self.t_w, self.t_h)
        nail_in = _resize_no_pad_center_crop(roi_n, self.OUT_W_LEFT, self.OUT_H)
        tip_in = _resize_no_pad_center_crop(roi_t, self.OUT_W_RIGHT, self.OUT_H)
        net_in = np.concatenate([nail_in, tip_in], axis=1)  # BGRのまま保存

        net_rgb = cv2.cvtColor(net_in, cv2.COLOR_BGR2RGB)
        self.image_plt.set_array(net_rgb)

        self.Fz_line = self.Fz / 2 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
        self.Fzz_line = 0

        if self.rec_t < 5:
            self.Ff_line = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
        elif (5 <= self.rec_t and self.rec_t < 10) or (55 <= self.rec_t and self.rec_t < 60):
            self.Ff_line = (math.sqrt(3) / 2) * (self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line)))
        elif (10 <= self.rec_t and self.rec_t < 15) or (50 <= self.rec_t and self.rec_t < 55):
            self.Ff_line = (1 / 2) * (self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line)))
        elif (15 <= self.rec_t and self.rec_t < 20) or (45 <= self.rec_t and self.rec_t < 50):
            self.Ff_line = 0
        elif (20 <= self.rec_t and self.rec_t < 25) or (40 <= self.rec_t and self.rec_t < 45):
            self.Ff_line = (1 / 2) * self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.t_line) - 1)
        elif (25 <= self.rec_t and self.rec_t < 30) or (35 <= self.rec_t and self.rec_t < 40):
            self.Ff_line = (math.sqrt(3) / 2) * (self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.t_line) - 1))
        elif (30 <= self.rec_t and self.rec_t < 35):
            self.Ff_line = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line)) * (-1)

        if (self.rec_t < 5) or (30 <= self.rec_t and self.rec_t < 35):
            self.Fr_line = 0
        elif (5 <= self.rec_t and self.rec_t < 10) or (35 <= self.rec_t and self.rec_t < 40):
            self.Fr_line = (1 / math.sqrt(3)) * self.Ff_line
        elif (10 <= self.rec_t and self.rec_t < 15) or (40 <= self.rec_t and self.rec_t < 45):
            self.Fr_line = math.sqrt(3) * self.Ff_line
        elif (15 <= self.rec_t and self.rec_t < 20):
            self.Fr_line = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
        elif (25 <= self.rec_t and self.rec_t < 30) or (55 <= self.rec_t and self.rec_t < 60):
            self.Fr_line = (-1) * (1 / math.sqrt(3)) * self.Ff_line
        elif (20 <= self.rec_t and self.rec_t < 25) or (50 <= self.rec_t and self.rec_t < 55):
            self.Fr_line = (-1) * math.sqrt(3) * self.Ff_line
        elif (45 <= self.rec_t and self.rec_t < 50):
            self.Fr_line = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line)) * (-1)

        # ====== 回転オフセットを適用（追加） ======
        # 実際の測定値(Fr,Ff)は回転しない。お手本の矢印だけをずらすことで
        # 人がそれに追従すれば、施行ごとに違う方向の力が自然と入るようになる。
        self.Fr_line, self.Ff_line = self._rotate_fr_ff(self.Fr_line, self.Ff_line)
        # ====== ここまで追加 ======

        self.line.set_data(self.Fr_line, self.Ff_line)
        self.line_M.set_data(self.Fzz_line, self.Fz_line)

        Fz = normal_force.value / self.N2gf
        Fzz = 0
        Fr = shear_force1.value - shear_force3.value
        Ff = shear_force2.value - shear_force4.value

        if rec_flag.value:
            self.t = time.perf_counter() - self.graphstart
            self.rec_t = self.t - self.t_rest

            if self.rec_t < 0:
                print(self.rec_t)

                prep_t = self.t
                self.t_line = np.arange(0.0, self.t_rest, 0.001)
                self.Fz_line = self.Fz / 2 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
                self.Fzz_line = np.zeros_like(self.t_line)
                self.Ff_line = np.zeros_like(self.t_line)
                self.Fr_line = np.zeros_like(self.t_line)

                self.now_Fz = self.Fz / 2 * (1 - np.cos(2 * np.pi * self.fz * prep_t))
                self.now_Fzz = 0
                self.now_Ff = 0
                self.now_Fr = 0

            elif 0 < self.rec_t < self.rectime:
                self.t_line = np.arange(self.rec_t, self.rec_t + 5.0, 0.001)
                print('recording', 'time', round(self.rec_t, 3), 'Ff', round(Ff, 2), 'Fr', round(Fr, 2), 'raw')
                self.now_Fz = self.Fz / 2 * (1 - np.cos(2 * np.pi * self.fz * self.rec_t))
                self.now_Fzz = 0

                if self.rec_t < 5:
                    self.now_Ff = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.rec_t))
                elif (5 <= self.rec_t and self.rec_t < 10) or (55 <= self.rec_t and self.rec_t < 60):
                    self.now_Ff = (math.sqrt(3) / 2) * (self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.rec_t)))
                elif (10 <= self.rec_t and self.rec_t < 15) or (50 <= self.rec_t and self.rec_t < 55):
                    self.now_Ff = (1 / 2) * (self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.rec_t)))
                elif (15 <= self.rec_t and self.rec_t < 20) or (45 <= self.rec_t and self.rec_t < 50):
                    self.now_Ff = 0
                elif (20 <= self.rec_t and self.rec_t < 25) or (40 <= self.rec_t and self.rec_t < 45):
                    self.now_Ff = (1 / 2) * self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.rec_t) - 1)
                elif (25 <= self.rec_t and self.rec_t < 30) or (35 <= self.rec_t and self.rec_t < 40):
                    self.now_Ff = (math.sqrt(3) / 2) * (self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.rec_t) - 1))
                elif (30 <= self.rec_t and self.rec_t < 35):
                    self.now_Ff = self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.rec_t) - 1)

                if 15 <= self.rec_t < 20:
                    self.now_Fr = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.rec_t))
                elif (10 <= self.rec_t < 15) or (20 <= self.rec_t < 25):
                    self.now_Fr = (math.sqrt(3) / 2) * (self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.rec_t)))
                elif (5 <= self.rec_t < 10) or (25 <= self.rec_t < 30):
                    self.now_Fr = (1 / 2) * (self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.rec_t)))
                elif (self.rec_t < 5) or (30 <= self.rec_t < 35):
                    self.now_Fr = 0
                elif (35 <= self.rec_t < 40) or (55 <= self.rec_t < 60):
                    self.now_Fr = (1 / 2) * self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.rec_t) - 1)
                elif (40 <= self.rec_t < 45) or (50 <= self.rec_t < 55):
                    self.now_Fr = (math.sqrt(3) / 2) * (self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.rec_t) - 1))
                elif 45 <= self.rec_t < 50:
                    self.now_Fr = self.Fz / 4 * (np.cos(2 * np.pi * self.fz * self.rec_t) - 1)

                framename = self.save_dir + '/' + str(self.datanum) + '.png'
                cv2.imwrite(framename, net_in)  # BGRのまま保存
                self.data_writing.writerow([framename, Fz, Fr, Ff])
                self.datanum += 1
            else:
                self.data_csv.close()
                ser_flag.value = False
                print('press esc')
                self.ani.event_source.stop()

            self.rec2.set_data(Fzz, Fz)
            self.rec4.set_data(Fr, Ff)

            # ====== 回転オフセットを適用（追加） ======
            self.now_Fr, self.now_Ff = self._rotate_fr_ff(self.now_Fr, self.now_Ff)
            # ====== ここまで追加 ======

            self.now_F.set_data(self.now_Fr, self.now_Ff)
            self.now_R.set_data(self.now_Fzz, self.now_Fz)
            return self.rec2, self.rec4, self.image_plt, self.line, self.line_M, self.now_F, self.now_R
        else:
            self.rec1.set_data(Fzz, Fz)
            self.rec3.set_data(Fr, Ff)
            return self.rec1, self.rec3, self.image_plt, self.line, self.line_M

    def animation(self):
        self.ani = animation.FuncAnimation(self.fig, self.updateframe, interval=0, blit=True)
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

    # daemon=True にして、メイン側が異常終了しても
    # 子プロセスがCOMポートを掴んだまま残らないようにする（変更）
    sub_z = Process(target=gf2000.sub_loop, args=[z_port, ser_flag, normal_force], daemon=True)
    sub_z.start()

    count1 = Value('i', 0); count2 = Value('i', 0); count3 = Value('i', 0); count4 = Value('i', 0)
    sub_xy1 = Process(target=SC800IM700_1.sub_loop, args=[xy_port_1, xy_address_1, ser_flag, shear_force1, count1], daemon=True)
    sub_xy2 = Process(target=SC800IM700_2.sub_loop, args=[xy_port_2, xy_address_2, ser_flag, shear_force2, count2], daemon=True)
    sub_xy3 = Process(target=SC800IM700_3.sub_loop, args=[xy_port_3, xy_address_3, ser_flag, shear_force3, count3], daemon=True)
    sub_xy4 = Process(target=SC800IM700_4.sub_loop, args=[xy_port_4, xy_address_4, ser_flag, shear_force4, count4], daemon=True)
    sub_xy1.start(); sub_xy2.start(); sub_xy3.start(); sub_xy4.start()

    graph = GraphMake()
    graph.animation()