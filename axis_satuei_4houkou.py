# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import ndimage
import time
import math
import sys
import os
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from PIL import Image
import csv
import random
import concurrent.futures
from multiprocessing import Process,Value,Manager
import ctypes
import serial
import pandas as pd
import random

#Serial I2Cの変換基板(SC800IM700)を扱う
#送受信するコマンド仕様はSC800IM700のデータシート参照
#また，ロードセルのアンプ(NAU7802)のデータシートも参照
class SC800IM700_1:
  def __init__(self, ser_port, I2C_address):

    #シリアル通信の開始
    self.ser = serial.Serial(ser_port,
                             baudrate=9600,
                             bytesize=8,
                             parity='N',
                             stopbits=1,
                             timeout=0.01)

    #I2Cデバイスのアドレス定義
    self.read_address = ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    self.write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    self.address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    self.S = b'\x53' #I2C通信開始
    self.P = b'\x50' #I2C通信終了
    self.R = b'\x52' #通信変換基板のレジスタ読む
    self.W = b'\x57'  #通信変換基板のレジスタ書く

    self.I = b'\x49' #GPIO読む
    self.O = b'\x4F' #GPIO書く
    self.Z = b'\x5A' #電源切る

    #ゲイン引数(測定範囲とか変えたかったら適宜変える)
    self.gain = 'x16'

    print("Loadcell UART BRIDGE Connected")

  # 初期値を取得するメソッドを追加
  def initialize_value(self):
    self.initial_value = self.single_read()
    print(f"Initial value set to {self.initial_value}")


  def readRegister(self, address):
    self.ser.flushInput()
    send = (self.R + address + self.P)

    self.ser.write(send)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    if value == '' or value == b'':
      raise Exception('readRegister: No value Exception')

    return ord(value)

  def writeRegister(self, address, data):
    self.ser.flushInput()

    self.ser.write(self.W + address + data + self.P)

    if self.readRegister(address) == data:
      print('Write to register was successful')
    else:
      raise Exception('Register data mismatch from written value.')

  def write_byte_data(self, address, reg, value):
    packet = (self.S +
              self.write_address +
              b'\x02' +
              reg +
              value +
              self.P)

    w = self.ser.write(packet)


  def write_byte(self, value):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              value +
              self.P)

    #送信パケット確認用
    #print(packet.hex())

    w = self.ser.write(packet)

  def read_byte(self, reg):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              reg +
              self.S +
              self.read_address +
              b'\x01' +
              self.P)

    self.ser.flushInput()
    w = self.ser.write(packet)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    return ord(value)

  def read_byte_data(self, reg):

    self.write_byte(b'\x00')
    return self.read_byte(reg)


  #I2Cの接続が正常かチェック
  def connect_check(self):
    connect = hex(self.readRegister(b'\x0A'))
    if connect == "0xf0":
      print("I2C Connect")

    else:
      raise Exception('I2C Connect Error')

  #ゲイン設定値の保存用
  def gain_make(self, base_gain):
    print("base_gain is",base_gain)
    if base_gain == 'x128':
      #3.3V電源,ゲイン128倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x27')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x40
      self.offset05 = 0x00

      #ゲインフィルタ調整値
      self.gain06 = 0x01
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    #0~1071.13gで校正済
    elif base_gain == 'x16':
      #3.3V電源,ゲイン16倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x24')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x03
      self.offset05 = 0xE8

      #ゲインフィルタ調整値
      self.gain06 = 0x0F
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    elif base_gain == 'x1':
      #3.3V電源,ゲイン1倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x20')

      #オフセットフィルタ調整値
      self.offset03 = 0x80
      self.offset04 = 0x00
      self.offset05 = 0xC8

      #ゲインフィルタ調整値
      self.gain06 = 0xFF
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    else:
      raise Exception('Unknown Gain')

    #オフセット値書き込み
    self.write_byte_data(self.address, b'\x02', b'\x32')
    self.write_byte_data(self.address, b'\x03', self.offset03.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x04', self.offset04.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x05', self.offset05.to_bytes(1, 'big'))

    #ゲイン書き込み
    self.write_byte_data(self.address, b'\x02', b'\x33')
    self.write_byte_data(self.address, b'\x06', self.gain06.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x07', self.gain07.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x08', self.gain08.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x09', self.gain09.to_bytes(1, 'big'))


  #センサ準備
  def power_on(self):
    #レジスタのリセット(RR=1)
    self.write_byte_data(self.address, b'\x00', b'\x01')
    print("Register reset")

    #(RR=0,PUD=1)にして200msec以上待つ
    self.write_byte_data(self.address, b'\x00', b'\x0E')
    time.sleep(1)
    print(hex(self.read_byte_data(b'\x00')))
    time.sleep(1)

    #構成の設定
    self.write_byte_data(self.address, b'\x00', b'\xAE')
    self.write_byte_data(self.address, b'\x15', b'\x30')

    #ゲインの調整
    self.gain_make(self.gain)

    #通信レート設定(80SPS)
    self.write_byte_data(self.address, b'\x02', b'\x30')

    time.sleep(1)
    #読み出しスタート
    self.write_byte_data(self.address, b'\x00', b'\xBE')
    print("Loadcell Ready!!")

  #上位2byteを読む(下位のデータはノイズ影響大)
  def single_read(self):

    data1 = self.read_byte(b'\x12')
    data2 = self.read_byte(b'\x13')

    data_row = int.from_bytes(data1.to_bytes(1, 'big') + data2.to_bytes(1, 'big'),
                              'big',
                              signed='False')

    if self.gain == 'x16':
      gf_data = (data_row*10+4739)/11432 #1番
      gf_data = gf_data * (gf_data > 0)

    else:
      raise Exception('It is not calibrated yet.')
      # 初期値を引く
      if self.initial_value is not None:
        gf_data -= self.initial_value
        if gf_data < 0:
            gf_data = 0

      return gf_data

  #サブプロセス準備用，一度シリアル接続を切る
  def sub_ready(self):
    self.ser.close()

  #サブプロセスで並列処理する用
  #共有メモリ変数を更新し続ける
  @classmethod
  def sub_loop(cls, ser_port,I2C_address,ser_flag, shear_force1, count1):
    ser = serial.Serial(ser_port,
                        baudrate=9600,
                        bytesize=8,
                        parity='N',
                        stopbits=1,
                        timeout=0.01)

    #I2Cデバイスのアドレス定義
    read_address =  ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    S = b'\x53' #I2C通信開始
    P = b'\x50'  #I2C通信終了

    while ser_flag.value:
      #上位1バイト目を読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x12' +
                S +
                read_address +
                b'\x01' +
                 P)

      ser.flushInput()
      w = ser.write(packet)
      data1 = ''
      while data1 == '' or data1 == b'':
        data1 = ser.read()

      #上位2バイト目読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x13' +
                S +
                read_address +
                b'\x01' +
                P)
      ser.flushInput()
      w = ser.write(packet)
      data2 = ''
      while data2 == '' or data2 == b'':
        data2 = ser.read()

      #上位2バイトをくっつけた生データ
      data_row = int.from_bytes(data1 + data2, 'big',signed='False')

      #生データを[gf]に変換，変換式は適宜実験値を入力する

      gf_data = (data_row*10+4739)/11432 #1番
      #gf_data = data_row
      count1.value += 1
      if count1.value == 1:
        gf_data_ini = gf_data

      gf_data = gf_data - gf_data_ini
      #共有メモリ変数更新
      shear_force1.value = gf_data * (gf_data > 0)
    ser.close()


class SC800IM700_2:
  def __init__(self, ser_port, I2C_address):

    #シリアル通信の開始
    self.ser = serial.Serial(ser_port,
                             baudrate=9600,
                             bytesize=8,
                             parity='N',
                             stopbits=1,
                             timeout=0.01)

    #I2Cデバイスのアドレス定義
    self.read_address = ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    self.write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    self.address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    self.S = b'\x53' #I2C通信開始
    self.P = b'\x50' #I2C通信終了
    self.R = b'\x52' #通信変換基板のレジスタ読む
    self.W = b'\x57'  #通信変換基板のレジスタ書く

    self.I = b'\x49' #GPIO読む
    self.O = b'\x4F' #GPIO書く
    self.Z = b'\x5A' #電源切る

    #ゲイン引数(測定範囲とか変えたかったら適宜変える)
    self.gain = 'x16'

    print("Loadcell UART BRIDGE Connected")
  # 初期値を取得するメソッドを追加
  def initialize_value(self):
    self.initial_value = self.single_read()
    print(f"Initial value set to {self.initial_value}")

  def readRegister(self, address):
    self.ser.flushInput()
    send = (self.R + address + self.P)

    self.ser.write(send)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    if value == '' or value == b'':
      raise Exception('readRegister: No value Exception')

    return ord(value)

  def writeRegister(self, address, data):
    self.ser.flushInput()

    self.ser.write(self.W + address + data + self.P)

    if self.readRegister(address) == data:
      print('Write to register was successful')
    else:
      raise Exception('Register data mismatch from written value.')

  def write_byte_data(self, address, reg, value):
    packet = (self.S +
              self.write_address +
              b'\x02' +
              reg +
              value +
              self.P)

    w = self.ser.write(packet)


  def write_byte(self, value):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              value +
              self.P)

    #送信パケット確認用
    #print(packet.hex())

    w = self.ser.write(packet)

  def read_byte(self, reg):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              reg +
              self.S +
              self.read_address +
              b'\x01' +
              self.P)

    self.ser.flushInput()
    w = self.ser.write(packet)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    return ord(value)

  def read_byte_data(self, reg):

    self.write_byte(b'\x00')
    return self.read_byte(reg)


  #I2Cの接続が正常かチェック
  def connect_check(self):
    connect = hex(self.readRegister(b'\x0A'))
    if connect == "0xf0":
      print("I2C Connect")

    else:
      raise Exception('I2C Connect Error')

  #ゲイン設定値の保存用
  def gain_make(self, base_gain):
    print("base_gain is",base_gain)
    if base_gain == 'x128':
      #3.3V電源,ゲイン128倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x27')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x40
      self.offset05 = 0x00

      #ゲインフィルタ調整値
      self.gain06 = 0x01
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    #0~1071.13gで校正済
    elif base_gain == 'x16':
      #3.3V電源,ゲイン16倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x24')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x03
      self.offset05 = 0xE8

      #ゲインフィルタ調整値
      self.gain06 = 0x0F
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    elif base_gain == 'x1':
      #3.3V電源,ゲイン1倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x20')

      #オフセットフィルタ調整値
      self.offset03 = 0x80
      self.offset04 = 0x00
      self.offset05 = 0xC8

      #ゲインフィルタ調整値
      self.gain06 = 0xFF
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    else:
      raise Exception('Unknown Gain')

    #オフセット値書き込み
    self.write_byte_data(self.address, b'\x02', b'\x32')
    self.write_byte_data(self.address, b'\x03', self.offset03.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x04', self.offset04.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x05', self.offset05.to_bytes(1, 'big'))

    #ゲイン書き込み
    self.write_byte_data(self.address, b'\x02', b'\x33')
    self.write_byte_data(self.address, b'\x06', self.gain06.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x07', self.gain07.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x08', self.gain08.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x09', self.gain09.to_bytes(1, 'big'))


  #センサ準備
  def power_on(self):
    #レジスタのリセット(RR=1)
    self.write_byte_data(self.address, b'\x00', b'\x01')
    print("Register reset")

    #(RR=0,PUD=1)にして200msec以上待つ
    self.write_byte_data(self.address, b'\x00', b'\x0E')
    time.sleep(1)
    print(hex(self.read_byte_data(b'\x00')))
    time.sleep(1)

    #構成の設定
    self.write_byte_data(self.address, b'\x00', b'\xAE')
    self.write_byte_data(self.address, b'\x15', b'\x30')

    #ゲインの調整
    self.gain_make(self.gain)

    #通信レート設定(80SPS)
    self.write_byte_data(self.address, b'\x02', b'\x30')

    time.sleep(1)
    #読み出しスタート
    self.write_byte_data(self.address, b'\x00', b'\xBE')
    print("Loadcell Ready!!")

  #上位2byteを読む(下位のデータはノイズ影響大)
  def single_read(self):

    data1 = self.read_byte(b'\x12')
    data2 = self.read_byte(b'\x13')

    data_row = int.from_bytes(data1.to_bytes(1, 'big') + data2.to_bytes(1, 'big'),
                              'big',
                              signed='False')

    if self.gain == 'x16':
      gf_data = (data_row * 10 + 4944) / 6792  # 2番
      gf_data = gf_data * (gf_data > 0)

    else:
      raise Exception('It is not calibrated yet.')
      # 初期値を引く
      if self.initial_value is not None:
        gf_data -= self.initial_value
        if gf_data < 0:
            gf_data = 0

      return gf_data


  #サブプロセス準備用，一度シリアル接続を切る
  def sub_ready(self):
    self.ser.close()

  #サブプロセスで並列処理する用
  #共有メモリ変数を更新し続ける
  @classmethod
  def sub_loop(cls, ser_port,I2C_address,ser_flag, shear_force2, count2):
    ser = serial.Serial(ser_port,
                        baudrate=9600,
                        bytesize=8,
                        parity='N',
                        stopbits=1,
                        timeout=0.01)

    #I2Cデバイスのアドレス定義
    read_address =  ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    S = b'\x53' #I2C通信開始
    P = b'\x50'  #I2C通信終了

    while ser_flag.value:
      #上位1バイト目を読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x12' +
                S +
                read_address +
                b'\x01' +
                 P)

      ser.flushInput()
      w = ser.write(packet)
      data1 = ''
      while data1 == '' or data1 == b'':
        data1 = ser.read()

      #上位2バイト目読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x13' +
                S +
                read_address +
                b'\x01' +
                P)
      ser.flushInput()
      w = ser.write(packet)
      data2 = ''
      while data2 == '' or data2 == b'':
        data2 = ser.read()

      #上位2バイトをくっつけた生データ
      data_row = int.from_bytes(data1 + data2, 'big',signed='False')

      #生データを[gf]に変換，変換式は適宜実験値を入力する
      gf_data = (data_row * 10 + 4944) / 6792  # 2番
      #gf_data = data_row
      count2.value += 1
      if count2.value == 1:
        gf_data_ini = gf_data

      gf_data = gf_data - gf_data_ini

      #共有メモリ変数更新
      shear_force2.value = gf_data * (gf_data > 0)
    ser.close()

#荷重計(gf2000)を扱うクラス,引数はつながっているCOMポート
#受け取るデータの仕様などはgf2000のデータシート参照
class gf2000:
  def __init__(self, ser_port):
    self.ser = serial.Serial(ser_port,
                             baudrate=9600,
                             bytesize=serial.SEVENBITS,
                             parity = serial.PARITY_EVEN)
    print("Normal force loadcell Ready!!")

  #サブプロセスで動かすためには一度接続を切らないといけない
  def sub_ready(self):
    self.ser.close()

  #テスト用,数値を一度だけ読んで出力
  def single_read(self):
    line = ''
    while line == '' or line == b'':
      line = self.ser.readline()

    line = line.decode('utf-8')
    line = line[-9:-2]
    normal_force = float(line)
    return normal_force

  #サブプロセスで並列処理する用
  #共有メモリ変数を更新し続ける
  @classmethod
  def sub_loop(cls, ser_port, ser_flag, normal_force):
    #通信を開く
    ser = serial.Serial(ser_port,
                        baudrate=9600,
                        bytesize=serial.SEVENBITS,
                        parity = serial.PARITY_EVEN)
    #ser_flagがtrueの間測定値を更新し続ける
    while ser_flag.value:
      line = "" #受け取るデータ(str型)
      line = ser.readline() #データを受け取り
      line = line.decode('utf-8') #utf-8形式に変換
      line = line[-9:-2]#符号などの情報を除去
      normal_force.value = float(line)#float形式にしてデータ更新

    ser.close()

class SC800IM700_3:
  def __init__(self, ser_port, I2C_address):

    #シリアル通信の開始
    self.ser = serial.Serial(ser_port,
                             baudrate=9600,
                             bytesize=8,
                             parity='N',
                             stopbits=1,
                             timeout=0.01)

    #I2Cデバイスのアドレス定義
    self.read_address = ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    self.write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    self.address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    self.S = b'\x53' #I2C通信開始
    self.P = b'\x50' #I2C通信終了
    self.R = b'\x52' #通信変換基板のレジスタ読む
    self.W = b'\x57'  #通信変換基板のレジスタ書く

    self.I = b'\x49' #GPIO読む
    self.O = b'\x4F' #GPIO書く
    self.Z = b'\x5A' #電源切る

    #ゲイン引数(測定範囲とか変えたかったら適宜変える)
    self.gain = 'x16'

    print("Loadcell UART BRIDGE Connected")

  # 初期値を取得するメソッドを追加
  def initialize_value(self):
    self.initial_value = self.single_read()
    print(f"Initial value set to {self.initial_value}")

  def readRegister(self, address):
    self.ser.flushInput()
    send = (self.R + address + self.P)

    self.ser.write(send)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    if value == '' or value == b'':
      raise Exception('readRegister: No value Exception')

    return ord(value)

  def writeRegister(self, address, data):
    self.ser.flushInput()

    self.ser.write(self.W + address + data + self.P)

    if self.readRegister(address) == data:
      print('Write to register was successful')
    else:
      raise Exception('Register data mismatch from written value.')

  def write_byte_data(self, address, reg, value):
    packet = (self.S +
              self.write_address +
              b'\x02' +
              reg +
              value +
              self.P)

    w = self.ser.write(packet)


  def write_byte(self, value):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              value +
              self.P)

    #送信パケット確認用
    #print(packet.hex())

    w = self.ser.write(packet)

  def read_byte(self, reg):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              reg +
              self.S +
              self.read_address +
              b'\x01' +
              self.P)

    self.ser.flushInput()
    w = self.ser.write(packet)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    return ord(value)

  def read_byte_data(self, reg):

    self.write_byte(b'\x00')
    return self.read_byte(reg)


  #I2Cの接続が正常かチェック
  def connect_check(self):
    connect = hex(self.readRegister(b'\x0A'))
    if connect == "0xf0":
      print("I2C Connect")

    else:
      raise Exception('I2C Connect Error')

  #ゲイン設定値の保存用
  def gain_make(self, base_gain):
    print("base_gain is",base_gain)
    if base_gain == 'x128':
      #3.3V電源,ゲイン128倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x27')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x40
      self.offset05 = 0x00

      #ゲインフィルタ調整値
      self.gain06 = 0x01
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    #0~1071.13gで校正済
    elif base_gain == 'x16':
      #3.3V電源,ゲイン16倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x24')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x03
      self.offset05 = 0xE8

      #ゲインフィルタ調整値
      self.gain06 = 0x0F
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    elif base_gain == 'x1':
      #3.3V電源,ゲイン1倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x20')

      #オフセットフィルタ調整値
      self.offset03 = 0x80
      self.offset04 = 0x00
      self.offset05 = 0xC8

      #ゲインフィルタ調整値
      self.gain06 = 0xFF
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    else:
      raise Exception('Unknown Gain')

    #オフセット値書き込み
    self.write_byte_data(self.address, b'\x02', b'\x32')
    self.write_byte_data(self.address, b'\x03', self.offset03.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x04', self.offset04.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x05', self.offset05.to_bytes(1, 'big'))

    #ゲイン書き込み
    self.write_byte_data(self.address, b'\x02', b'\x33')
    self.write_byte_data(self.address, b'\x06', self.gain06.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x07', self.gain07.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x08', self.gain08.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x09', self.gain09.to_bytes(1, 'big'))


  #センサ準備
  def power_on(self):
    #レジスタのリセット(RR=1)
    self.write_byte_data(self.address, b'\x00', b'\x01')
    print("Register reset")

    #(RR=0,PUD=1)にして200msec以上待つ
    self.write_byte_data(self.address, b'\x00', b'\x0E')
    time.sleep(1)
    print(hex(self.read_byte_data(b'\x00')))
    time.sleep(1)

    #構成の設定
    self.write_byte_data(self.address, b'\x00', b'\xAE')
    self.write_byte_data(self.address, b'\x15', b'\x30')

    #ゲインの調整
    self.gain_make(self.gain)

    #通信レート設定(80SPS)
    self.write_byte_data(self.address, b'\x02', b'\x30')

    time.sleep(1)
    #読み出しスタート
    self.write_byte_data(self.address, b'\x00', b'\xBE')
    print("Loadcell Ready!!")

  #上位2byteを読む(下位のデータはノイズ影響大)
  def single_read(self):

    data1 = self.read_byte(b'\x12')
    data2 = self.read_byte(b'\x13')

    data_row = int.from_bytes(data1.to_bytes(1, 'big') + data2.to_bytes(1, 'big'),
                              'big',
                              signed='False')

    if self.gain == 'x16':
      gf_data = (data_row * 10 - 11400) / 11160 # 3番
      gf_data = gf_data * (gf_data > 0)

    else:
      raise Exception('It is not calibrated yet.')
      # 初期値を引く
      if self.initial_value is not None:
        gf_data -= self.initial_value
        if gf_data < 0:
            gf_data = 0

      return gf_data


  #サブプロセス準備用，一度シリアル接続を切る
  def sub_ready(self):
    self.ser.close()

  #サブプロセスで並列処理する用
  #共有メモリ変数を更新し続ける
  @classmethod
  def sub_loop(cls, ser_port,I2C_address,ser_flag, shear_force3,count3):
    ser = serial.Serial(ser_port,
                        baudrate=9600,
                        bytesize=8,
                        parity='N',
                        stopbits=1,
                        timeout=0.01)

    #I2Cデバイスのアドレス定義
    read_address =  ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    S = b'\x53' #I2C通信開始
    P = b'\x50'  #I2C通信終了

    while ser_flag.value:
      #上位1バイト目を読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x12' +
                S +
                read_address +
                b'\x01' +
                 P)

      ser.flushInput()
      w = ser.write(packet)
      data1 = ''
      while data1 == '' or data1 == b'':
        data1 = ser.read()

      #上位2バイト目読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x13' +
                S +
                read_address +
                b'\x01' +
                P)
      ser.flushInput()
      w = ser.write(packet)
      data2 = ''
      while data2 == '' or data2 == b'':
        data2 = ser.read()

      #上位2バイトをくっつけた生データ
      data_row = int.from_bytes(data1 + data2, 'big',signed='False')

      #生データを[gf]に変換，変換式は適宜実験値を入力する
      gf_data = (data_row * 10 - 11400) / 11160  # 3番
      #gf_data = data_row

      count3.value += 1
      if count3.value == 1:
        gf_data_ini = gf_data

      gf_data = gf_data - gf_data_ini

      #共有メモリ変数更新
      shear_force3.value = gf_data * (gf_data > 0)
    ser.close()

class SC800IM700_4:
  def __init__(self, ser_port, I2C_address):

    #シリアル通信の開始
    self.ser = serial.Serial(ser_port,
                             baudrate=9600,
                             bytesize=8,
                             parity='N',
                             stopbits=1,
                             timeout=0.01)

    #I2Cデバイスのアドレス定義
    self.read_address = ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    self.write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    self.address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    self.S = b'\x53' #I2C通信開始
    self.P = b'\x50' #I2C通信終了
    self.R = b'\x52' #通信変換基板のレジスタ読む
    self.W = b'\x57'  #通信変換基板のレジスタ書く

    self.I = b'\x49' #GPIO読む
    self.O = b'\x4F' #GPIO書く
    self.Z = b'\x5A' #電源切る

    #ゲイン引数(測定範囲とか変えたかったら適宜変える)
    self.gain = 'x16'

    print("Loadcell UART BRIDGE Connected")

  # 初期値を取得するメソッドを追加
  def initialize_value(self):
    self.initial_value = self.single_read()
    print(f"Initial value set to {self.initial_value}")

  def readRegister(self, address):
    self.ser.flushInput()
    send = (self.R + address + self.P)

    self.ser.write(send)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    if value == '' or value == b'':
      raise Exception('readRegister: No value Exception')

    return ord(value)

  def writeRegister(self, address, data):
    self.ser.flushInput()

    self.ser.write(self.W + address + data + self.P)

    if self.readRegister(address) == data:
      print('Write to register was successful')
    else:
      raise Exception('Register data mismatch from written value.')

  def write_byte_data(self, address, reg, value):
    packet = (self.S +
              self.write_address +
              b'\x02' +
              reg +
              value +
              self.P)

    w = self.ser.write(packet)


  def write_byte(self, value):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              value +
              self.P)

    #送信パケット確認用
    #print(packet.hex())

    w = self.ser.write(packet)

  def read_byte(self, reg):
    packet = (self.S +
              self.write_address +
              b'\x01' +
              reg +
              self.S +
              self.read_address +
              b'\x01' +
              self.P)

    self.ser.flushInput()
    w = self.ser.write(packet)

    value = ''
    while value == '' or value == b'':
      value = self.ser.read()

    return ord(value)

  def read_byte_data(self, reg):

    self.write_byte(b'\x00')
    return self.read_byte(reg)


  #I2Cの接続が正常かチェック
  def connect_check(self):
    connect = hex(self.readRegister(b'\x0A'))
    if connect == "0xf0":
      print("I2C Connect")

    else:
      raise Exception('I2C Connect Error')

  #ゲイン設定値の保存用
  def gain_make(self, base_gain):
    print("base_gain is",base_gain)
    if base_gain == 'x128':
      #3.3V電源,ゲイン128倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x27')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x40
      self.offset05 = 0x00

      #ゲインフィルタ調整値
      self.gain06 = 0x01
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    #0~1071.13gで校正済
    elif base_gain == 'x16':
      #3.3V電源,ゲイン16倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x24')

      #オフセットフィルタ調整値
      self.offset03 = 0x00
      self.offset04 = 0x03
      self.offset05 = 0xE8

      #ゲインフィルタ調整値
      self.gain06 = 0x0F
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    elif base_gain == 'x1':
      #3.3V電源,ゲイン1倍に設定
      self.write_byte_data(self.address, b'\x01', b'\x20')

      #オフセットフィルタ調整値
      self.offset03 = 0x80
      self.offset04 = 0x00
      self.offset05 = 0xC8

      #ゲインフィルタ調整値
      self.gain06 = 0xFF
      self.gain07 = 0x00
      self.gain08 = 0x00
      self.gain09 = 0x00

    else:
      raise Exception('Unknown Gain')

    #オフセット値書き込み
    self.write_byte_data(self.address, b'\x02', b'\x32')
    self.write_byte_data(self.address, b'\x03', self.offset03.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x04', self.offset04.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x05', self.offset05.to_bytes(1, 'big'))

    #ゲイン書き込み
    self.write_byte_data(self.address, b'\x02', b'\x33')
    self.write_byte_data(self.address, b'\x06', self.gain06.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x07', self.gain07.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x08', self.gain08.to_bytes(1, 'big'))
    self.write_byte_data(self.address, b'\x09', self.gain09.to_bytes(1, 'big'))


  #センサ準備
  def power_on(self):
    #レジスタのリセット(RR=1)
    self.write_byte_data(self.address, b'\x00', b'\x01')
    print("Register reset")

    #(RR=0,PUD=1)にして200msec以上待つ
    self.write_byte_data(self.address, b'\x00', b'\x0E')
    time.sleep(1)
    print(hex(self.read_byte_data(b'\x00')))
    time.sleep(1)

    #構成の設定
    self.write_byte_data(self.address, b'\x00', b'\xAE')
    self.write_byte_data(self.address, b'\x15', b'\x30')

    #ゲインの調整
    self.gain_make(self.gain)

    #通信レート設定(80SPS)
    self.write_byte_data(self.address, b'\x02', b'\x30')

    time.sleep(1)
    #読み出しスタート
    self.write_byte_data(self.address, b'\x00', b'\xBE')
    print("Loadcell Ready!!")

  #上位2byteを読む(下位のデータはノイズ影響大)
  def single_read(self):

    data1 = self.read_byte(b'\x12')
    data2 = self.read_byte(b'\x13')

    data_row = int.from_bytes(data1.to_bytes(1, 'big') + data2.to_bytes(1, 'big'),
                              'big',
                              signed='False')

    if self.gain == 'x16':
      gf_data = (data_row*10+14240)/11560  # 4番
      gf_data = gf_data * (gf_data > 0)

    else:
      raise Exception('It is not calibrated yet.')
      # 初期値を引く
      if self.initial_value is not None:
        gf_data -= self.initial_value
        if gf_data < 0:
            gf_data = 0

      return gf_data


  #サブプロセス準備用，一度シリアル接続を切る
  def sub_ready(self):
    self.ser.close()

  #サブプロセスで並列処理する用
  #共有メモリ変数を更新し続ける
  @classmethod
  def sub_loop(cls, ser_port,I2C_address,ser_flag, shear_force4, count4):
    ser = serial.Serial(ser_port,
                        baudrate=9600,
                        bytesize=8,
                        parity='N',
                        stopbits=1,
                        timeout=0.01)

    #I2Cデバイスのアドレス定義
    read_address =  ((I2C_address << 1) | 0x01).to_bytes(1,'big') #0x55
    write_address = ((I2C_address << 1) & 0xfe).to_bytes(1,'big')  #0x54
    address = I2C_address.to_bytes(1, 'big')

    #各種コマンド
    S = b'\x53' #I2C通信開始
    P = b'\x50'  #I2C通信終了

    while ser_flag.value:
      #上位1バイト目を読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x12' +
                S +
                read_address +
                b'\x01' +
                 P)

      ser.flushInput()
      w = ser.write(packet)
      data1 = ''
      while data1 == '' or data1 == b'':
        data1 = ser.read()

      #上位2バイト目読む
      packet = (S +
                write_address +
                b'\x01' +
                b'\x13' +
                S +
                read_address +
                b'\x01' +
                P)
      ser.flushInput()
      w = ser.write(packet)
      data2 = ''
      while data2 == '' or data2 == b'':
        data2 = ser.read()

      #上位2バイトをくっつけた生データ
      data_row = int.from_bytes(data1 + data2, 'big',signed='False')

      #生データを[gf]に変換，変換式は適宜実験値を入力する
      gf_data = (data_row*10+14240)/11560  # 4番
      #gf_data = data_row
      count4.value += 1
      if count4.value == 1:
        gf_data_ini = gf_data

      gf_data = gf_data - gf_data_ini
      #共有メモリ変数更新
      shear_force4.value = gf_data * (gf_data > 0)
    ser.close()

class graphmake:
  def __init__(self):
    # ---------------------------初期処理-----------------------------------------------
    nowdir = os.path.dirname(__file__)  # プログラムのあるディレクトリを参照
    print("now_directry:", nowdir)
    os.chdir(nowdir)  # 作業ディレクトリをプログラムのあるディレクトリに
    os.chdir('..')
    # ---------------------------ここまで初期処理---------------------------------------------

    # -----------------------何Nの力まで測定するか選択---------------------------------

    self.Fz = 10  # int型の力,グラフの計算に利用
    self.force_path = "record0-10xyz"

    # -------------------------ここまで力の選択---------------------------------------

    # -------------------------剪断力角度の選択(右方向を0degとして±180deg)-------------

    self.degree_str = 360
    # -------------------------ここまで角度選択--------------------------------------

    # -------------------------保存パスの名前----------------------------------------
    print("名前と測定回数を英字半角で入力してください(例:watanabe1)")
    self.dirname = input()  # 名前の入力
    # -----------------------------------------------------------------------------

    # -------------------------保存パスを作成---------------------------------------

    # ./被験者名/剪断角度となるパス
    self.save_dir = "./datas/" + self.force_path + "/" + self.dirname + "/" + "360" + "deg"
    os.makedirs(self.save_dir)  # 上記のパスを持つディレクトリ作成

    # ./被験者名/剪断角度/datalog.csvとなるパス(ここに測定値を記入)
    self.datalog_path = self.save_dir + "/datalog.csv"
    self.namelist_path = "./datas/" + self.force_path + "/namelist.csv"

    # -------------------------ネームリストに書き込み---------------------------------
    # ./force_pathにあるcsvファイルに各被験者のdatalog.csvを書き込む(機械学習時にここから全データを参照するため)
    self.name_csv = open(self.namelist_path, "a", newline="")
    self.name_writing = csv.writer(self.name_csv)
    record_path = (self.force_path +
                   "/" +
                   self.dirname +
                   "/" +
                   "360" +
                   "deg" +
                   "/datalog.csv")
    self.name_writing.writerow([record_path, "360"])
    self.name_csv.close()
    # ------------------------ここまでネームリスト書き込み------------------------------
    self.data_csv = open(self.datalog_path, "w", newline="")
    self.data_writing = csv.writer(self.data_csv)
    # ------------------------測定値を格納するcsvの準備----------------------------------

    # -----------------------------グラフ用パラメータ----------------------------------------
    # せん断力の測定範囲で一部パラメータを分ける
    if self.Fz == 5:
      self.mu = 0.9  # 摩擦係数
      self.fz = 0.03  # 垂直力振動数[Hz]
      self.T_num = 6  # 測定周期数

    elif self.Fz == 10:
      self.mu = 0.5  # 摩擦係数
      self.fz = 0.05 * 4  # 垂直力振動数[Hz]
      self.T_num = 12  # 測定周期数

    self.Fr = self.Fz * self.mu
    self.now_Fz = 0.0  # 垂直力指令値
    self.now_Fz = 0.0  # せん断力指令値
    self.t_rest = 5.0  # [sec]予備時間
    self.T_count = 0  # 今何周期目か
    self.T = 1 / self.fz  # 指令値の振動周期
    self.rectime = self.T * self.T_num  # 測定時間
    self.recstart = self.t_rest  # 測定開始時間
    self.recfinish = self.t_rest + self.rectime  # 測定終了時間
    self.graphtime = self.rectime + 2 * self.t_rest
    self.frontT = 3.0  # x軸表示範囲
    self.Frange = 0.2  # 拡張値域
    self.N2gf = 101.972  # [N]を[gf]に変換する定数
    self.datanum = 0
    self.error_r = 0.5  # 誤差1N以内のデータを保存
    # ----------------------------------------------------------------------------------------

    # ------------------------画像初期処理-------------------------
    self.roix, self.roiy = 260, 325  # 左上座標
    self.w, self.h = 140, 155  # 幅,高さ
    self.capture = cv2.VideoCapture(0)
    if self.capture.isOpened() is False:
      raise ("IO Error")
    # -----------------------------------------------------------

    # グラフ生成
    self.fig = plt.figure(figsize=(15, 6))
    self.fig.subplots_adjust(wspace=0.5)
    plt.get_current_fig_manager().window.wm_geometry("+20+50")
    gs = GridSpec(1,5)


    # 左側グラフ(現在の爪画像を表示)
    self.axL = self.fig.add_subplot(gs[:,0:2], xlabel="width", ylabel="height")

    # 中央グラフ(2番目のグラフ)
    self.axM = self.fig.add_subplot(gs[:,2:3], xlabel="xForce", ylabel="yForce")

    # 右側グラフ(3番目のグラフ)
    self.axR = self.fig.add_subplot(gs[:,3:5], xlabel="another_x", ylabel="zForce")
    # --------------------------axM設定-------------------------------------------
    self.axM.set_xlim(-1, 1)  # x軸表示範囲
    self.axM.set_ylim(-0.5, 10.5)  # y軸表示範囲
    self.axM.xaxis.set_ticks_position("bottom")  # 目盛りを軸の下に表示


    self.t = 0.00
    self.rec_t = 0.00

    # 数秒後までの指令値軌道を表示するオブジェクト
    self.line_M, = self.axM.plot([], [], color='green', label="example")

    # 指令値をプロットするオブジェクト
    self.now_R, = self.axM.plot([],
                                [],
                                color='red',
                                marker=".",
                                markersize=10,
                                label="now_ex")

    # 記録開始前の計測値をプロットするオブジェクト
    self.rec1, = self.axM.plot([],
                               [],
                               color='blue',
                               marker='.',
                               markersize=10,
                               label='before')

    # 記録中の計測値をプロットするオブジェクト
    self.rec2, = self.axM.plot([],
                               [],
                               color='blue',
                               marker='.',
                               markersize=10,
                               label='recording')
    # -------------------------------------------------------------------------------
    # --------------------------axR設定-------------------------------------------
    self.axR.set_xlim(-6, 6)  # x軸表示範囲
    self.axR.set_ylim(-6, 6)  # y軸表示範囲
    self.axR.xaxis.set_ticks_position("bottom")  # 目盛りを軸の下に表示

    self.t = 0.00
    self.rec_t = 0.00

    # 数秒後までの指令値軌道を表示するオブジェクト
    self.line, = self.axR.plot([], [], color='green', label="example")

    # 指令値をプロットするオブジェクト
    self.now_F, = self.axR.plot([],
                                 [],
                                 color='red',
                                 marker=".",
                                 markersize=10,
                                 label="now_ex")

    # 記録開始前の計測値をプロットするオブジェクト
    self.rec3, = self.axR.plot([],
                               [],
                               color='blue',
                               marker='.',
                               markersize=10,
                               label='before')

    # 記録中の計測値をプロットするオブジェクト
    self.rec4, = self.axR.plot([],
                               [],
                               color='blue',
                               marker='.',
                               markersize=10,
                               label='recording')
    # -------------------------------------------------------------------------------

    # -------------------------axL設定------------------------------
    self.image_init0 = np.zeros((self.h, self.w), dtype="uint8")
    # vmin.vmaxを設定しないとちゃんと表示されない
    self.image_plt = self.axL.imshow(self.image_init0,
                                     animated=True,
                                     cmap="gray",
                                     vmin=0,
                                     vmax=255)
    # --------------------------------------------------------------

    # -----------------------教示グラフ-------------------------------------------------------------------------------------------------
    self.t_line = np.arange(0, 5, 0.001)

    # 指先力の指令値
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

    self.now_Fz = 0.0
    self.now_Ff = 0.0
    self.now_Fr = 0.0
    self.now_Fzz = 0.0

  # -----------キー入力----------------------------
  def onkey(self, event):
    # escキーで終了処理
    if event.key == 'escape':
      print("esc")
      ser_flag.value = False
      self.data_csv.close()
      plt.close(event.canvas.figure)

    # rキーで測定の開始
    if event.key == 'r':
      print("r")
      rec_flag.value = True
      print(rec_flag.value)
      self.graphstart = time.perf_counter()

    # その他のキーなら無視
    if str.isdecimal(event.key):
      pass

  # ---------ここまでキー入力----------------------

  def updateframe(self, dum):
    t0 = time.perf_counter()

    # カメラからフレームを取得
    ret, base = self.capture.read()
    if not ret:
      return self.image_plt,

    # ROIを切り出し（BGR）
    ROI = base[self.roiy:self.roiy + self.h, self.roix:self.roix + self.w]

    # ---- 画像フィルタ（共通） ----
    def apply_filter(img):
      """1ch or 3ch画像に対してGaussian+CLAHEを適用"""
      if img.ndim == 2:  # グレースケール
        blur = cv2.GaussianBlur(img, (5, 5), 1)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        return clahe.apply(blur)
      else:
        channels = cv2.split(img)
        proc = []
        for ch in channels:
          blur = cv2.GaussianBlur(ch, (5, 5), 1)
          clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
          proc.append(clahe.apply(blur))
        return cv2.merge(proc)

    # ---- 3種類の表現を生成 ----
    roi_color = ROI.copy()                              # BGR
    g_channel = roi_color[:, :, 1]                      # G
    g_filtered = apply_filter(g_channel)                # 1ch
    rgb_filtered = apply_filter(roi_color)              # 3ch(BGRのまま)
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    hsv_filtered = apply_filter(hsv)                    # 3ch(HSV)
    hsv_bgr = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)

    # 左側の表示はGチャンネルで
    self.image_plt.set_array(g_filtered)

    # 教示線の更新（self.rec_tに応じて）
    self.Fz_line = self.Fz / 2 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
    self.Fzz_line = 0

    if self.rec_t < 5:
      self.Ff_line = self.Fz / 4 * (1 - np.cos(2 * np.pi * self.fz * self.t_line))
    elif (5 <= self.rec_t < 10) or (55 <= self.rec_t < 60):
      self.Ff_line = (math.sqrt(3)/2) * (self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.t_line)))
    elif (10 <= self.rec_t < 15) or (50 <= self.rec_t < 55):
      self.Ff_line = 0.5 * (self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.t_line)))
    elif (15 <= self.rec_t < 20) or (45 <= self.rec_t < 50):
      self.Ff_line = 0
    elif (20 <= self.rec_t < 25) or (40 <= self.rec_t < 45):
      self.Ff_line = 0.5 * self.Fz/4 * (np.cos(2*np.pi*self.fz*self.t_line) - 1)
    elif (25 <= self.rec_t < 30) or (35 <= self.rec_t < 40):
      self.Ff_line = (math.sqrt(3)/2) * (self.Fz/4 * (np.cos(2*np.pi*self.fz*self.t_line) - 1))
    elif (30 <= self.rec_t < 35):
      self.Ff_line = - self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.t_line))

    if (self.rec_t < 5) or (30 <= self.rec_t < 35):
      self.Fr_line = 0
    elif (5 <= self.rec_t < 10) or (35 <= self.rec_t < 40):
      self.Fr_line = (1/math.sqrt(3)) * self.Ff_line
    elif (10 <= self.rec_t < 15) or (40 <= self.rec_t < 45):
      self.Fr_line = math.sqrt(3) * self.Ff_line
    elif (15 <= self.rec_t < 20):
      self.Fr_line = self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.t_line))
    elif (25 <= self.rec_t < 30) or (55 <= self.rec_t < 60):
      self.Fr_line = - (1/math.sqrt(3)) * self.Ff_line
    elif (20 <= self.rec_t < 25) or (50 <= self.rec_t < 55):
      self.Fr_line = - math.sqrt(3) * self.Ff_line
    elif (45 <= self.rec_t < 50):
      self.Fr_line = - self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.t_line))

    # 教示線を描画
    self.line.set_data(self.Fr_line, self.Ff_line)
    self.line_M.set_data(self.Fzz_line, self.Fz_line)

    # センサ値（共有メモリから）
    Fz = normal_force.value / self.N2gf
    Fzz = 0
    Fr = shear_force1.value - shear_force3.value
    Ff = shear_force2.value - shear_force4.value

    # 記録状態
    if rec_flag.value:
      # タイミング更新
      self.t = time.perf_counter() - self.graphstart
      self.rec_t = self.t - self.t_rest

      if self.rec_t < 0:
        self.t_line = np.arange(0.0, 5.0, 0.001)

      elif 0 < self.rec_t < self.rectime:
        self.t_line = np.arange(self.rec_t, self.rec_t + 5.0, 0.001)

        # 指令値（表示用）
        self.now_Fz  = self.Fz/2 * (1 - np.cos(2*np.pi*self.fz*self.rec_t))
        self.now_Fzz = 0

        if self.rec_t < 5:
          self.now_Ff = self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.rec_t))
        elif (5 <= self.rec_t < 10) or (55 <= self.rec_t < 60):
          self.now_Ff = (math.sqrt(3)/2) * (self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.rec_t)))
        elif (10 <= self.rec_t < 15) or (50 <= self.rec_t < 55):
          self.now_Ff = 0.5 * (self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.rec_t)))
        elif (15 <= self.rec_t < 20) or (45 <= self.rec_t < 50):
          self.now_Ff = 0
        elif (20 <= self.rec_t < 25) or (40 <= self.rec_t < 45):
          self.now_Ff = 0.5 * self.Fz/4 * (np.cos(2*np.pi*self.fz*self.rec_t) - 1)
        elif (25 <= self.rec_t < 30) or (35 <= self.rec_t < 40):
          self.now_Ff = (math.sqrt(3)/2) * (self.Fz/4 * (np.cos(2*np.pi*self.fz*self.rec_t) - 1))
        elif (30 <= self.rec_t < 35):
          self.now_Ff = (self.Fz/4 * (np.cos(2*np.pi*self.fz*self.rec_t) - 1))

        if (15 <= self.rec_t < 20):
          self.now_Fr = self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.rec_t))
        elif (10 <= self.rec_t < 15) or (20 <= self.rec_t < 25):
          self.now_Fr = (math.sqrt(3)/2) * (self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.rec_t)))
        elif (5 <= self.rec_t < 10) or (25 <= self.rec_t < 30):
          self.now_Fr = 0.5 * (self.Fz/4 * (1 - np.cos(2*np.pi*self.fz*self.rec_t)))
        elif (self.rec_t < 5) or (30 <= self.rec_t < 35):
          self.now_Fr = 0
        elif (35 <= self.rec_t < 40) or (55 <= self.rec_t < 60):
          self.now_Fr = 0.5 * (self.Fz/4 * (np.cos(2*np.pi*self.fz*self.rec_t) - 1))
        elif (40 <= self.rec_t < 45) or (50 <= self.rec_t < 55):
          self.now_Fr = (math.sqrt(3)/2) * (self.Fz/4 * (np.cos(2*np.pi*self.fz*self.rec_t) - 1))
        elif (45 <= self.rec_t < 50):
          self.now_Fr = (self.Fz/4 * (np.cos(2*np.pi*self.fz*self.rec_t) - 1))

        # ここで3種類を保存
        basepath = f"{self.save_dir}/{self.datanum:06d}"
        cv2.imwrite(basepath + "_G.png",   g_filtered)   # 1ch
        cv2.imwrite(basepath + "_RGB.png", rgb_filtered) # 3ch(BGR)
        cv2.imwrite(basepath + "_HSV.png", hsv_bgr)      # 3ch(BGR)

        # CSVは基準として _G を記録（学習時は *_G/_RGB/_HSV を拾う）
        self.data_writing.writerow([basepath + "_G.png", Fz, Fr, Ff])
        self.datanum += 1

      else:
        # 記録終了
        self.data_csv.close()
        ser_flag.value = False
        print("press esc")
        self.ani.event_source.stop()

    # プロット更新
    self.rec2.set_data(Fzz, Fz)
    self.rec4.set_data(Fr, Ff)
    self.now_F.set_data(self.now_Fr, self.now_Ff)
    self.now_R.set_data(self.now_Fzz, self.now_Fz)

    if rec_flag.value:
      return self.rec2, self.rec4, self.image_plt, self.line, self.line_M, self.now_F, self.now_R
    else:
      # 待機時は中央/右に現在値を置いておく
      self.rec1.set_data(Fzz, Fz)
      self.rec3.set_data(Fr, Ff)
      return self.rec1, self.rec3, self.image_plt, self.line, self.line_M

  # グラフを随時更新する関数（クラス内に戻す！）
  def animation(self):
    self.ani = animation.FuncAnimation(self.fig,
                                       self.updateframe,
                                       interval=0,
                                       blit=True)
    self.cid = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
  #--------メモリ共有変数(グローバル変数と違うがどこからでもアクセスできる)
  normal_force = Value('f', 0.00)
  shear_force1 = Value('f', 0.00)
  shear_force2 = Value('f', 0.00)
  shear_force3 = Value('f', 0.00)
  shear_force4 = Value('f', 0.00)
  ser_flag = Value('b', True)   #シリアル通信フラグ(Trueで荷重計，MD共にループ開始)
  rec_flag = Value('b',False)   #測定フラグ(これがTrueの間測定)
  #-----------------------------------

  #ロードセル測定準備
  xy_port_1 = "COM8"
  xy_address_1 = 0x2A
  shear_loadcell_1 = SC800IM700_1(xy_port_1, xy_address_1)#クラスの定義
  shear_loadcell_1.power_on() #ロードセルの通信開始
  shear_loadcell_1.sub_ready()#サブプロセスの準備

  #ロードセル測定準備
  xy_port_2 = "COM9"
  xy_address_2 = 0x2A
  shear_loadcell_2 = SC800IM700_2(xy_port_2, xy_address_2)#クラスの定義
  shear_loadcell_2.power_on() #ロードセルの通信開始
  shear_loadcell_2.sub_ready()#サブプロセスの準備

  # ロードセル測定準備
  xy_port_3 = "COM10"
  xy_address_3 = 0x2A
  shear_loadcell_3 = SC800IM700_3(xy_port_3, xy_address_3)#クラスの定義
  shear_loadcell_3.power_on() #ロードセルの通信開始
  shear_loadcell_3.sub_ready()#サブプロセスの準備

  # ロードセル測定準備
  xy_port_4 = "COM12"
  xy_address_4 = 0x2A
  shear_loadcell_4 = SC800IM700_4(xy_port_4, xy_address_4)#クラスの定義
  shear_loadcell_4.power_on() #ロードセルの通信開始
  shear_loadcell_4.sub_ready()#サブプロセスの準備


  #荷重計測定準備
  z_port = "COM13"
  normal_loadcell = gf2000(z_port)  #クラスの定義
  normal_loadcell.sub_ready() #サブプロセスの準備

  """
  サブプロセス開始(各種通信)
  並列処理したい関数がクラス内の関数(メソッド)の場合エラーが起きる
  その場合，メソッドをクラスメソッドとして定義してやると動かせる
  """
  #垂直力計測プロセスの定義
  sub_z = Process(target=gf2000.sub_loop,
                  args=[z_port,ser_flag,normal_force])

  #垂直力計測プロセスの開始
  sub_z.start()

  count1 = Value('i', 0)
  count2 = Value('i', 0)
  count3 = Value('i', 0)
  count4 = Value('i', 0)

  #せん断力計測プロセスの定義
  sub_xy1 = Process(target=SC800IM700_1.sub_loop,
                   args=[xy_port_1,xy_address_1,ser_flag,shear_force1, count1])
  sub_xy2 = Process(target=SC800IM700_2.sub_loop,
                   args=[xy_port_2,xy_address_2,ser_flag,shear_force2, count2])
  sub_xy3 = Process(target=SC800IM700_3.sub_loop,
                   args=[xy_port_3,xy_address_3,ser_flag,shear_force3, count3])
  sub_xy4 = Process(target=SC800IM700_4.sub_loop,
                   args=[xy_port_4,xy_address_4,ser_flag,shear_force4, count4])

  #せん断力計測プロセスの開始
  sub_xy1.start()
  sub_xy2.start()
  sub_xy3.start()
  sub_xy4.start()

  graph = graphmake()

  #グラフ表示
  graph.animation()



  #-----各サブプロセス終了
  if ser_flag ==False:
    sub_z.join()
    sub_xy1.join()
    sub_xy2.join()
    sub_xy3.join()
    sub_xy4.join()
    pass
