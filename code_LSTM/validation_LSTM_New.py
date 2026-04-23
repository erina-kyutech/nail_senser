from __future__ import print_function
from re import X
from statistics import mean
import os
from collections import deque
import numpy as np
import tensorflow as tf
import cv2
from keras.applications import VGG16
from tensorflow import keras
from keras import regularizers
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Dropout, Lambda

from tensorflow.python.keras import models
from keras.optimizers import adam_v2 as Adam
from tensorflow.python.keras.constraints import non_neg
from keras.layers import GlobalMaxPooling2D

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

import matplotlib.pyplot as plt
import pandas as pd
import h5py
import smtplib

# ===== LSTM追加のために必要なimport（追加）=====
from keras.layers import Reshape, LSTM
# ===== 本当の時系列入力に必要（追加）=====
from keras.layers import TimeDistributed


# ===== 本当の時系列の「連続フレーム長」を指定（追加）=====
# 例：4なら「4枚連続フレーム」を1サンプルとして扱う
SEQUENCE_LENGTH = 4


# ディレクトリの移動
# データセットのあるフォルダや結果を保存するフォルダへアクセスするために
# ソースコードのあるフォルダから1つ上のディレクトリへ移動
def directry_initialize():
    nowdir = os.path.dirname(__file__)  # プログラムのあるディレクトリを参照
    os.chdir(nowdir)  # 作業ディレクトリをプログラムのあるディレクトリに
    os.chdir("..")  # 一つ上のディレクトリに移動


# データを読み出すためのクラス
class data_loader(object):
    def __init__(self, name=None, Fz_range=10.0, dummy_flag=False, num_workers=8):
        self.dummy_flag = dummy_flag
        self.name = name
        # 垂直力の測定範囲に応じて正規化定数を変える
        if Fz_range == 5.0:
            self.normal_force_normalize = 5.0
            self.thear_force_normalize = 4.0

        elif Fz_range == 10.0:
            self.normal_force_normalize = 10.0
            self.thear_force_normalize = 5.0

    # 個人のデータを読む
    def personal_dataload(self):
        # ダミーモードの時はデータのごく一部しか読み込まない
        # csvパスの指定
        if self.dummy_flag:
            namelist_path = "./datas/record0-10xyz/namelist_dum.csv"
        else:
            namelist_path = "./datas/record0-10xyz/namelist_all10.csv"

        # ネームリストを読み込み
        names = pd.read_csv(namelist_path, header=None)

        alldatas_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy"])

        # ネームリスト内のパスを順に参照しその中のデータを順に格納
        for names_index, names_item in names.iterrows():
            # 各フォルダのデータリスト(画像のパスと測定した力を格納したcsv)のパスを参照
            now_namedir = "./datas/" + names_item[0]

            # 引数の文字列(被験者名)がパスに含まれているものだけを読み込み
            if self.name in now_namedir:
                # 画像パスと指先力測定値を記録したcsvを開く
                csv_record = pd.read_csv(now_namedir, header=0)

                now_degree = int(names_item[1])  # せん断角度[degree]
                now_rad = np.deg2rad(now_degree)  # せん断角度[radian]

                csv_record.columns = ["path", "Fz", "Fr"]
                img_path = csv_record["path"]
                Fz = csv_record["Fz"]
                Fx = csv_record["Fr"] * np.cos(now_rad)
                Fy = csv_record["Fr"] * np.sin(now_rad)

                append_df = pd.concat([img_path, Fz, Fx, Fy], axis=1)
                append_df.columns = ["img_path", "Fz", "Fx", "Fy"]
                alldatas_df = pd.concat([alldatas_df, append_df])

        return alldatas_df

    # データの正規化
    def data_normalize(self, X, Y):

        # Xの正規化
        X = X.astype("float64")

        # ===== 本当の時系列対応：Xが 5次元 (N,T,H,W,1) の場合は reshape しない（変更）=====
        # （旧コードの reshape((-1,155,140,1)) は時系列を潰してしまうので）
        if X.ndim == 4:
            X = X.reshape((-1, 155, 140, 1))  # 旧仕様互換用

        # 255で割って0~1.0の範囲にする
        X /= 255.0

        # Yの正規化
        # 垂直力の正規化(0~1の範囲に)
        Y[:, 0] /= self.normal_force_normalize
        # せんだん力の正規化(0~1の範囲に)
        Y[:, 1] += self.thear_force_normalize
        Y[:, 2] += self.thear_force_normalize
        Y[:, 1] /= (self.thear_force_normalize * 2)
        Y[:, 2] /= (self.thear_force_normalize * 2)
        return X, Y

    # 正規化されたデータを元に戻す関数
    def data_unnormalize(self, Y):
        # 垂直力を戻す
        Y[:, 0] *= self.normal_force_normalize
        # せんだん力の正規化
        Y[:, 1] *= (self.thear_force_normalize * 2)
        Y[:, 2] *= (self.thear_force_normalize * 2)
        Y[:, 1] -= self.thear_force_normalize
        Y[:, 2] -= self.thear_force_normalize

        return Y


# 自作層(グレースケールをカラー画像にする)関数
def tensor_gray2BGR(grayX):
    # ===== 本当の時系列対応：axis=3固定だと TimeDistributed/5D で壊れるので axis=-1 に（変更）=====
    blank = tf.zeros_like(grayX)
    BGR_X = tf.concat([blank, grayX], axis=-1)
    BGR_X = tf.concat([BGR_X, blank], axis=-1)
    return BGR_X


class multitask_CNN(object):
    def __init__(self):

        # ★ 何よりも先に、必ずここ
        self.model_dir = (
            "C:/Users"
            + "/Hiraku Komura"
            + "/PycharmProjects/result/CNN_result/vgg16_gray_Reshape/"
        )

        # ★ その次に existence check
        if os.path.exists(self.model_dir):
            pass
        else:
            os.mkdir(self.model_dir)
            os.mkdir(self.model_dir + "weight")
            os.mkdir(self.model_dir + "indiv_score")

        # ★ 最後にモデル構築
        self.CNN_build()


    # モデルの構築
    def CNN_build(self):
        l2_alpha = 0.001  # L2正則化の係数
        he_normal_seed = 111  # 乱数シード

        # 隠れ層ノード数は(入力ノード数+出力ノード数)*2/3が目安らしい
        middle_class_recurrence = 342  # 全結合中間層のノード数

        last_activation = "linear"  # 最終層の活性化関数

        # 最適化手法
        # optimizer=Adam(lr=1e-04, decay=1e-6, beta_1=0.9, beta_2=0.999)

        # モデルの保存するパス

        # ===== 本当の時系列：入力を (T,155,140,1) に（変更）=====
        input_tensor = Input(shape=(SEQUENCE_LENGTH, 155, 140, 1), name="input_tensor")

        # ===== 本当の時系列：TimeDistributed で gray->BGR を各フレームに適用（変更）=====
        conv_input = TimeDistributed(
            Lambda(tensor_gray2BGR,
                   output_shape=(155, 140, 3),
                   name="gray2BGR")
        )(input_tensor)

        conv_inputshape = (155, 140, 3)

        # ===== 本当の時系列：VGG16 を TimeDistributed で各フレームに適用（変更）=====
        vgg = VGG16(weights="imagenet",
                    input_shape=conv_inputshape,
                    include_top=False)

        conv = TimeDistributed(vgg)(conv_input)
        # conv の形： (None, T, H, W, C) 例：(None, T, 4, 4, 512)

        # ===== 本当の時系列：各フレームの特徴をプーリングして (None,T,512) に（変更）=====
        pooled = TimeDistributed(GlobalMaxPooling2D())(conv)
        # pooled の形： (None, T, 512)

        # ===== 本当の時系列：時間方向に LSTM（変更）=====
        lstm_out = LSTM(128, name="lstm_layer")(pooled)
        flatten = lstm_out
        # ===== 変更ここまで（以降の全結合はそのまま）=====

        ##ここから全結合層を構築
        # 垂直力推定タスク(回帰)
        Fz = Dense(middle_class_recurrence,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(l2_alpha))(flatten)
        Fz = Dropout(0.2)(Fz)
        Fz = Dense(1,
                   activation=last_activation,
                   name="Fz",
                   kernel_regularizer=regularizers.l2(l2_alpha))(Fz)

        # せん断力推定タスク(回帰)
        Fx = Dense(middle_class_recurrence,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(l2_alpha))(flatten)
        Fx = Dropout(0.2)(Fx)
        Fx = Dense(1,
                   activation=last_activation,
                   name="Fx",
                   kernel_regularizer=regularizers.l2(l2_alpha))(Fx)

        # せん断力推定タスク(回帰)
        Fy = Dense(middle_class_recurrence,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(l2_alpha))(flatten)
        Fy = Dropout(0.2)(Fy)
        Fy = Dense(1,
                   activation=last_activation,
                   name="Fy",
                   kernel_regularizer=regularizers.l2(l2_alpha))(Fy)

        # 角度回帰タスク(回帰)
        Angle = Dense(middle_class_recurrence,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(l2_alpha))(flatten)
        Angle = Dropout(0.2)(Angle)
        Angle = Dense(1,
                      activation=last_activation,
                      name="Angle",
                      kernel_regularizer=regularizers.l2(l2_alpha))(Angle)

        # 最終出力のリストを作成
        predicts = [Fz, Fx, Fy, Angle]

        # モデルを構成
        self.model = Model(input_tensor, predicts)

        # ===== ここは元の流れを崩さず、jsonが無ければ作ってから読む（最小追加）=====
        model_json_path = self.model_dir + "for0-10_reshape.json"
        if os.path.exists(model_json_path) is False:
            json_string = self.model.to_json()
            open(model_json_path, "w").write(json_string)

        model_json_string = open(model_json_path).read()

        # モデル構造の読み込み
        self.model = model_from_json(model_json_string,
                                     custom_objects={'tf': tf, 'K': K})

        # ===== 重みが存在する場合のみロード（最小追加）=====
        model_weight_path = self.model_dir + "weight/weight_ryusetsu2_for0-10_reshape.h5"
        if os.path.exists(model_weight_path):
            self.model.load_weights(model_weight_path)
        else:
            # 学習前などで重みが無い場合はここに来る
            pass

    # モデル構造と重みを読み込み
    def model_load_from_path(self):
        model_json_path = self.model_dir + "for0-10_reshape.json"
        model_json_string = open(model_json_path).read()

        # モデル構造の読み込み
        self.model = model_from_json(model_json_string,
                                     custom_objects={'tf': tf, 'K': K})

        # モデル重みの読み込み
        # model_weight_path = self.model_dir+"/weight_komura_n7_for0-10.h5"
        # self.model.load_weights(model_weight_path)


# 実際に学習に使うクラス
class Trainer(object):
    def __init__(self, model_obj, datas_obj):
        self.datas = datas_obj
        self.model = model_obj
        self.name = self.datas.name

    # 一人のデータ
    def base_train(self):
        # 個人データを読み取る
        personaldatas_df = self.datas.personal_dataload()

        # ===== 本当の時系列：シーケンス開始点（start index）を作る（追加）=====
        # 連続フレームT枚が取れる start index のみを使う
        data_length = len(personaldatas_df)
        max_start = data_length - SEQUENCE_LENGTH
        if max_start < 0:
            raise ValueError("データ数がSEQUENCE_LENGTHより少ないため、時系列サンプルを作れません。")

        # フォルダをまたぐと「連続フレーム」にならないので、startとendの親ディレクトリ一致を条件にする
        valid_start_indices = []
        for s in range(0, max_start + 1):
            p0 = personaldatas_df.iloc[s, 0]
            p1 = personaldatas_df.iloc[s + SEQUENCE_LENGTH - 1, 0]
            if os.path.dirname(p0) == os.path.dirname(p1):
                valid_start_indices.append(s)

        valid_start_indices = np.array(valid_start_indices, dtype="uint32")
        valid_start_indices = np.random.permutation(valid_start_indices)

        # 評価用データ
        X_test, Y_test = self.data_indexread(personaldatas_df, valid_start_indices)

        # 正規化
        X_test, Y_test = self.datas.data_normalize(X_test, Y_test)

        # =============================ここから評価データでの評価==================================================
        self.evaluate_save(personaldatas_df, valid_start_indices, keyword="val")

    # データの読み込み
    # ===== 本当の時系列：index_array は「シーケンス開始index」の配列に変更（変更）=====
    def data_indexread(self, datas_df, index_array):

        X_seq_list = []
        Y_list = []

        print("\r", "now image loading", end="")

        for start_idx in index_array:
            # T枚連続で読み込む
            frames = []
            for t in range(SEQUENCE_LENGTH):
                path = datas_df.iloc[start_idx + t, 0]
                img = cv2.imread(path, 0)
                img = img.reshape(155, 140, 1)
                frames.append(img)

            # (T,155,140,1)
            seq = np.array(frames)
            X_seq_list.append(seq)

            # ラベルは「最後のフレーム」の力を採用
            Y_array = datas_df.iloc[start_idx + SEQUENCE_LENGTH - 1, [1, 2, 3]].values
            Y_list.append(Y_array)

        X_array = np.array(X_seq_list)   # (N,T,155,140,1)
        Y_array = np.array(Y_list)       # (N,3)

        return X_array, Y_array

    # 推定値と真値との差を記録する関数、keywordで名前を設定
    def evaluate_save(self, datas_df, eval_index_array, keyword=""):
        # データ格納用のディレクトリがあるか確認、無かったら作る
        indivisual_dir = self.model.model_dir + "indiv_score/" + self.name + "/"
        if os.path.exists(indivisual_dir):
            pass
        else:
            os.mkdir(indivisual_dir)

        # データを分割して読み出す数
        split_num = 30000
        # データの総数
        data_length = len(eval_index_array)
        index_split_num = int(data_length / split_num) + 1

        # 分割データ数がデータ総数より多いなら一括処理
        if split_num > data_length:
            # データを読み出し
            X, Y = self.data_indexread(datas_df, eval_index_array)

            # 正規化
            X, Y = self.datas.data_normalize(X, Y)
            # 評価する
            Y_predict_list = self.model.model.predict(X)
            Y_predict = np.concatenate([Y_predict_list[0],
                                        Y_predict_list[1],
                                        Y_predict_list[2]],
                                       axis=1)

            # 正規化状態から戻す
            Y_true = self.datas.data_unnormalize(Y)
            Y_predict = self.datas.data_unnormalize(Y_predict)

            Fz_pred = Y_predict[:, 0]
            Fx_pred = Y_predict[:, 1]
            Fy_pred = Y_predict[:, 2]
            Fz_true = Y_true[:, 0]
            Fx_true = Y_true[:, 1]
            Fy_true = Y_true[:, 2]
            Fz_error = Fz_pred - Fz_true
            Fx_error = Fx_pred - Fx_true
            Fy_error = Fy_pred - Fy_true

            eval_log_df = pd.DataFrame([
                Fz_pred,
                Fx_pred,
                Fy_pred,
                Fz_true,
                Fx_true,
                Fy_true,
                Fz_error,
                Fx_error,
                Fy_error],
                index=[
                    "Fz_predict",
                    "Fx_predict",
                    "Fy_predict",
                    "Fz_true",
                    "Fx_true",
                    "Fy_true",
                    "Fz_error",
                    "Fx_error",
                    "Fy_error"])
        else:
            index_array = np.array_split(eval_index_array, index_split_num)

            read_length = len(index_array)
            for i in range(read_length):
                now_index_array = index_array[i]

                X, Y = self.data_indexread(datas_df, now_index_array)

                X, Y = self.datas.data_normalize(X, Y)

                Y_predict_list = self.model.model.predict(X,
                                                          batch_size=128,
                                                          verbose=1)
                Y_predict = np.concatenate([Y_predict_list[0],
                                            Y_predict_list[1],
                                            Y_predict_list[2]],
                                           axis=1)

                Y_predict = self.datas.data_unnormalize(Y_predict)
                Y_true = self.datas.data_unnormalize(Y)

                Fz_pred = Y_predict[:, 0]
                Fx_pred = Y_predict[:, 1]
                Fy_pred = Y_predict[:, 2]
                Fz_true = Y_true[:, 0]
                Fx_true = Y_true[:, 1]
                Fy_true = Y_true[:, 2]
                Fz_error = Fz_pred - Fz_true
                Fx_error = Fx_pred - Fx_true
                Fy_error = Fy_pred - Fy_true

                if i == 0:
                    eval_log_df = pd.DataFrame([
                        Fz_pred,
                        Fx_pred,
                        Fy_pred,
                        Fz_true,
                        Fx_true,
                        Fy_true,
                        Fz_error,
                        Fx_error,
                        Fy_error],
                        index=[
                            "Fz_predict",
                            "Fx_predict",
                            "Fy_predict",
                            "Fz_test",
                            "Fx_test",
                            "Fy_test",
                            "Fz_error",
                            "Fx_error",
                            "Fy_error"])
                else:
                    concat_df = pd.DataFrame([
                        Fz_pred,
                        Fx_pred,
                        Fy_pred,
                        Fz_true,
                        Fx_true,
                        Fy_true,
                        Fz_error,
                        Fx_error,
                        Fy_error],
                        index=[
                            "Fz_predict",
                            "Fx_predict",
                            "Fy_predict",
                            "Fz_test",
                            "Fx_test",
                            "Fy_test",
                            "Fz_error",
                            "Fx_error",
                            "Fy_error"])
                    eval_log_df = pd.concat([eval_log_df, concat_df], axis=1)

        if self.datas.dummy_flag:
            eval_log_path = indivisual_dir + "evaluate_" + keyword + "_for0-10_dum.csv"
        else:
            eval_log_path = indivisual_dir + "evaluate_" + keyword + "_for0-10.csv"
        eval_log_df = eval_log_df.transpose()

        eval_log_df.to_csv(eval_log_path, encoding="shift-jis")


if __name__ == "__main__":

    namelist = ["ryusetsu2"]
    directry_initialize()

    for now_name in namelist:
        CNN = multitask_CNN()
        database = data_loader(name=now_name, Fz_range=10.0, dummy_flag=False)
        trainer = Trainer(CNN, database)
        trainer.base_train()
        del CNN, database, trainer
