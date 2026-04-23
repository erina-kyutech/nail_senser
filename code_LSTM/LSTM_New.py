from __future__ import print_function
from statistics import mean
import os
from collections import deque
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from keras import regularizers
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model, model_from_json
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Lambda,
    LSTM,
    TimeDistributed,
    Reshape,
    Flatten,
)  # Reshape, LSTMを保持

from keras import models
from keras.optimizers import Adam
from keras.constraints import non_neg
from keras.layers import GlobalMaxPooling2D
from keras.utils import plot_model

import matplotlib.pyplot as plt
import pandas as pd
import h5py
import smtplib
import gc

tf.compat.v1.disable_eager_execution()

# =========================================================
# ここが「本当の時系列」の要：連続フレーム枚数（ステップ数）
# 例：4なら「4フレームの並び」を1サンプルとして学習します
# =========================================================
SEQUENCE_LENGTH = 4

# 既存定数（元コードの雰囲気を崩さないため残置）
NEW_SEQUENCE_LENGTH = 4

# モデル共通の定数 (mini.pyより採用)
L2_ALPHA = 0.001  # L2正則化の係数
MIDDLE_CLASS_RECURRENCE = 342  # 全結合中間層のノード数
LSTM_UNITS = 128  # LSTM層のユニット数
LAST_ACTIVATION = "linear"  # 最終層の活性化関数


# ディレクトリの移動
def directry_initialize():
    nowdir = os.path.dirname(__file__)  # プログラムのあるディレクトリを参照
    os.chdir(nowdir)  # 作業ディレクトリをプログラムのあるディレクトリに
    os.chdir("..")  # 一つ上のディレクトリに移動


# データを読み出すためのクラス
class data_loader(object):
    def __init__(self, name=None, Fz_range=10.0, dummy_flag=False, num_workers=8):
        self.dummy_flag = dummy_flag
        self.name = name
        if Fz_range == 5.0:
            self.normal_force_normalize = 5.0
            self.thear_force_normalize = 4.0
        elif Fz_range == 10.0:
            self.normal_force_normalize = 10.0
            self.thear_force_normalize = 5.0

    # 個人のデータを読む
    def personal_dataload(self):
        if self.dummy_flag:
            namelist_path = "./datas/record0-10xyz/namelist_dum.csv"
        else:
            namelist_path = "./datas/record0-10xyz/namelist.csv"

        names = pd.read_csv(namelist_path, header=None)

        alldatas_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy"])

        for names_index, names_item in names.iterrows():
            now_namedir = "./datas/" + names_item[0]

            if self.name in now_namedir:
                csv_record = pd.read_csv(now_namedir, header=0)

                now_degree = int(names_item[1])  # せん断角度[degree]
                now_rad = np.deg2rad(now_degree)  # せん断角度[radian]

                csv_record.columns = ["path", "Fz", "Fr", "Ff"]
                img_path = csv_record["path"]
                Fz = csv_record["Fz"]
                Fx = csv_record["Fr"]
                Fy = csv_record["Ff"]

                append_df = pd.concat([img_path, Fz, Fx, Fy], axis=1)
                append_df.columns = ["img_path", "Fz", "Fx", "Fy"]
                alldatas_df = pd.concat([alldatas_df, append_df])

        return alldatas_df

    # データの正規化
    # Xは (サンプル数, T, 155, 140, 1) を想定
    def data_normalize(self, X, Y):
        X = X.astype("float64")
        X /= 255.0

        Y[:, 0] /= self.normal_force_normalize
        Y[:, 1] += self.thear_force_normalize
        Y[:, 2] += self.thear_force_normalize
        Y[:, 1] /= self.thear_force_normalize * 2
        Y[:, 2] /= self.thear_force_normalize * 2
        return X, Y

    def data_unnormalize(self, Y):
        Y[:, 0] *= self.normal_force_normalize
        Y[:, 1] *= self.thear_force_normalize * 2
        Y[:, 2] *= self.thear_force_normalize * 2
        Y[:, 1] -= self.thear_force_normalize
        Y[:, 2] -= self.thear_force_normalize
        return Y


# 自作層(グレースケールをカラー画像にする)関数
def tensor_gray2BGR(grayX):
    blank = tf.zeros_like(grayX)
    BGR_X = tf.concat([blank, grayX], axis=-1)
    BGR_X = tf.concat([BGR_X, blank], axis=-1)
    return BGR_X


class multitask_CNN(object):
    def __init__(self):
        # 保存先は元コードのまま（フォルダ名も維持）
        self.model_dir = "./result/CNN_result/vgg16_gray_Reshape" + "/"

        if os.path.exists(self.model_dir):
            pass
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            os.mkdir(self.model_dir + "weight")
            os.mkdir(self.model_dir + "indiv_score")

        self.CNN_build()

    def CNN_build(self):
        l2_alpha = L2_ALPHA
        middle_class_recurrence = MIDDLE_CLASS_RECURRENCE
        lstm_units = LSTM_UNITS
        last_activation = LAST_ACTIVATION

        optimizer = Adam(lr=1e-04, decay=1e-6, beta_1=0.9, beta_2=0.999)

        # 本当の時系列モデルなのでファイル名を分ける（上書き事故防止）
        model_path = self.model_dir + "for0-10_seq.json"
        model_fig_path = self.model_dir + "for0-10_seq.png"

        # =========================================================
        # ここが変更点：入力が (T,155,140,1)
        # =========================================================
        input_tensor = Input(
            shape=(SEQUENCE_LENGTH, 155, 140, 1), name="input_tensor"
        )

        # =========================================================
        # ここが変更点：TimeDistributed で各フレームに同じ処理を適用
        # =========================================================
        conv_input = TimeDistributed(
            Lambda(tensor_gray2BGR, output_shape=(155, 140, 3), name="gray2BGR"),
            name="td_gray2BGR",
        )(input_tensor)

        conv_inputshape = (155, 140, 3)

        # VGG16（全結合なし）を1回だけ作ってTimeDistributedで使い回す
        base_vgg = VGG16(
            weights="imagenet", input_shape=conv_inputshape, include_top=False
        )

        conv = TimeDistributed(base_vgg, name="td_vgg16")(conv_input)
        # conv: (None, T, 4, 4, 512)

        # (4,4,512) -> (512) を各フレームで作る
        frame_feat = TimeDistributed(
            GlobalMaxPooling2D(name="flatten"),
            name="td_gap",
        )(conv)
        # frame_feat: (None, T, 512)

        # =========================================================
        # ここが「本当の時系列」：T個のフレーム特徴(512次元)をLSTMへ
        # =========================================================
        lstm_out = LSTM(
            lstm_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,
            name="lstm_layer",
        )(frame_feat)
        # lstm_out: (None, 128)

        final_feature = lstm_out

        # 全結合（元のマルチタスク構造は維持）
        Fz = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha),
            name="Fz_middle",
        )(final_feature)
        Fz = Dropout(0.2)(Fz)
        Fz = Dense(
            1,
            activation=last_activation,
            name="Fz",
            kernel_regularizer=regularizers.l2(l2_alpha),
        )(Fz)

        Fx = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha),
            name="Fx_middle",
        )(final_feature)
        Fx = Dropout(0.2)(Fx)
        Fx = Dense(
            1,
            activation=last_activation,
            name="Fx",
            kernel_regularizer=regularizers.l2(l2_alpha),
        )(Fx)

        Fy = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha),
            name="Fy_middle",
        )(final_feature)
        Fy = Dropout(0.2)(Fy)
        Fy = Dense(
            1,
            activation=last_activation,
            name="Fy",
            kernel_regularizer=regularizers.l2(l2_alpha),
        )(Fy)

        predicts = [Fz, Fx, Fy]

        self.model = Model(input_tensor, predicts)

        self.model.compile(
            loss={
                "Fz": "mean_squared_error",
                "Fx": "mean_squared_error",
                "Fy": "mean_squared_error",
            },
            optimizer=optimizer,
        )

        if os.path.exists(model_path):
            pass
        else:
            json_string = self.model.to_json()
            open(model_path, "w").write(json_string)

        if os.path.exists(model_fig_path):
            pass
        else:
            plot_model(self.model, show_shapes=True, to_file=model_fig_path)

    # モデル構造と重みを読み込み
    def model_load_from_path(self, name):
        model_json_path = self.model_dir + "for0-10_seq.json"
        model_json_string = open(model_json_path).read()

        self.model = model_from_json(
            model_json_string, custom_objects={"tf": tf, "K": K}
        )

        # 重みも本当の時系列用として分ける
        model_weight_path = (
            self.model_dir + "/weight/weight_" + name + "_for0-10_seq.h5"
        )
        self.model.load_weights(model_weight_path)


class Trainer(object):
    def __init__(self, model_obj, datas_obj):
        self.datas = datas_obj
        self.model = model_obj
        self.name = self.datas.name

        if os.path.exists(self.model.model_dir + "indiv_score/" + self.name):
            pass
        else:
            os.mkdir(self.model.model_dir + "indiv_score/" + self.name)

        if self.datas.dummy_flag:
            self.epoch = 2
        else:
            self.epoch = 20

        self.batch_size = 32
        self.sequence_length = SEQUENCE_LENGTH

    # =========================================================
    # ここが変更点：連続T枚を作れる開始indexだけを使う
    # さらに「別フォルダ（別計測）を跨がない」ようにする
    # =========================================================
    def make_valid_start_indices(self, datas_df):
        valid = []
        T = self.sequence_length
        n = len(datas_df)
        for i in range(0, n - T + 1):
            d0 = os.path.dirname(str(datas_df.iloc[i, 0]))
            ok = True
            for k in range(1, T):
                dk = os.path.dirname(str(datas_df.iloc[i + k, 0]))
                if dk != d0:
                    ok = False
                    break
            if ok:
                valid.append(i)
        return np.array(valid, dtype="uint32")

    def base_train(self):
        callbacks = []

        personaldatas_df = self.datas.personal_dataload()

        # 連続T枚が作れる開始indexのみ抽出
        valid_start_indices = self.make_valid_start_indices(personaldatas_df)
        data_length = len(valid_start_indices)

        data_index_array = np.array(range(data_length), dtype="uint32")
        data_index_array = np.random.permutation(data_index_array)

        # valid_start_indices をシャッフルして使う
        start_index_array = valid_start_indices[data_index_array]

        train_rate = 0.85
        train_index_array, test_index_array = np.split(
            start_index_array, [int(len(start_index_array) * train_rate)]
        )

        X_train, Y_train = self.data_indexread(personaldatas_df, train_index_array)
        X_test, Y_test = self.data_indexread(personaldatas_df, test_index_array)

        X_train, Y_train = self.datas.data_normalize(X_train, Y_train)
        X_test, Y_test = self.datas.data_normalize(X_test, Y_test)

        history = self.model.model.fit(
            X_train,
            [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]],
            epochs=self.epoch,
            validation_data=(X_test, [Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]]),
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        if self.datas.dummy_flag:
            weight_path = (
                self.model.model_dir
                + "weight/weight_"
                + self.name
                + "_for0-10"
                + "_seq_dum.h5"
            )
        else:
            weight_path = (
                self.model.model_dir
                + "weight/weight_"
                + self.name
                + "_for0-10"
                + "_seq.h5"
            )

        if os.path.exists(self.model.model_dir + "weight"):
            pass
        else:
            os.mkdir(self.model.model_dir + "weight")

        self.model.model.save_weights(weight_path)

        if self.datas.dummy_flag:
            log_csvpath = (
                self.model.model_dir
                + "indiv_score/"
                + self.name
                + "/learning_log_seq_dum.csv"
            )
        else:
            log_csvpath = (
                self.model.model_dir
                + "indiv_score/"
                + self.name
                + "/learning_log_seq.csv"
            )

        log_df = pd.DataFrame(
            history.history,
            columns=[
                "Fx_loss",
                "Fy_loss",
                "Fz_loss",
                "loss",
                "val_Fx_loss",
                "val_Fy_loss",
                "val_Fz_loss",
                "val_loss",
            ],
        )
        log_df.to_csv(log_csvpath)

        self.evaluate_save(personaldatas_df, train_index_array, keyword="train")
        self.evaluate_save(personaldatas_df, test_index_array, keyword="val")

    # =========================================================
    # ここが変更点：indexは「シーケンス開始index」
    # Xは (N, T, 155, 140, 1)
    # Yは「最後のフレームの力」を教師にします（一般的）
    # =========================================================
    def data_indexread(self, datas_df, index_array):
        X_list = []
        Y_list = []
        T = self.sequence_length

        print("\r", "now image loading", end="")

        for start_index in index_array:
            seq_imgs = []
            for k in range(T):
                X_path = datas_df.iloc[start_index + k, 0]
                img = cv2.imread(X_path, 0)
                img_array = img.reshape(155, 140, 1)
                seq_imgs.append(img_array)

            seq_imgs = np.array(seq_imgs)  # (T,155,140,1)
            X_list.append(seq_imgs)

            # 教師は「シーケンス最後のフレーム」
            Y_array = datas_df.iloc[start_index + (T - 1), [1, 2, 3]].values
            Y_list.append(Y_array)

        X_array = np.array(X_list)  # (N,T,155,140,1)
        Y_array = np.array(Y_list)  # (N,3)

        return X_array, Y_array

    def evaluate_save(self, datas_df, eval_index_array, keyword=""):
        indivisual_dir = self.model.model_dir + "indiv_score/" + self.name + "/"
        if os.path.exists(indivisual_dir):
            pass
        else:
            os.mkdir(indivisual_dir)

        split_num = 80000
        data_length = len(eval_index_array)
        index_split_num = int(data_length / split_num) + 1

        if split_num > data_length:
            X, Y_true = self.data_indexread(datas_df, eval_index_array)

            X, Y_true_norm = self.datas.data_normalize(X, Y_true.copy())
            Y_predict_list = self.model.model.predict(
                X, batch_size=self.batch_size, verbose=1
            )
            Y_predict_norm = np.concatenate(
                [Y_predict_list[0], Y_predict_list[1], Y_predict_list[2]], axis=1
            )

            Y_predict = self.datas.data_unnormalize(Y_predict_norm.copy())

            Fz_pred = Y_predict[:, 0]
            Fx_pred = Y_predict[:, 1]
            Fy_pred = Y_predict[:, 2]
            Fz_true = Y_true[:, 0]
            Fx_true = Y_true[:, 1]
            Fy_true = Y_true[:, 2]
            Fz_error = Fz_pred - Fz_true
            Fx_error = Fx_pred - Fx_true
            Fy_error = Fy_pred - Fy_true

            eval_log_df = pd.DataFrame(
                [
                    Fz_pred,
                    Fx_pred,
                    Fy_pred,
                    Fz_true,
                    Fx_true,
                    Fy_true,
                    Fz_error,
                    Fx_error,
                    Fy_error,
                ],
                index=[
                    "Fz_predict",
                    "Fx_predict",
                    "Fy_predict",
                    "Fz_true",
                    "Fx_true",
                    "Fy_true",
                    "Fz_error",
                    "Fx_error",
                    "Fy_error",
                ],
            )
        else:
            index_array_split = np.array_split(eval_index_array, index_split_num)

            read_length = len(index_array_split)
            for i in range(read_length):
                now_index_array = index_array_split[i]

                X, Y_true = self.data_indexread(datas_df, now_index_array)

                X, Y_true_norm = self.datas.data_normalize(X, Y_true.copy())
                Y_predict_list = self.model.model.predict(
                    X, batch_size=self.batch_size, verbose=1
                )
                Y_predict_norm = np.concatenate(
                    [Y_predict_list[0], Y_predict_list[1], Y_predict_list[2]], axis=1
                )

                Y_predict = self.datas.data_unnormalize(Y_predict_norm.copy())

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
                    eval_log_df = pd.DataFrame(
                        [
                            Fz_pred,
                            Fx_pred,
                            Fy_pred,
                            Fz_true,
                            Fx_true,
                            Fy_true,
                            Fz_error,
                            Fx_error,
                            Fy_error,
                        ],
                        index=[
                            "Fz_predict",
                            "Fx_predict",
                            "Fy_predict",
                            "Fz_true",
                            "Fx_true",
                            "Fy_true",
                            "Fz_error",
                            "Fx_error",
                            "Fy_error",
                        ],
                    )
                else:
                    concat_df = pd.DataFrame(
                        [
                            Fz_pred,
                            Fx_pred,
                            Fy_pred,
                            Fz_true,
                            Fx_true,
                            Fy_true,
                            Fz_error,
                            Fx_error,
                            Fy_error,
                        ],
                        index=[
                            "Fz_predict",
                            "Fx_predict",
                            "Fy_predict",
                            "Fz_true",
                            "Fx_true",
                            "Fy_true",
                            "Fz_error",
                            "Fx_error",
                            "Fy_error",
                        ],
                    )
                    eval_log_df = pd.concat([eval_log_df, concat_df], axis=1)

        if self.datas.dummy_flag:
            eval_log_path = (
                indivisual_dir + "evaluate_" + keyword + "_for0-10_seq_dum.csv"
            )
        else:
            eval_log_path = indivisual_dir + "evaluate_" + keyword + "_for0-10_seq.csv"

        eval_log_df = eval_log_df.transpose()
        eval_log_df.to_csv(eval_log_path, encoding="shift-jis")


if __name__ == "__main__":
    dummy_flag = False
    test_size = 0.15
    split_seed = 111

    batch_size = 32

    if dummy_flag:
        namelist = ["ryusetsu"]
        epoch = 2
    else:
        namelist = ["ryusetsu"]
        epoch = 20

    directry_initialize()

    for now_name in namelist:
        CNN = multitask_CNN()
        database = data_loader(name=now_name, Fz_range=10.0, dummy_flag=dummy_flag)
        trainer = Trainer(CNN, database)

        trainer.base_train()

        del CNN, database, trainer
        gc.collect()
