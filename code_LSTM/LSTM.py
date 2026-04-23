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

# グローバルなシーケンス長の設定
# VGG16の出力の空間次元を利用して時間軸を生成するため、入力画像は1枚とする
SEQUENCE_LENGTH = 1

# VGG16の出力形状 (4, 4, 512) を (4, 2048) にReshapeする際の新しいシーケンス長
# VGG16の出力のHまたはWが4になることを想定 (155x140 / 32 = 4.8x4.3, VGG16はmax-poolingで5回ダウンサンプリングするため、4x4が出力される)
NEW_SEQUENCE_LENGTH = 4
# NEW_FEATURES_DIM = 2048 # 4 * 512 <- この定数を削除し、コード内で動的に計算します

# モデル共通の定数 (mini.pyより採用)
L2_ALPHA = 0.001  # L2正則化の係数
MIDDLE_CLASS_RECURRENCE = 342  # 全結合中間層のノード数
LSTM_UNITS = 128  # LSTM層のユニット数
LAST_ACTIVATION = "linear"  # 最終層の活性化関数


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
            namelist_path = "./datas/record0-10xyz/namelist.csv"

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
    # Xはシーケンスデータとして扱うため、reshapeはTrainer側で行う
    def data_normalize(self, X, Y):
        # Xの正規化
        # Xは (サンプル数, 155, 140, 1) の形状を想定
        X = X.astype("float64")

        # 255で割って0~1.0の範囲にする
        X /= 255.0

        # Yの正規化
        # 垂直力の正規化(0~1の範囲に)
        Y[:, 0] /= self.normal_force_normalize
        # せんだん力の正規化(0~1の範囲に)
        Y[:, 1] += self.thear_force_normalize
        Y[:, 2] += self.thear_force_normalize
        Y[:, 1] /= self.thear_force_normalize * 2
        Y[:, 2] /= self.thear_force_normalize * 2
        return X, Y

    # 正規化されたデータを元に戻す関数
    def data_unnormalize(self, Y):
        # 垂直力を戻す
        Y[:, 0] *= self.normal_force_normalize
        # せんだん力の正規化
        Y[:, 1] *= self.thear_force_normalize * 2
        Y[:, 2] *= self.thear_force_normalize * 2
        Y[:, 1] -= self.thear_force_normalize
        Y[:, 2] -= self.thear_force_normalize

        return Y


# 自作層(グレースケールをカラー画像にする)関数
# 入力形状は[..., H, W, 1]となる
def tensor_gray2BGR(grayX):
    # TimeDistributedを外したため、axis=3をaxis=-1に修正（チャンネル軸）
    # TimeDistributedを使用しない場合、入力は(None, 155, 140, 1)
    blank = tf.zeros_like(grayX)

    # 軸-1（チャンネル軸）に対して結合
    # [..., 1] -> [..., 2] (blank, grayX)
    BGR_X = tf.concat([blank, grayX], axis=-1)
    # [..., 2] -> [..., 3] (blank, grayX, blank) -> (B, G, R)の順ではないが、VGG16の入力として機能する
    BGR_X = tf.concat([BGR_X, blank], axis=-1)

    return BGR_X


class multitask_CNN(object):
    # クラスを呼び出したときに同時に呼び出される関数
    def __init__(self):
        # モデル構造や重み、結果を保存するディレクトリ
        # モデル名にReshapeを追加
        self.model_dir = "./result/CNN_result/vgg16_gray_Reshape" + "/"

        # 指定したディレクトリがなければ作る
        if os.path.exists(self.model_dir):
            pass
        else:
            os.mkdir(self.model_dir)
            os.mkdir(self.model_dir + "weight")
            os.mkdir(self.model_dir + "indiv_score")

        # モデルを構築
        self.CNN_build()

    # モデルの構築
    def CNN_build(self):
        l2_alpha = L2_ALPHA
        middle_class_recurrence = MIDDLE_CLASS_RECURRENCE
        lstm_units = LSTM_UNITS
        last_activation = LAST_ACTIVATION

        # Reshape後の新しいシーケンス長（時間軸）
        new_seq_len = NEW_SEQUENCE_LENGTH  # 4

        # 最適化手法
        optimizer = Adam(lr=1e-04, decay=1e-6, beta_1=0.9, beta_2=0.999)

        # モデルの保存するパス
        model_path = self.model_dir + "for0-10_reshape.json"
        model_fig_path = self.model_dir + "for0-10_reshape.png"

        # ここからモデルの構築
        # 入力画像は1枚なので (None, 155, 140, 1) を想定
        input_tensor = Input(shape=(155, 140, 1), name="input_tensor")

        # グレースケール画像をカラー画像(チャンネル数3)に変換
        conv_input = Lambda(
            tensor_gray2BGR, output_shape=(155, 140, 3), name="gray2BGR"
        )(input_tensor)
        conv_inputshape = (155, 140, 3)

        # VGG16を呼び出し(全結合層は含まない)
        conv = VGG16(
            weights="imagenet", input_shape=conv_inputshape, include_top=False
        )(conv_input)
        # convの形状は (None, 4, 4, 512)

        # 畳み込み層の出力をReshape
        # (None, 4, 4, 512) -> (None, 4, 2048) へ変換
        # 空間次元 (H=4) を新しい時間軸として利用 (new_seq_len=4)

        # Reshape後の特徴量次元 (W * C = 4 * 512) を動的に計算するためのLambda関数
        def calc_reshape_dim(conv_output):
            # conv_outputの形状は (None, H, W, C)
            # HはNEW_SEQUENCE_LENGTHとして利用 (ここでは4)
            # W (axis=-2, shape[2]) と C (axis=-1, shape[3]) の積を計算
            conv_shape = K.int_shape(conv_output)
            W = conv_shape[2]  # 4
            C = conv_shape[3]  # 512
            new_features = W * C  # 4 * 512 = 2048
            return new_features

        # ラムダ層で動的に特徴量次元を計算
        # K.int_shapeの結果はテンソルではないため、Reshape層の引数として直接渡すことはできません。
        # そこで、Reshape層に直接計算結果を渡すために、ローカル変数として計算します。
        # VGG16の出力は固定されているため、ここではコードの意図を明示するために計算を記述します。
        # K.int_shapeは実行時ではなく、グラフ構築時に形状を取得します。

        # Reshape後の特徴量次元 (W * C = 4 * 512) を取得
        conv_shape = K.int_shape(conv)  # (None, 4, 4, 512)
        new_features = conv_shape[2] * conv_shape[3]  # 4 * 512 = 2048

        # Reshape層の定義
        # (None, 4, 4, 512) -> (None, 4, 2048)
        reshape = Reshape((new_seq_len, new_features), name="reshape_for_lstm")(conv)

        # reshapeの出力をLSTM層に入力
        # return_sequences=False (デフォルト) のため、出力は (None, lstm_units)
        lstm_out = LSTM(
            lstm_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=False,  # 最終ステップの出力のみを使用
            name="lstm_layer",
        )(reshape)

        # lstm_outを全結合層の入力とする
        final_feature = lstm_out  # 形状: (None, 128)

        ##ここから全結合層を構築 (Multitask-CNNの構造は維持)
        # 垂直力推定タスク(回帰)
        Fz = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha),
            name="Fz_middle",
        )(
            final_feature
        )  # nameを追加 (mini.pyと同様に)
        Fz = Dropout(0.2)(Fz)
        Fz = Dense(
            1,
            activation=last_activation,
            name="Fz",
            kernel_regularizer=regularizers.l2(l2_alpha),
        )(Fz)

        # せん断力推定タスク(回帰)
        Fx = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha),
            name="Fx_middle",
        )(
            final_feature
        )  # nameを追加 (mini.pyと同様に)
        Fx = Dropout(0.2)(Fx)
        Fx = Dense(
            1,
            activation=last_activation,
            name="Fx",
            kernel_regularizer=regularizers.l2(l2_alpha),
        )(Fx)

        # せん断力推定タスク(回帰)
        Fy = Dense(
            middle_class_recurrence,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_alpha),
            name="Fy_middle",
        )(
            final_feature
        )  # nameを追加 (mini.pyと同様に)
        Fy = Dropout(0.2)(Fy)
        Fy = Dense(
            1,
            activation=last_activation,
            name="Fy",
            kernel_regularizer=regularizers.l2(l2_alpha),
        )(Fy)

        # 最終出力のリストを作成
        predicts = [Fz, Fx, Fy]

        # モデルを構成
        self.model = Model(input_tensor, predicts)

        # モデルの構成を表示(主にデバッグ用)
        # self.model.summary()

        # モデルのコンパイル
        self.model.compile(
            loss={
                "Fz": "mean_squared_error",
                "Fx": "mean_squared_error",
                "Fy": "mean_squared_error",
            },
            optimizer=optimizer,
        )

        # モデルの構造json形式でを保存
        if os.path.exists(model_path):
            pass
        else:
            json_string = self.model.to_json()
            open(model_path, "w").write(json_string)

        # モデルの構造を示す図を保存
        if os.path.exists(model_fig_path):
            pass
        else:
            plot_model(self.model, show_shapes=True, to_file=model_fig_path)

    # モデル構造と重みを読み込み
    def model_load_from_path(self, name):
        model_json_path = self.model_dir + "for0-10_reshape.json"
        model_json_string = open(model_json_path).read()

        # モデル構造の読み込み
        self.model = model_from_json(
            model_json_string, custom_objects={"tf": tf, "K": K}
        )

        # モデル重みの読み込み
        model_weight_path = (
            self.model_dir + "/weight/weight_" + name + "_for0-10_reshape.h5"
        )
        self.model.load_weights(model_weight_path)


# 実際に学習に使うクラス
# 引数にCNNクラス、データ読み出しクラスをとる
class Trainer(object):
    def __init__(self, model_obj, datas_obj):
        self.datas = datas_obj
        self.model = model_obj

        self.name = self.datas.name

        # 個人学習データのディレクトリ
        # なかったら作る
        if os.path.exists(self.model.model_dir + "indiv_score/" + self.name):
            pass
        else:
            os.mkdir(self.model.model_dir + "indiv_score/" + self.name)

        # ダミーモードのときはepoch数1で学習を終了
        # 通常時はepoch数50
        # プログラム全体の動作を手早く見る用
        if self.datas.dummy_flag:
            self.epoch = 2
        else:
            self.epoch = 20

        # バッチサイズ
        self.batch_size = 32

        self.sequence_length = SEQUENCE_LENGTH  # グローバル変数からシーケンス長を取得

    # 一人のデータで学習
    def base_train(self):
        # コールバック関数のリスト(適宜追加)
        callbacks = []

        # 個人データを読み取る
        personaldatas_df = self.datas.personal_dataload()

        # データを並べ替え(X,Yの相関は保ったまま)
        # このモデルではシーケンス長が1のため、単純なシャッフルを行う
        data_length = len(personaldatas_df)

        # シーケンス開始インデックスではなく、データ全体のインデックスを使用
        data_index_array = np.array(range(data_length), dtype="uint32")
        data_index_array = np.random.permutation(data_index_array)

        # 訓練データと評価データの分割
        # 何パーセントを評価に使うか
        train_rate = 0.85
        train_index_array, test_index_array = np.split(
            data_index_array, [int(len(data_index_array) * train_rate)]
        )

        # 学習データの総数 (サンプル数)
        train_data_length = len(train_index_array)

        # 学習用データ (単一画像としてロード)
        X_train, Y_train = self.data_indexread(personaldatas_df, train_index_array)

        # 評価用データ (単一画像としてロード)
        X_test, Y_test = self.data_indexread(personaldatas_df, test_index_array)

        # 正規化
        X_train, Y_train = self.datas.data_normalize(X_train, Y_train)
        X_test, Y_test = self.datas.data_normalize(X_test, Y_test)

        # 学習する (入力X_trainの形状は (サンプル数, 155, 140, 1))
        history = self.model.model.fit(
            X_train,
            [Y_train[:, 0], Y_train[:, 1], Y_train[:, 2]],
            epochs=self.epoch,
            validation_data=(X_test, [Y_test[:, 0], Y_test[:, 1], Y_test[:, 2]]),
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # 学習結果を保存
        if self.datas.dummy_flag:
            weight_path = (
                self.model.model_dir
                + "weight/weight_"
                + self.name
                + "_for0-10"
                + "_reshape_dum.h5"
            )
        else:
            weight_path = (
                self.model.model_dir
                + "weight/weight_"
                + self.name
                + "_for0-10"
                + "_reshape.h5"
            )

        # 重み保存ディレクトリがなければ作る
        if os.path.exists(self.model.model_dir + "weight"):
            pass
        else:
            os.mkdir(self.model.model_dir + "weight")

        # 学習した重みをh5形式で保存
        self.model.model.save_weights(weight_path)

        # lossの経過を保存
        if self.datas.dummy_flag:
            log_csvpath = (
                self.model.model_dir
                + "indiv_score/"
                + self.name
                + "/learning_log_reshape_dum.csv"
            )
        else:
            log_csvpath = (
                self.model.model_dir
                + "indiv_score/"
                + self.name
                + "/learning_log_reshape.csv"
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
        # ==============================ここまで学習=====================================================

        # =============================ここから学習データでの評価=================================================
        self.evaluate_save(personaldatas_df, train_index_array, keyword="train")

        # =============================ここから評価データでの評価==================================================
        self.evaluate_save(personaldatas_df, test_index_array, keyword="val")

    # データの読み込み
    # index_arrayは単一画像のインデックスを格納
    def data_indexread(self, datas_df, index_array):

        # Xを格納するリスト (単一画像)
        X_list = []
        # Yを格納するリスト (単一画像に対応する力)
        Y_list = []

        print("\r", "now image loading", end="")

        # 各インデックスに対して単一の画像をロード
        for index in index_array:

            # 画像パスを読み込み
            X_path = datas_df.iloc[index, 0]

            # 画像そのものを読み出し
            img = cv2.imread(X_path, 0)

            # (155, 140, 1) に次元を調整 (チャンネル次元を追加)
            img_array = img.reshape(155, 140, 1)

            X_list.append(img_array)

            # Yは単一フレームに対応する力を使用
            Y_array = datas_df.iloc[index, [1, 2, 3]].values
            Y_list.append(Y_array)

        # np.arrayに変換
        X_array = np.array(X_list)
        # 形状: (サンプル数, 155, 140, 1)

        # Y_listをnp.arrayに変換
        Y_array = np.array(Y_list)
        # 形状: (サンプル数, 3)

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
        split_num = 80000
        # データの総数
        data_length = len(eval_index_array)
        index_split_num = int(data_length / split_num) + 1

        # 分割データ数がデータ総数より多いなら一括処理
        if split_num > data_length:
            # データを読み出し
            X, Y_true = self.data_indexread(datas_df, eval_index_array)

            # 正規化
            X, Y_true_norm = self.datas.data_normalize(
                X, Y_true.copy()
            )  # Y_trueを破壊しないようコピーを渡す
            # 評価する
            Y_predict_list = self.model.model.predict(
                X, batch_size=self.batch_size, verbose=1
            )
            Y_predict_norm = np.concatenate(
                [Y_predict_list[0], Y_predict_list[1], Y_predict_list[2]], axis=1
            )

            # 正規化状態から戻す (予測値と真値をそれぞれ戻す)
            Y_predict = self.datas.data_unnormalize(Y_predict_norm.copy())
            # Y_trueはdata_indexreadで読み込んだオリジナルをそのまま使用

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
        # 分割データ数がデータ総数より少ないなら分割処理
        # そうしないとGPUのメモリが足りないので
        else:
            index_array_split = np.array_split(eval_index_array, index_split_num)

            # 分割数分くりかえす
            read_length = len(index_array_split)
            for i in range(read_length):
                now_index_array = index_array_split[i]

                # データを読み出し
                X, Y_true = self.data_indexread(datas_df, now_index_array)

                # 正規化
                X, Y_true_norm = self.datas.data_normalize(X, Y_true.copy())
                # 評価する
                Y_predict_list = self.model.model.predict(
                    X, batch_size=self.batch_size, verbose=1
                )
                Y_predict_norm = np.concatenate(
                    [Y_predict_list[0], Y_predict_list[1], Y_predict_list[2]], axis=1
                )

                # 正規化状態から戻す
                Y_predict = self.datas.data_unnormalize(Y_predict_norm.copy())
                # Y_trueはdata_indexreadで読み込んだオリジナルをそのまま使用

                Fz_pred = Y_predict[:, 0]
                Fx_pred = Y_predict[:, 1]
                Fy_pred = Y_predict[:, 2]
                Fz_true = Y_true[:, 0]
                Fx_true = Y_true[:, 1]
                Fy_true = Y_true[:, 2]
                Fz_error = Fz_pred - Fz_true
                Fx_error = Fx_pred - Fx_true
                Fy_error = Fy_pred - Fy_true

                # 評価データをDataFrameに格納
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
                        ]
                    )
                    # DataFrameの連結は行に対して行う
                    eval_log_df = pd.concat(
                        [eval_log_df.transpose(), concat_df.transpose()]
                    ).transpose()

        # 保存パスにLSTMモデルであることを示す文字列を追加
        if self.datas.dummy_flag:
            eval_log_path = (
                indivisual_dir + "evaluate_" + keyword + "_for0-10_reshape_dum.csv"
            )
        else:
            eval_log_path = (
                indivisual_dir + "evaluate_" + keyword + "_for0-10_reshape.csv"
            )

        eval_log_df = eval_log_df.transpose()

        # DataFrameをcsv形式で保存
        eval_log_df.to_csv(eval_log_path, encoding="shift-jis")


if __name__ == "__main__":
    # Trueのとき少量のデータで学習
    #
    dummy_flag = False
    test_size = 0.15  # 用意したデータの何割を評価用に使うか
    split_seed = 111  # データランダム分割のシード値

    batch_size = 32

    # 学習に使う被験者名のリスト
    if dummy_flag:
        namelist = ["ryusetsu2"]
        epoch = 2
    else:
        # namelist=["watanabe","hamada","hanai","isogai","komura","mochiduki","sho","takeshige","tsuji","chen"]
        namelist = ["ryusetsu2"]
        epoch = 20

    # ディレクトリ移動(共通の処理)
    directry_initialize()

    # 一気に学習
    for now_name in namelist:
        # モデル構築クラスの用意
        CNN = multitask_CNN()

        # データ読み出しクラスの用意
        database = data_loader(name=now_name, Fz_range=10.0, dummy_flag=dummy_flag)

        # 学習用クラスの用意
        trainer = Trainer(CNN, database)

        # 学習の実施
        trainer.base_train()

        # 次の学習に備えてオブジェクトの消去
        del CNN, database, trainer
        gc.collect()  # ガベージコレクションを強制実行し、メモリを解放
