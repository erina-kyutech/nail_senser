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
from keras.layers import Input, Dense, Dropout, Lambda

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


# ディレクトリの移動
# データセットのあるフォルダや結果を保存するフォルダへアクセスするために
# ソースコードのあるフォルダから1つ上のディレクトリへ移動
def directry_initialize():
    nowdir = os.path.dirname(__file__)  # プログラムのあるディレクトリを参照
    os.chdir(nowdir)  # 作業ディレクトリをプログラムのあるディレクトリに
    os.chdir("..")  # 一つ上のディレクトリに移動

def build_input_image(bgr_img, mode="rgb"):
    """
    bgr_img: cv2.imread() で読んだBGR画像 (H,W,3)
    mode:
      "rgb" : RGBのまま使う
      "g"   : Gチャンネルだけを3ch化して使う
      "hs"  : HSVのH,SをR,Gに入れてB=0にする（BGR順に戻す）
    return: (H,W,3) uint8
    """
    if mode == "rgb":

        # BGR -> RGB にして学習したいならここで変換
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return rgb

    elif mode == "g":

        g = bgr_img[:, :, 1]
        g3 = cv2.merge([g, g, g])
        return g3

    elif mode == "hs":

        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]  # 0-179
        S = hsv[:, :, 1]  # 0-255
        Z = np.zeros_like(H, dtype=np.uint8)

        # 「RにH, GにS, Bは0」で学習したい、というルールを
        # OpenCV配列の並び(B,G,R)に合わせて B=0, G=S, R=H にする
        bgr_like = cv2.merge([Z, S, H])  # (B,G,R)
        return bgr_like

    else:
        raise ValueError(f"Unknown mode: {mode}")

# データを読み出すためのクラス
class data_loader(object):

    def __init__(self, name=None, Fz_range=10.0, dummy_flag=False, img_mode = "rgb",um_workers=8):
        self.dummy_flag = dummy_flag
        self.name = name
        self.img_mode = img_mode  # ← 追加: "rgb" / "g" / "hs"
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

                #Fzが大きすぎたら無視する(100N以上で無視）
                csv_record = csv_record[pd.to_numeric(csv_record["Fz"], errors="coerce") <= 100].copy()

                img_path = csv_record["path"]
                Fz = csv_record["Fz"]
                Fx = csv_record["Fr"]
                Fy = csv_record["Ff"]

                append_df = pd.concat([img_path, Fz, Fx, Fy], axis=1)
                append_df.columns = ["img_path", "Fz", "Fx", "Fy"]
                alldatas_df = pd.concat([alldatas_df, append_df])

        return alldatas_df

    # データの正規化
    def data_normalize(self, X, Y):
        X = X.astype("float32")
        X = X.reshape((-1, 150, 290, 3))  # ← 3ch
        X /= 255.0

        # Yは今まで通り
        Y[:, 0] /= self.normal_force_normalize
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


# # 自作層(グレースケールをカラー画像にする)関数
# def tensor_gray2BGR(grayX):
#     blank = tf.zeros_like(grayX)
#     BGR_X = tf.concat([blank, grayX], axis=3)
#     BGR_X = tf.concat([BGR_X, blank], axis=3)
#
#     return BGR_X


class multitask_CNN(object):
    def __init__(self, img_mode="g"):
        self.img_mode = img_mode

        # ★ concat用に保存先を分ける（ここが超重要）
        self.model_dir = f"./result/CNN_result/vgg16_{img_mode}_concat_150x290/"

        # ★ 親も子もまとめて作る（存在しててもOK）
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "weight"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "indiv_score"), exist_ok=True)

        self.CNN_build()


    # モデルの構築
    def CNN_build(self):
        l2_alpha = 0.001  # L2正則化の係数
        he_normal_seed = 111  # 乱数シード

        # 隠れ層ノード数は(入力ノード数+出力ノード数)*2/3が目安らしい
        middle_class_recurrence = 342  # 全結合中間層のノード数

        last_activation = "linear"  # 最終層の活性化関数

        # 最適化手法
        optimizer = Adam(lr=1e-04, decay=1e-6, beta_1=0.9, beta_2=0.999)

        # モデルの保存するパス
        model_path = self.model_dir + "for0-10.json"
        model_fig_path = self.model_dir + "for0-10.png"

        # ここからモデルの構築
        # 構築方法はFunctionAPI
        # 入力画像(画像縦サイズ、横サイズ、チャンネル数を指定)
        # 縦150pixel,横290pixelのグレースケール画像(チャンネル数1)を入力
        # VGG16の構造を読み込む(注:入力するのがカラー画像でないと学習済み重み値を利用できない)
        # VGG16で抽出した特徴量をFlattenで1次元化
        # グレースケール画像を赤、青要素0のカラー画像(チャンネル数3)に変換

        # VGG16を呼び出し(全結合層は含まない)
        input_tensor = Input(shape=(150, 290, 3), name="input_tensor")

        conv = VGG16(weights="imagenet",
                     input_shape=(150, 290, 3),
                     include_top=False)(input_tensor)

        # 畳み込み層の出力を1次元化
        flatten = GlobalMaxPooling2D(name="flatten")(conv)

        ##ここから全結合層を構築
        # Multitask-CNNなのでタスクごとに全結合層を作る
        # それぞれ活性化関数と正則化を指定している。
        # 過学習防止にDropout層をつけている

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

        # 最終出力のリストを作成
        predicts = [Fz, Fx, Fy]

        # モデルを構成
        self.model = Model(input_tensor, predicts)

        # モデルの構成を表示(主にデバッグ用)
        # self.model.summary()

        # モデルのコンパイル
        # 各出力層毎に損失関数を辞書形式で設定
        # 最適化手法もここで設定する
        self.model.compile(loss={"Fz": "mean_squared_error",
                                 "Fx": "mean_squared_error",
                                 "Fy": "mean_squared_error"},
                           optimizer=optimizer)
        print("MODEL_DIR:", self.model_dir)
        print("MODEL_INPUT:", self.model.input_shape)

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
        model_json_path = self.model_dir + "for0-10.json"
        model_json_string = open(model_json_path).read()

        # モデル構造の読み込み
        self.model = model_from_json(model_json_string,
                                     custom_objects={'tf': tf, 'K': K})

        # モデル重みの読み込み
        model_weight_path = self.model_dir + "/weight/weight_" + name + "_for0-10.h5"
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
        # 個人学習データのディレクトリ
        os.makedirs(os.path.join(self.model.model_dir, "indiv_score", self.name), exist_ok=True)

        # ダミーモードのときはepoch数1で学習を終了
        # 通常時はepoch数50
        # プログラム全体の動作を手早く見る用
        if self.datas.dummy_flag:
            self.epoch = 2
        else:
            self.epoch = 10

        # バッチサイズ
        self.batch_size = 32

    # 一人のデータで学習
    def base_train(self):
        # コールバック関数のリスト(適宜追加)
        callbacks = []

        # 個人データを読み取る
        personaldatas_df = self.datas.personal_dataload()

        # データを並べ替え(X,Yの相関は保ったまま)
        data_length = len(personaldatas_df)
        index_array = np.array(range(data_length), dtype="uint32")
        index_array = np.random.permutation(index_array)

        # 何パーセントを評価に使うか
        train_rate = 0.85
        train_index_array, test_index_array = np.split(index_array,
                                                       [int(len(index_array) * train_rate)])

        # 学習データの総数
        train_data_length = len(train_index_array)

        # 学習用データ
        X_train, Y_train = self.data_indexread(personaldatas_df, train_index_array)

        # 評価用データ
        X_test, Y_test = self.data_indexread(personaldatas_df, test_index_array)

        # 正規化
        X_train, Y_train = self.datas.data_normalize(X_train, Y_train)
        X_test, Y_test = self.datas.data_normalize(X_test, Y_test)

        # 学習する
        history = self.model.model.fit(X_train,
                                       [Y_train[:, 0],
                                        Y_train[:, 1],
                                        Y_train[:, 2]],
                                       epochs=self.epoch,
                                       validation_data=(X_test,
                                                        [Y_test[:, 0],
                                                         Y_test[:, 1],
                                                         Y_test[:, 2]]),
                                       batch_size=self.batch_size,
                                       callbacks=callbacks,
                                       verbose=1)

        # 学習結果を保存
        # 重みのパスを指定を保存
        if self.datas.dummy_flag:
            weight_path = (self.model.model_dir +
                           "weight/weight_" +
                           self.name +
                           "_for0-10" + "_dum.h5")
        else:
            weight_path = (self.model.model_dir +
                           "weight/weight_" +
                           self.name +
                           "_for0-10" +
                           ".h5")

        # 重み保存ディレクトリがなければ作る
        if os.path.exists(self.model.model_dir + "weight"):
            pass
        else:
            os.mkdir(self.model.model_dir + "weight")

        # 学習した重みをh5形式で保存
        self.model.model.save_weights(weight_path)

        # lossの経過を保存
        if self.datas.dummy_flag:
            log_csvpath = (self.model.model_dir +
                           "indiv_score/" +
                           self.name +
                           "/learning_log_dum.csv")
        else:
            log_csvpath = (self.model.model_dir +
                           "indiv_score/" +
                           self.name +
                           "/learning_log.csv")

        log_df = pd.DataFrame(history.history,
                              columns=["Fx_loss",
                                       "Fy_loss",
                                       "Fz_loss",
                                       "loss",
                                       "val_Fx_loss",
                                       "val_Fy_loss",
                                       "val_Fz_loss",
                                       "val_loss"])
        log_df.to_csv(log_csvpath)
        # ==============================ここまで学習=====================================================

        # =============================ここから学習データでの評価=================================================
        self.evaluate_save(personaldatas_df, train_index_array, keyword="train")

        # =============================ここから評価データでの評価==================================================
        self.evaluate_save(personaldatas_df, test_index_array, keyword="val")

    # データの読み込み
    # mapを使うことでfor文より早く読み出せる
    # Xが入力画像
    # Yがそれに対応する指先力
    def data_indexread(self, datas_df, index_array):
        def path2img(path):
            print("\r", "now image loading", end="")
            bgr = cv2.imread(path)  # ← カラーで読む（BGR）
            if bgr is None:
                raise FileNotFoundError(f"image not found: {path}")
            x = build_input_image(bgr, mode=self.datas.img_mode)  # ← mode反映
            return x

        index_list = list(index_array)
        X_path_list = list(datas_df.iloc[index_list, 0])
        X_img_list = list(map(path2img, X_path_list))

        #デバックの保存
        if not hasattr(self, "_debug_saved"):
            self._debug_saved = True

            dbg_dir = f"./debug_input/{self.datas.name}/{self.datas.img_mode}"
            os.makedirs(dbg_dir, exist_ok=True)

            #先頭から16枚保存
            for i, img in enumerate(X_img_list[:16]):
                out = img.copy() #imgは（H,W,3）unit８を想定

                #OpenCVで保存するのでBGRに直して保存（見た目の確認用）
                #rgbモード：ない分はRGBにしてるならBGRに戻す
                if self.datas.img_mode == "rgb":
                    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

                #gモード：3ch同値ならそのままでもOK（見やすさ重視）
                elif self.datas.img_mode == "g":
                    out_bgr = out

                #hsモード：そのままBGR保存
                elif self.datas.img_mode == "hs":
                    out_bgr = out

                else:
                    out_bgr = out

                cv2.imwrite(os.path.join(dbg_dir, f"sample_{i:02d}.png"), out_bgr)

            print(f"[DEBUG] saved samples -> {dbg_dir}")





        X_array = np.array(X_img_list)  # (N,150,290,3)
        Y_array = datas_df.iloc[index_list, [1, 2, 3]].values
        return X_array, Y_array

    # 推定値と真値との差を記録する関数、keywordで名前を設定
    def evaluate_save(self, personaldatas_df, index_array, keyword="train", chunk_size=1024, pred_batch_size=128):
        """
        index_array を chunk_size ごとに分割して
        画像ロード→正規化→predict を回し、結果を最後にCSVへ保存する（メモリ爆発しない）
        """
        import os
        import numpy as np
        import pandas as pd

        # 保存先（元コードの model_dir / indiv_score_dir の作り方に合わせて調整してね）
        # たぶんこういう構造になってるはず： ./result/CNN_result/vgg16_.../indiv_score/<name>/
        save_dir = os.path.join(self.model_dir, "indiv_score", self.name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"evaluate_{keyword}_for0-10.csv")

        # 結果をためる箱
        rows = []

        N = len(index_array)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_idx = index_array[start:end]

            # --- ここが一番重要：このchunk分だけ画像ロードする ---
            # 既存コードで「指定したindexだけロード」する関数/処理があるはず。
            # 例：X, Y = self.datas.image_loading(personaldatas_df, batch_idx, img_mode=self.img_mode, ...)
            # ↓↓↓ ここの1行を「元の evaluate_save がやってる image_loading の呼び方」に合わせてコピペしてね
            X, Y = self.datas.image_loading(personaldatas_df, batch_idx)
            # ↑↑↑ 合わない場合は、この行だけ元の呼び方に合わせる

            # 正規化（ここでX全体が巨大にならない）
            X, Y = self.datas.data_normalize(X, Y)

            # 予測（ここはGPUに優しいようにバッチ指定）
            pred = self.model.predict(X, batch_size=pred_batch_size, verbose=0)

            # --- pred の形を元コードに合わせて整形 ---
            # あなたのモデルは [Fz, Fx, Fy] の3出力のはず（Multi-output）
            # Kerasのmulti-outputだと pred は [pred_Fz, pred_Fx, pred_Fy] のリストで来ることが多い
            if isinstance(pred, (list, tuple)) and len(pred) == 3:
                pred_Fz, pred_Fx, pred_Fy = pred
            else:
                # もし pred が (N,3) ならこう
                pred_Fz, pred_Fx, pred_Fy = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

            # 正解Y側も同様に取り出し（ここも元コードのY形式に合わせる）
            # Y が dict や list の場合があるので吸収
            if isinstance(Y, dict):
                true_Fz = np.asarray(Y.get("Fz", Y.get("fz", None)))
                true_Fx = np.asarray(Y.get("Fx", Y.get("fx", None)))
                true_Fy = np.asarray(Y.get("Fy", Y.get("fy", None)))
            elif isinstance(Y, (list, tuple)) and len(Y) == 3:
                true_Fz, true_Fx, true_Fy = map(np.asarray, Y)
            else:
                # (N,3) を想定
                true_Fz, true_Fx, true_Fy = Y[:, 0], Y[:, 1], Y[:, 2]

            # 1次元に揃える
            pred_Fz = np.squeeze(pred_Fz)
            pred_Fx = np.squeeze(pred_Fx)
            pred_Fy = np.squeeze(pred_Fy)
            true_Fz = np.squeeze(true_Fz)
            true_Fx = np.squeeze(true_Fx)
            true_Fy = np.squeeze(true_Fy)

            # 行を追加（元の evaluate_*.csv の列名に合わせている）
            for i in range(len(batch_idx)):
                rows.append({
                    "Fz_predict": float(pred_Fz[i]),
                    "Fx_predict": float(pred_Fx[i]),
                    "Fy_predict": float(pred_Fy[i]),
                    "Fz_true": float(true_Fz[i]),
                    "Fx_true": float(true_Fx[i]),
                    "Fy_true": float(true_Fy[i]),
                    "Fz_error": float(pred_Fz[i] - true_Fz[i]),
                    "Fx_error": float(pred_Fx[i] - true_Fx[i]),
                    "Fy_error": float(pred_Fy[i] - true_Fy[i]),
                })

            print(f"[{keyword}] evaluated {end}/{N} samples")

            # ここでメモリ解放（地味に効く）
            del X, Y, pred, pred_Fz, pred_Fx, pred_Fy

        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print("saved:", save_path)


if __name__ == "__main__":
    # Trueのとき少量のデータで学習
    #
    dummy_flag = False
    test_size = 0.15  # 用意したデータの何割を評価用に使うか
    split_seed = 111  # データランダム分割のシード値
    batch_size = 32

    modes = ["rgb", "g", "hs"]
    # 学習に使う被験者名のリスト
    if dummy_flag:
        namelist = ["ifuku"]
        epoch = 2
    else:
        # namelist=["watanabe","hamada","hanai","isogai","komura","mochiduki","sho","takeshige","tsuji","chen"]
        namelist = ["ifuku"]
        epoch = 10

    # ディレクトリ移動(共通の処理)
    directry_initialize()

    # 一気に学習
    for now_name in namelist:
        for mode in modes:
            print(f"\n===== TRAIN: name={now_name}, mode={mode} =====")

            CNN = multitask_CNN(img_mode=mode) # モデル構築クラスの用意

            # データ読み出しクラスの用意
            database = data_loader(
                name=now_name,
                Fz_range = 10.0,
                dummy_flag = dummy_flag,
                img_mode = mode
            )

            # 学習用クラスの用意
            trainer = Trainer(CNN, database)

            # 学習の実施
            trainer.base_train()

            # 次の学習に備えてオブジェクトの消去
            del CNN, database, trainer

