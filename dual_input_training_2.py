# -*- coding: utf-8 -*-
"""
dual_input_training.py

2入力VGG16モデル（指先ブランチ + 爪ブランチ）による3軸力推定。
concat画像(150×290)を左右に分割して2つのVGG16ブランチに入力し、
特徴量を結合してFz, Fx, Fyを推定する。

concat画像の構成:
  左側 0:150  → 指先画像 (fingertip)  shape=(150, 150, 3)
  右側 150:290 → 爪画像   (nail)       shape=(150, 140, 3)
"""

from __future__ import print_function
import os
import gc

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K, regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D, Concatenate
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model


# =========================================================
# 設定
# =========================================================
SUBJECT_NAME = "ifuku"
DUMMY_FLAG = False

IMG_H = 150
FINGERTIP_W = 150   # concat画像の左側
NAIL_W = 140        # concat画像の右側
CONCAT_W = FINGERTIP_W + NAIL_W  # 290

NORMAL_FORCE_NORMALIZE = 10.0
SHEAR_FORCE_NORMALIZE = 5.0

MODEL_DIR = "./result/CNN_result/vgg16_dual_input_150x290/"

# ifuku番号による学習/テスト分割
TRAINVAL_START = 1
TRAINVAL_END   = 166
TEST_START     = 167
TEST_END       = 180
VAL_RATIO      = 0.15  # trainval内でのval割合


# =========================================================
# ディレクトリ初期化
# =========================================================
def directry_initialize():
    nowdir = os.path.dirname(__file__)
    os.chdir(nowdir)
    os.chdir("..")


# =========================================================
# concat画像を2つに分割して返す
# =========================================================
def split_concat_image(concat_bgr):
    """
    concat画像(150×290, BGR)を指先画像と爪画像に分割する。
    左側 0:150  → 指先画像 (fingertip)
    右側 150:290 → 爪画像   (nail)
    """
    fingertip_bgr = concat_bgr[:, 0:FINGERTIP_W, :]
    nail_bgr = concat_bgr[:, FINGERTIP_W:FINGERTIP_W + NAIL_W, :]
    return fingertip_bgr, nail_bgr


def build_input_images(concat_bgr):
    """
    concat BGR画像から指先・爪のRGB画像を返す。
    """
    fingertip_bgr, nail_bgr = split_concat_image(concat_bgr)
    fingertip_rgb = cv2.cvtColor(fingertip_bgr, cv2.COLOR_BGR2RGB)
    nail_rgb = cv2.cvtColor(nail_bgr, cv2.COLOR_BGR2RGB)
    return fingertip_rgb, nail_rgb


# =========================================================
# データ読み込みクラス
# =========================================================
class data_loader(object):
    def __init__(self, name=None, Fz_range=10.0, dummy_flag=False):
        self.dummy_flag = dummy_flag
        self.name = name

        if Fz_range == 10.0:
            self.normal_force_normalize = NORMAL_FORCE_NORMALIZE
            self.shear_force_normalize = SHEAR_FORCE_NORMALIZE
        else:
            raise ValueError(f"Unsupported Fz_range: {Fz_range}")

    @staticmethod
    def parse_ifuku_id(path_str):
        """パスからifuku番号を抽出する。例: .../ifuku12/... -> 12"""
        import re
        m = re.search(r'ifuku(\d+)', str(path_str))
        return int(m.group(1)) if m else None

    def personal_dataload(self):
        if self.dummy_flag:
            namelist_path = "./datas/record0-10xyz/namelist_dum.csv"
        else:
            namelist_path = "./datas/record0-10xyz/namelist.csv"

        names = pd.read_csv(namelist_path, header=None)
        alldatas_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy"])

        for _, names_item in names.iterrows():
            now_namedir = "./datas/" + names_item[0]

            if self.name in now_namedir:
                csv_record = pd.read_csv(now_namedir, header=0)
                csv_record.columns = ["path", "Fz", "Fr", "Ff"]
                csv_record = csv_record[
                    pd.to_numeric(csv_record["Fz"], errors="coerce") <= 100
                ].copy()

                append_df = pd.concat(
                    [csv_record["path"], csv_record["Fz"],
                     csv_record["Fr"], csv_record["Ff"]],
                    axis=1
                )
                append_df.columns = ["img_path", "Fz", "Fx", "Fy"]
                alldatas_df = pd.concat([alldatas_df, append_df], ignore_index=True)

        # ifuku番号を付与
        alldatas_df["ifuku_id"] = alldatas_df["img_path"].apply(
            self.parse_ifuku_id
        )
        alldatas_df = alldatas_df.dropna(subset=["ifuku_id"]).copy()
        alldatas_df["ifuku_id"] = alldatas_df["ifuku_id"].astype(int)

        return alldatas_df

    def data_normalize(self, X_fingertip, X_nail, Y):
        """
        X_fingertip, X_nail: float32, 0-255
        Y: (N, 3) float32
        """
        X_fingertip = X_fingertip.astype("float32") / 255.0
        X_nail = X_nail.astype("float32") / 255.0

        Y = Y.astype("float32").copy()
        Y[:, 0] /= self.normal_force_normalize
        Y[:, 1] += self.shear_force_normalize
        Y[:, 2] += self.shear_force_normalize
        Y[:, 1] /= (self.shear_force_normalize * 2)
        Y[:, 2] /= (self.shear_force_normalize * 2)
        return X_fingertip, X_nail, Y

    def data_unnormalize(self, Y):
        Y = Y.astype("float32").copy()
        Y[:, 0] *= self.normal_force_normalize
        Y[:, 1] *= (self.shear_force_normalize * 2)
        Y[:, 2] *= (self.shear_force_normalize * 2)
        Y[:, 1] -= self.shear_force_normalize
        Y[:, 2] -= self.shear_force_normalize
        return Y


# =========================================================
# 2入力モデル
# =========================================================
class DualInputCNN(object):
    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.join(MODEL_DIR, "weight"), exist_ok=True)
        os.makedirs(os.path.join(MODEL_DIR, "indiv_score"), exist_ok=True)
        self.model_dir = MODEL_DIR
        self.build_model()

    def build_model(self):
        l2_alpha = 0.001
        middle_units = 342
        optimizer = Adam(learning_rate=1e-4)

        # ── 入力 ──────────────────────────────────────────
        fingertip_input = Input(
            shape=(IMG_H, FINGERTIP_W, 3),
            name="fingertip_input"
        )
        nail_input = Input(
            shape=(IMG_H, NAIL_W, 3),
            name="nail_input"
        )

        # ── 指先ブランチ（VGG16） ─────────────────────────
        vgg_fingertip_model = VGG16(
            weights="imagenet",
            input_shape=(IMG_H, FINGERTIP_W, 3),
            include_top=False
        )
        vgg_fingertip_model._name = "vgg16_fingertip"
        vgg_fingertip = vgg_fingertip_model(fingertip_input)
        feat_fingertip = GlobalMaxPooling2D(
            name="pool_fingertip"
        )(vgg_fingertip)

        # ── 爪ブランチ（VGG16） ───────────────────────────
        vgg_nail_model = VGG16(
            weights="imagenet",
            input_shape=(IMG_H, NAIL_W, 3),
            include_top=False
        )
        vgg_nail_model._name = "vgg16_nail"
        vgg_nail = vgg_nail_model(nail_input)
        feat_nail = GlobalMaxPooling2D(
            name="pool_nail"
        )(vgg_nail)

        # ── 特徴量結合 ────────────────────────────────────
        merged = Concatenate(name="concat_features")(
            [feat_fingertip, feat_nail]
        )  # shape = (1024,)

        # ── 回帰ヘッド（Fz） ──────────────────────────────
        fz = Dense(middle_units, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_alpha))(merged)
        fz = Dropout(0.2)(fz)
        fz = Dense(1, activation="linear", name="Fz",
                   kernel_regularizer=regularizers.l2(l2_alpha))(fz)

        # ── 回帰ヘッド（Fx） ──────────────────────────────
        fx = Dense(middle_units, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_alpha))(merged)
        fx = Dropout(0.2)(fx)
        fx = Dense(1, activation="linear", name="Fx",
                   kernel_regularizer=regularizers.l2(l2_alpha))(fx)

        # ── 回帰ヘッド（Fy） ──────────────────────────────
        fy = Dense(middle_units, activation="relu",
                   kernel_regularizer=regularizers.l2(l2_alpha))(merged)
        fy = Dropout(0.2)(fy)
        fy = Dense(1, activation="linear", name="Fy",
                   kernel_regularizer=regularizers.l2(l2_alpha))(fy)

        # ── モデル組み立て ────────────────────────────────
        self.model = Model(
            inputs=[fingertip_input, nail_input],
            outputs=[fz, fx, fy]
        )
        self.model.compile(
            loss={
                "Fz": "mean_squared_error",
                "Fx": "mean_squared_error",
                "Fy": "mean_squared_error",
            },
            optimizer=optimizer
        )

        print("MODEL INPUT SHAPES:")
        for inp in self.model.inputs:
            print(" ", inp.name, inp.shape)

        # アーキテクチャ保存
        model_json_path = os.path.join(self.model_dir, "for0-10.json")
        if not os.path.exists(model_json_path):
            with open(model_json_path, "w", encoding="utf-8") as f:
                f.write(self.model.to_json())

        model_fig_path = os.path.join(self.model_dir, "for0-10.png")
        if not os.path.exists(model_fig_path):
            plot_model(self.model, show_shapes=True, to_file=model_fig_path)

    def model_load_weights(self, name):
        weight_path = os.path.join(
            self.model_dir, "weight", f"weight_{name}_for0-10.h5"
        )
        self.model.load_weights(weight_path)


# =========================================================
# Sequence（バッチごとに画像を読み込む）
# =========================================================
class DualInputSequence(tf.keras.utils.Sequence):
    """
    concat画像をバッチごとに読み込み、指先・爪に分割して返す。
    全画像を一括でRAMに乗せないのでメモリを節約できる。
    """
    def __init__(self, df, normal_force_normalize, shear_force_normalize,
                 batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.normal_force_normalize = normal_force_normalize
        self.shear_force_normalize = shear_force_normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_df = self.df.iloc[batch_idx]

        ft_list = []
        nail_list = []

        for path in batch_df["img_path"].tolist():
            concat_bgr = cv2.imread(path)
            if concat_bgr is None:
                raise FileNotFoundError(f"image not found: {path}")
            ft_rgb, nail_rgb = build_input_images(concat_bgr)
            ft_list.append(ft_rgb)
            nail_list.append(nail_rgb)

        X_ft   = np.array(ft_list,   dtype="float32") / 255.0
        X_nail = np.array(nail_list, dtype="float32") / 255.0

        Y = batch_df[["Fz", "Fx", "Fy"]].values.astype("float32")
        Y[:, 0] /= self.normal_force_normalize
        Y[:, 1] += self.shear_force_normalize
        Y[:, 2] += self.shear_force_normalize
        Y[:, 1] /= (self.shear_force_normalize * 2)
        Y[:, 2] /= (self.shear_force_normalize * 2)

        return [X_ft, X_nail], [Y[:, 0], Y[:, 1], Y[:, 2]]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================================================
# Trainer
# =========================================================
class Trainer(object):
    def __init__(self, model_obj, datas_obj):
        self.datas = datas_obj
        self.model = model_obj
        self.name = self.datas.name

        os.makedirs(
            os.path.join(self.model.model_dir, "indiv_score", self.name),
            exist_ok=True
        )

        self.epoch = 2 if self.datas.dummy_flag else 10
        self.batch_size = 32

    def base_train(self):
        personaldatas_df = self.datas.personal_dataload()

        # ── ifuku番号でtrain/val/testを分割 ──────────────
        trainval_df = personaldatas_df[
            (personaldatas_df["ifuku_id"] >= TRAINVAL_START) &
            (personaldatas_df["ifuku_id"] <= TRAINVAL_END)
        ].copy()
        test_df = personaldatas_df[
            (personaldatas_df["ifuku_id"] >= TEST_START) &
            (personaldatas_df["ifuku_id"] <= TEST_END)
        ].copy()

        # trainval内をifuku単位でtrain/valに分割
        import random as _random
        trainval_ids = list(range(TRAINVAL_START, TRAINVAL_END + 1))
        _random.Random(42).shuffle(trainval_ids)
        val_size  = max(1, int(len(trainval_ids) * VAL_RATIO))
        val_ids   = sorted(trainval_ids[:val_size])
        train_ids = sorted(trainval_ids[val_size:])

        train_df = trainval_df[
            trainval_df["ifuku_id"].isin(train_ids)
        ].reset_index(drop=True)
        val_df = trainval_df[
            trainval_df["ifuku_id"].isin(val_ids)
        ].reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"train ifuku: {train_ids[0]}〜{train_ids[-1]}  ({len(train_df)} samples)")
        print(f"val   ifuku: {val_ids}  ({len(val_df)} samples)")
        print(f"test  ifuku: {TEST_START}〜{TEST_END}  ({len(test_df)} samples)")

        # Sequenceを作成（画像はバッチごとに読み込まれる）
        train_seq = DualInputSequence(
            train_df,
            self.datas.normal_force_normalize,
            self.datas.shear_force_normalize,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_seq = DualInputSequence(
            val_df,
            self.datas.normal_force_normalize,
            self.datas.shear_force_normalize,
            batch_size=self.batch_size,
            shuffle=False
        )

        history = self.model.model.fit(
            train_seq,
            epochs=self.epoch,
            validation_data=val_seq,
            verbose=1
        )

        weight_name = (
            f"weight_{self.name}_for0-10_dum.h5"
            if self.datas.dummy_flag
            else f"weight_{self.name}_for0-10.h5"
        )
        weight_path = os.path.join(
            self.model.model_dir, "weight", weight_name
        )
        self.model.model.save_weights(weight_path)
        print(f"weights saved: {weight_path}")

        log_name = (
            "learning_log_dum.csv"
            if self.datas.dummy_flag
            else "learning_log.csv"
        )
        log_path = os.path.join(
            self.model.model_dir, "indiv_score", self.name, log_name
        )
        pd.DataFrame(history.history).to_csv(log_path, index=False)

        self.evaluate_save(test_df, keyword="test")

    def evaluate_save(self, df, keyword="val",
                      chunk_size=512, pred_batch_size=128):
        save_dir = os.path.join(
            self.model.model_dir, "indiv_score", self.name
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir, f"evaluate_{keyword}_for0-10.csv"
        )

        rows = []
        N = len(df)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_df = df.iloc[start:end]

            ft_list = []
            nail_list = []
            for path in batch_df["img_path"].tolist():
                concat_bgr = cv2.imread(path)
                if concat_bgr is None:
                    raise FileNotFoundError(f"image not found: {path}")
                ft_rgb, nail_rgb = build_input_images(concat_bgr)
                ft_list.append(ft_rgb)
                nail_list.append(nail_rgb)

            X_ft   = np.array(ft_list,   dtype="float32") / 255.0
            X_nail = np.array(nail_list, dtype="float32") / 255.0
            Y_true = batch_df[["Fz", "Fx", "Fy"]].values.astype("float32")

            pred = self.model.model.predict(
                [X_ft, X_nail],
                batch_size=pred_batch_size,
                verbose=0
            )
            pred_concat = np.concatenate(
                [pred[0], pred[1], pred[2]], axis=1
            )
            pred_concat = self.datas.data_unnormalize(pred_concat)

            for i in range(len(batch_df)):
                rows.append({
                    "Fz_predict": float(pred_concat[i, 0]),
                    "Fx_predict": float(pred_concat[i, 1]),
                    "Fy_predict": float(pred_concat[i, 2]),
                    "Fz_true":    float(Y_true[i, 0]),
                    "Fx_true":    float(Y_true[i, 1]),
                    "Fy_true":    float(Y_true[i, 2]),
                    "Fz_error":   float(pred_concat[i, 0] - Y_true[i, 0]),
                    "Fx_error":   float(pred_concat[i, 1] - Y_true[i, 1]),
                    "Fy_error":   float(pred_concat[i, 2] - Y_true[i, 2]),
                })

            print(f"[{keyword}] evaluated {end}/{N}")
            del X_ft, X_nail, pred, pred_concat
            gc.collect()

        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"saved: {save_path}")


# =========================================================
# メイン
# =========================================================
if __name__ == "__main__":
    directry_initialize()

    print(f"\n===== DUAL INPUT TRAIN: name={SUBJECT_NAME} =====")

    CNN = DualInputCNN()
    CNN.model.summary()

    database = data_loader(
        name=SUBJECT_NAME,
        Fz_range=10.0,
        dummy_flag=DUMMY_FLAG
    )
    trainer = Trainer(CNN, database)
    trainer.base_train()

    del CNN, database, trainer
    gc.collect()
