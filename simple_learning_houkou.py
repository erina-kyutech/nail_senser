# -*- coding: utf-8 -*-
from __future__ import print_function
import os, re, sys, glob
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import cv2
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, Lambda, GlobalMaxPooling2D
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam

# ----------------------------
# 設定
# ----------------------------
IMG_H, IMG_W = 155, 140
FZ_RANGE = 10.0       # 以前と同じ正規化レンジ
THEAR_RANGE = 5.0     # Fx,Fyの正規化レンジ（±5を0-1にマップ）
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_RATE = 0.85

# データルート（namelist.csv がある階層）
DATA_ROOT = "./datas/record0-10xyz"

# 出力先
RESULT_DIR = "./result/CNN_result/compare_G_RGB_HS"

# ----------------------------
# ユーティリティ
# ----------------------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def tensor_gray2BGR(x):
    # (N,H,W,1) -> (N,H,W,3)  [0, gray, 0]
    blank = tf.zeros_like(x)
    return tf.concat([blank, x, blank], axis=3)

def tensor_HS2HS0(x):
    # (N,H,W,2) -> (N,H,W,3)  [H,S,0]
    zeros = tf.zeros_like(x[..., :1])
    return tf.concat([x, zeros], axis=3)

def normalize_targets(Y):
    Y = Y.astype("float64").copy()
    # Fz: 0..FZ_RANGE -> 0..1
    Y[:, 0] /= FZ_RANGE
    # Fx,Fy: [-THEAR_RANGE .. +THEAR_RANGE] -> 0..1
    Y[:, 1] += THEAR_RANGE
    Y[:, 2] += THEAR_RANGE
    Y[:, 1] /= (2 * THEAR_RANGE)
    Y[:, 2] /= (2 * THEAR_RANGE)
    return Y

def unnormalize_targets(Y):
    Y = Y.copy()
    Y[:, 0] *= FZ_RANGE
    Y[:, 1] *= (2 * THEAR_RANGE)
    Y[:, 2] *= (2 * THEAR_RANGE)
    Y[:, 1] -= THEAR_RANGE
    Y[:, 2] -= THEAR_RANGE
    return Y

def split_train_val(N, rate=TRAIN_RATE, seed=111):
    rng = np.random.RandomState(seed)
    idx = np.arange(N, dtype=np.int64)
    rng.shuffle(idx)
    k = int(N * rate)
    return idx[:k], idx[k:]

# ----------------------------
# データ読み込み
#  - 同じフォルダに _G.png / _RGB.png / _HSV.png がある前提
#  - datalog.csv が
#     1) path列で *_G.png を指す  もしくは
#     2) id列（000001 など）を持つ
# ----------------------------
SUF_G   = "_G.png"
SUF_RGB = "_RGB.png"
SUF_HSV = "_HSV.png"

def infer_triplet_paths_from_row(row):
    """
    row に 'path' があればそこからIDを復元。
    - 例: .../000123_G.png -> base_id=000123
    row に 'id' があればそれを使う。
    戻り値: (path_G, path_RGB, path_HSV)
    """
    if "id" in row and isinstance(row["id"], str):
        base_dir = os.path.dirname(row.get("base_dir", row.get("path", "")))  # fallback
        # base_dir が空なら namelist のエントリから推測するのが安全だが、
        # 通常は datalog.csv と同じディレクトリに画像がいるはず
        if not base_dir:
            # datalog.csv 自体の場所を基準にする
            # （呼び出し側で渡すことも可だが簡略化）
            base_dir = os.getcwd()
        stem = row["id"]
        return (os.path.join(base_dir, f"{stem}{SUF_G}"),
                os.path.join(base_dir, f"{stem}{SUF_RGB}"),
                os.path.join(base_dir, f"{stem}{SUF_HSV}"))

    if "path" in row:
        p = row["path"]
        # *_G.png や *_RGB.png など何でもOK。末尾を正規化してID抽出
        m = re.search(r"(.*)(?:_G|_RGB|_HSV)\.png$", p)
        if m:
            stem_path = m.group(1)
            return (stem_path + SUF_G, stem_path + SUF_RGB, stem_path + SUF_HSV)
        else:
            # 旧式（拡張子なし番号のみ）の場合にも対応
            m2 = re.search(r"(.*)(?:\.png)?$", p)
            if m2:
                stem_path = m2.group(1)
                return (stem_path + SUF_G, stem_path + SUF_RGB, stem_path + SUF_HSV)

    raise ValueError("datalog.csv の行から画像パスを推測できませんでした")

def load_triplet_images(path_G, path_RGB, path_HSV):
    """
    3種類すべて読み込む:
      - G:        単チャネル(155,140) -> (155,140,1)
      - RGB:      (155,140,3) BGR読込 → RGBにしない（VGG16はImageNet前提だが今回は相対比較が主目的。揃えればOK）
      - HSV:      ファイルは3chで保存されている想定。ここでは H,S を使って (155,140,2)
    画像値は 0..255 のまま（正規化は後段）
    """
    # G
    imgG = cv2.imread(path_G, cv2.IMREAD_GRAYSCALE)
    if imgG is None:
        raise FileNotFoundError(path_G)
    if imgG.shape != (IMG_H, IMG_W):
        imgG = cv2.resize(imgG, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    imgG = imgG[..., None]  # (H,W,1)

    # RGB (BGRで入るが、学習は相対比較用なので統一運用でOK)
    imgRGB = cv2.imread(path_RGB, cv2.IMREAD_COLOR)
    if imgRGB is None:
        raise FileNotFoundError(path_RGB)
    if imgRGB.shape[:2] != (IMG_H, IMG_W):
        imgRGB = cv2.resize(imgRGB, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    # HSV（保存時にHSV空間のままpng化している前提。読み出しは「数値をそのまま」受け取る）
    imgHSV = cv2.imread(path_HSV, cv2.IMREAD_COLOR)
    if imgHSV is None:
        raise FileNotFoundError(path_HSV)
    if imgHSV.shape[:2] != (IMG_H, IMG_W):
        imgHSV = cv2.resize(imgHSV, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    # H,S を取り出し（保存された配列の ch0, ch1 が H,S である前提）
    HS = imgHSV[..., :2]  # (H,W,2)

    return imgG, imgRGB, HS

def build_dataset_from_namelist(name_filter=None):
    """
    ./datas/record0-10xyz/namelist.csv を辿って
    各 datalog.csv を集約し、3モード分のXとYを返す。
    name_filter: ["ifuku", ...] など、対象者名を含むサブディレクトリ名のフィルタ
    """
    namelist_path = os.path.join(DATA_ROOT, "namelist.csv")
    names = pd.read_csv(namelist_path, header=None)
    # columns: [relative_path_to_datalog.csv, degree]
    rows = []

    for _, item in names.iterrows():
        rel = item[0]  # e.g., "record0-10xyz/ifuku/360deg/datalog.csv"
        degree = int(item[1])
        datalog_path = os.path.join("./datas", rel)
        # name_filter がある場合、path に含まれるときのみ採用
        if name_filter and not any(n in datalog_path for n in name_filter):
            continue
        if not os.path.exists(datalog_path):
            continue
        df = pd.read_csv(datalog_path, header=None)
        # 想定: [path, Fz, Fr, Ff] または [id, Fz, Fr, Ff]
        if df.shape[1] == 4:
            df.columns = ["path_or_id", "Fz", "Fr", "Ff"]
        else:
            raise ValueError("datalog.csv のカラム数が想定外です: " + datalog_path)

        # path_or_id がフルパスなら "path" として扱い、番号だけなら "id" として扱う
        local_rows = []
        for _, r in df.iterrows():
            val = str(r["path_or_id"])
            row = {"Fz": float(r["Fz"]), "Fx": float(r["Fr"]), "Fy": float(r["Ff"])}
            if os.path.sep in val or val.endswith(".png"):
                row["path"] = val
            else:
                row["id"] = val
                row["base_dir"] = os.path.dirname(datalog_path)  # id→パス生成用
            local_rows.append(row)

        rows.extend(local_rows)

    # 実ファイル読み込み
    Xg, Xrgb, Xhs, Y = [], [], [], []
    for row in rows:
        pG, pRGB, pHSV = infer_triplet_paths_from_row(row)
        imgG, imgRGB, HS = load_triplet_images(pG, pRGB, pHSV)
        Xg.append(imgG)
        Xrgb.append(imgRGB)
        Xhs.append(HS)
        Y.append([row["Fz"], row["Fx"], row["Fy"]])

    Xg = np.stack(Xg).astype("float64")   # (N,H,W,1)
    Xrgb = np.stack(Xrgb).astype("float64")  # (N,H,W,3)
    Xhs = np.stack(Xhs).astype("float64") # (N,H,W,2)
    Y = np.array(Y, dtype="float64")      # (N,3)

    # 0..1 正規化（画像）
    Xg   /= 255.0
    Xrgb /= 255.0
    Xhs  /= 255.0

    # 教師の正規化
    Y = normalize_targets(Y)

    return Xg, Xrgb, Xhs, Y

# ----------------------------
# モデル（VGG16ベース、3出力回帰）
#  - G： (H,W,1) -> gray2BGR -> VGG16
#  - RGB：(H,W,3) -> VGG16
#  - HS：(H,W,2) -> HS0 3ch化 -> VGG16
# ----------------------------
def build_model(input_shape, mode):
    l2_alpha = 0.001
    middle = 342
    last_activation = "linear"
    optimizer = Adam(lr=1e-4, decay=1e-6, beta_1=0.9, beta_2=0.999)

    inp = Input(shape=input_shape, name=f"input_{mode}")

    if mode == "G":
        x = Lambda(tensor_gray2BGR, name="gray2BGR")(inp)          # -> (H,W,3)
    elif mode == "RGB":
        x = inp                                                     # そのまま3ch
    elif mode == "HS":
        x = Lambda(tensor_HS2HS0, name="HS2HS0")(inp)               # -> (H,W,3)
    else:
        raise ValueError("unknown mode")

    conv = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_H, IMG_W, 3))(x)
    feat = GlobalMaxPooling2D(name="flatten")(conv)

    def head(name):
        h = Dense(middle, activation="relu",
                  kernel_regularizer=regularizers.l2(l2_alpha))(feat)
        h = Dropout(0.2)(h)
        h = Dense(1, activation=last_activation, name=name,
                  kernel_regularizer=regularizers.l2(l2_alpha))(h)
        return h

    Fz = head("Fz")
    Fx = head("Fx")
    Fy = head("Fy")

    model = Model(inp, [Fz, Fx, Fy])
    model.compile(loss={"Fz":"mse", "Fx":"mse", "Fy":"mse"}, optimizer=optimizer)
    return model

def train_one_mode(X, Y, mode, outdir):
    ensure_dir(outdir)
    N = X.shape[0]
    idx_tr, idx_va = split_train_val(N, TRAIN_RATE, seed=111)
    Xtr, Ytr = X[idx_tr], Y[idx_tr]
    Xva, Yva = X[idx_va], Y[idx_va]

    model = build_model(X.shape[1:], mode)
    hist = model.fit(
        Xtr, [Ytr[:,0], Ytr[:,1], Ytr[:,2]],
        validation_data=(Xva, [Yva[:,0], Yva[:,1], Yva[:,2]]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1
    )

    # 重み保存
    model.save_weights(os.path.join(outdir, f"weight_{mode}.h5"))

    # ログ保存
    df_log = pd.DataFrame(hist.history,
        columns=[
            "Fx_loss","Fy_loss","Fz_loss","loss",
            "val_Fx_loss","val_Fy_loss","val_Fz_loss","val_loss"
        ])
    df_log.to_csv(os.path.join(outdir, f"learning_log_{mode}.csv"), index=False)

    return model

# ----------------------------
# メイン
# ----------------------------
def main():
    ensure_dir(RESULT_DIR)

    # 例：特定の被験者だけで学習したい時は name_filter=["ifuku"]
    Xg, Xrgb, Xhs, Y = build_dataset_from_namelist(name_filter=None)

    print("Dataset shapes:",
          "\n  G  :", Xg.shape,
          "\n  RGB:", Xrgb.shape,
          "\n  HS :", Xhs.shape,
          "\n  Y  :", Y.shape)

    # G
    out_G = os.path.join(RESULT_DIR, "vgg16_G")
    train_one_mode(Xg, Y, "G", out_G)

    # RGB
    out_RGB = os.path.join(RESULT_DIR, "vgg16_RGB")
    train_one_mode(Xrgb, Y, "RGB", out_RGB)

    # HS (H+S -> HS0で3ch化)
    out_HS = os.path.join(RESULT_DIR, "vgg16_HS")
    train_one_mode(Xhs, Y, "HS", out_HS)

if __name__ == "__main__":
    main()
