# -*- coding: utf-8 -*-
"""
train_material_classifier.py

素材分類器を学習するスクリプト。
VGG16ベースで4クラス（felt/acrylic/paper/aluminum）を分類する。

入力：マスク済み画像（nail_and_tip）150×290px
出力：各素材の確率（softmax、4クラス）

学習データ：ifuku1〜30の_dedup_masked/nail_and_tip画像
テストデータ：ifuku31〜35（評価には使わない、分類器はリアルタイム用）

保存先：
  result/CNN_result/material_classifier/
"""

from __future__ import annotations

import gc
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

# =========================================================
# 設定
# =========================================================
RANDOM_SEED = 42
IMG_H = 150
IMG_W = 290
IMG_C = 3
BATCH_SIZE = 32
EPOCHS     = 10
VAL_RATIO  = 0.15

MASK_PATTERN = "nail_and_tip"

# 学習に使うifuku番号範囲
TRAIN_IFUKU_START = 1
TRAIN_IFUKU_END   = 30

PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")
DATA_ROOT    = PROJECT_ROOT / "datas"
RESULT_ROOT  = PROJECT_ROOT / "result" / "CNN_result" / "material_classifier"

# 素材とラベルの対応
MATERIAL_DIRS = {
    "felt":     "felt_0-10xyz_dedup_masked",
    "acrylic":  "acrylic_0-10xyz_dedup_masked",
    "paper":    "paper_0-10xyz_dedup_masked",
    "aluminum": "aluminum_0-10xyz_dedup_masked",
}

LABEL_MAP = {
    "felt":     0,
    "acrylic":  1,
    "paper":    2,
    "aluminum": 3,
}
NUM_CLASSES = 4


# =========================================================
# ユーティリティ
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_ifuku_id(path_str):
    m = re.search(r'ifuku(\d+)', str(path_str))
    return int(m.group(1)) if m else None


def resolve_img_path(path_str: str) -> Path:
    """datalog.csvのパスを_dedup_masked/nail_and_tipのパスに変換"""
    p = Path(path_str)
    if not p.is_absolute():
        path_str2 = str(p).replace("./", "", 1).replace(".\\", "", 1)
        if path_str2.startswith("datas\\") or path_str2.startswith("datas/"):
            p = PROJECT_ROOT / path_str2
        else:
            p = DATA_ROOT / path_str2

    parts = list(p.parts)

    # 素材フォルダに_maskedを付ける
    for i, part in enumerate(parts):
        if "_0-10xyz" in part and "_masked" not in part:
            parts[i] = part + "_masked"
            break

    # 360degの直下にnail_and_tipを挿入
    try:
        deg_idx = parts.index("360deg")
        parts.insert(deg_idx + 1, MASK_PATTERN)
    except ValueError:
        pass

    return Path(*parts)


# =========================================================
# データ読み込み
# =========================================================
def load_material_data(mat_name: str, mat_dir_name: str,
                       ifuku_start: int, ifuku_end: int) -> pd.DataFrame:
    """1素材分のデータを読み込んでラベルを付ける"""
    namelist_path = DATA_ROOT / mat_dir_name / "namelist.csv"

    # フォールバック
    if not namelist_path.exists():
        fallback_dedup = Path(str(namelist_path).replace("_dedup_masked", "_dedup"))
        fallback_orig  = Path(str(namelist_path).replace("_dedup_masked", "").replace("_masked", ""))
        if fallback_dedup.exists():
            namelist_path = fallback_dedup
        elif fallback_orig.exists():
            namelist_path = fallback_orig
        else:
            print(f"  [WARN] namelist.csvなし: {mat_name}")
            return pd.DataFrame()

    names = pd.read_csv(namelist_path, header=None)
    all_df = pd.DataFrame(columns=["img_path", "label", "ifuku_id"])

    for _, row in names.iterrows():
        relative_csv = str(row[0])
        ifuku_id = parse_ifuku_id(relative_csv)
        if ifuku_id is None:
            continue
        if not (ifuku_start <= ifuku_id <= ifuku_end):
            continue

        csv_path = PROJECT_ROOT / "datas" / relative_csv
        if not csv_path.exists():
            csv_path = DATA_ROOT / relative_csv
            if not csv_path.exists():
                continue

        df = pd.read_csv(csv_path, header=0)
        df.columns = ["img_path", "Fz", "Fx", "Fy"]
        df = df[pd.to_numeric(df["Fz"], errors="coerce") <= 100].copy()

        df["img_path"] = df["img_path"].apply(
            lambda p: str(resolve_img_path(p))
        )
        df["label"]    = LABEL_MAP[mat_name]
        df["ifuku_id"] = ifuku_id

        all_df = pd.concat([
            all_df,
            df[["img_path", "label", "ifuku_id"]]
        ], ignore_index=True)

    return all_df


# =========================================================
# Sequence
# =========================================================
class ClassifierSequence(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df  = self.df.iloc[batch_idx]

        x_list = []
        for p in batch_df["img_path"].tolist():
            bgr = cv2.imread(str(p))
            if bgr is None:
                x_list.append(np.zeros((IMG_H, IMG_W, IMG_C), dtype=np.uint8))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != IMG_H or rgb.shape[1] != IMG_W:
                rgb = cv2.resize(rgb, (IMG_W, IMG_H))
            x_list.append(rgb)

        x = np.array(x_list, dtype="float32") / 255.0
        y = tf.keras.utils.to_categorical(
            batch_df["label"].values, num_classes=NUM_CLASSES
        )
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================================================
# モデル構築
# =========================================================
def build_classifier() -> Model:
    inp  = Input(shape=(IMG_H, IMG_W, IMG_C), name="input_tensor")
    conv = VGG16(weights="imagenet",
                 input_shape=(IMG_H, IMG_W, IMG_C),
                 include_top=False)(inp)
    flat = GlobalMaxPooling2D(name="gap")(conv)

    x = Dense(256, activation="relu",
              kernel_regularizer=regularizers.l2(0.001))(flat)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation="softmax", name="material_prob")(x)

    model = Model(inp, out)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4),
        metrics=["accuracy"]
    )
    return model


# =========================================================
# メイン
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "weights").mkdir(exist_ok=True)

    weight_path = RESULT_ROOT / "weights" / "classifier_weights.h5"
    if weight_path.exists():
        print(f"[SKIP] 重みが既に存在します: {weight_path}")
        return

    # ── 全素材のデータを読み込む ────────────────────────────
    print("=== データ読み込み ===")
    all_dfs = []
    for mat_name, mat_dir_name in MATERIAL_DIRS.items():
        df = load_material_data(
            mat_name, mat_dir_name,
            TRAIN_IFUKU_START, TRAIN_IFUKU_END
        )
        if len(df) > 0:
            print(f"  {mat_name}: {len(df)} samples")
            all_dfs.append(df)

    if len(all_dfs) == 0:
        print("[ERROR] データが読み込めませんでした")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  合計: {len(combined_df)} samples")

    # ── ifuku番号単位でtrain/val分割 ─────────────────────────
    all_ids = sorted(combined_df["ifuku_id"].unique().tolist())
    random.Random(RANDOM_SEED).shuffle(all_ids)
    val_size  = max(1, int(len(all_ids) * VAL_RATIO))
    val_ids   = sorted(all_ids[:val_size])
    train_ids = sorted(all_ids[val_size:])

    train_df = combined_df[combined_df["ifuku_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = combined_df[combined_df["ifuku_id"].isin(val_ids)].reset_index(drop=True)

    print(f"\n  train: {len(train_df)} samples ({len(train_ids)} sessions)")
    print(f"  val  : {len(val_df)} samples ({len(val_ids)} sessions)")

    # ── Sequence作成 ──────────────────────────────────────────
    train_seq = ClassifierSequence(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_seq   = ClassifierSequence(val_df,   batch_size=BATCH_SIZE, shuffle=False)

    # ── モデル構築・学習 ──────────────────────────────────────
    print("\n=== 学習開始 ===")
    model = build_classifier()

    # アーキテクチャ保存
    json_path = RESULT_ROOT / "classifier.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(model.to_json())

    history = model.fit(
        train_seq,
        epochs=EPOCHS,
        validation_data=val_seq,
        verbose=1
    )

    model.save_weights(str(weight_path))
    print(f"\n重み保存: {weight_path}")

    pd.DataFrame(history.history).to_csv(
        RESULT_ROOT / "learning_log.csv",
        index=False, encoding="utf-8-sig"
    )

    # ── val精度表示 ───────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_seq, verbose=0)
    print(f"\nval accuracy: {val_acc:.4f}")
    print(f"val loss    : {val_loss:.4f}")

    # ── ラベルマップ保存（リアルタイム推定時に参照） ──────────
    import json
    label_info = {
        "label_map": LABEL_MAP,
        "id_to_material": {v: k for k, v in LABEL_MAP.items()},
        "mask_pattern": MASK_PATTERN
    }
    with open(RESULT_ROOT / "label_info.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    print(f"ラベル情報保存: {RESULT_ROOT / 'label_info.json'}")

    del model, train_seq, val_seq
    gc.collect()

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()