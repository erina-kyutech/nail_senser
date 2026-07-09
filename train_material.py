# -*- coding: utf-8 -*-
"""
train_material_models.py

素材ごと・マスクパターンごとにVGG16モデルを学習するスクリプト。
TF環境（Tactile_sensors_20230929）で実行する。

学習するモデルの組み合わせ：
  素材4種（felt, acrylic, paper, aluminum）× マスク3パターン（nail_and_tip, nail_only, tip_only）= 12モデル
  全素材混合 × マスク3パターン = 3モデル
  合計15モデル

train/val分割：ifuku番号単位（全データを学習に使う、valはifuku単位でランダム15%）
テスト：リアルタイムで実施（このスクリプトではテスト分割なし）

保存先：
  result/CNN_result/material_models/{素材名}_{マスクパターン}/
  result/CNN_result/material_models/all_materials_{マスクパターン}/
"""

from __future__ import annotations

import os
import gc
import random
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

NORMAL_FORCE_NORMALIZE = 10.0
SHEAR_FORCE_NORMALIZE  = 5.0

PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")
DATA_ROOT    = PROJECT_ROOT / "datas"
RESULT_ROOT  = PROJECT_ROOT / "result" / "CNN_result" / "material_models"

# 素材フォルダ名（_maskedを付けたもの）
MATERIAL_DIRS = {
    "felt":     "felt_0-10xyz_dedup_masked",
    "acrylic":  "acrylic_0-10xyz_dedup_masked",
    "paper":    "paper_0-10xyz_dedup_masked",
    "aluminum": "aluminum_0-10xyz_dedup_masked",
}

# マスクパターン（フォルダ名）
MASK_PATTERNS = ["nail_and_tip", "nail_only", "tip_only"]

# =========================================================
# ユーティリティ
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_img_path(path_str: str, mask_pattern: str) -> Path:
    """
    datalog.csvのパスをマスク済みパターンのパスに変換する。
    例：
      元パス: felt_0-10xyz/ifuku1/360deg/0.png
      変換後: felt_0-10xyz_masked/ifuku1/360deg/nail_and_tip/0.png

    やること：
      1. 素材フォルダ名に_maskedを付ける（felt_0-10xyz → felt_0-10xyz_masked）
      2. 360deg/の直下にマスクパターンフォルダを挿入
    """
    p = Path(path_str)
    if not p.is_absolute():
        path_str2 = str(p).replace("./", "", 1).replace(".\\", "", 1)
        # datasが既にパスに含まれてる場合は重複を避ける
        if path_str2.startswith("datas\\") or path_str2.startswith("datas/"):
            p = PROJECT_ROOT / path_str2
        else:
            p = DATA_ROOT / path_str2

    # パーツに分解して変換する
    parts = list(p.parts)

    # ① 素材フォルダ名（xxxxx_0-10xyz）に_maskedを付ける
    for i, part in enumerate(parts):
        if part.endswith("_0-10xyz") and not part.endswith("_masked"):
            parts[i] = part + "_masked"
            break

    # ② 360deg の直下にmask_patternフォルダを挿入
    try:
        deg_idx = parts.index("360deg")
        parts.insert(deg_idx + 1, mask_pattern)
    except ValueError:
        pass

    return Path(*parts)


def parse_ifuku_id(path_str: str) -> int | None:
    import re
    m = re.search(r'ifuku(\d+)', str(path_str))
    return int(m.group(1)) if m else None


def normalize_y(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype("float32")
    y[:, 0] /= NORMAL_FORCE_NORMALIZE
    y[:, 1] += SHEAR_FORCE_NORMALIZE
    y[:, 2] += SHEAR_FORCE_NORMALIZE
    y[:, 1] /= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 2] /= (SHEAR_FORCE_NORMALIZE * 2.0)
    return y


def unnormalize_y(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype("float32")
    y[:, 0] *= NORMAL_FORCE_NORMALIZE
    y[:, 1] *= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 2] *= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 1] -= SHEAR_FORCE_NORMALIZE
    y[:, 2] -= SHEAR_FORCE_NORMALIZE
    return y


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    out = {}
    for i, name in enumerate(["Fz", "Fx", "Fy"]):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        out[f"{prefix}_{name}_MAE"]  = mae
        out[f"{prefix}_{name}_RMSE"] = rmse
    return out


# =========================================================
# データ読み込み
# =========================================================
def load_datalog(namelist_path: Path, mask_pattern: str) -> pd.DataFrame:
    """
    namelist.csvを読んで全セッションのdatalogを結合する。
    img_pathをマスクパターンのパスに変換する。

    namelist.csvが_maskedフォルダにない場合は、
    元フォルダ（_maskedを除いたフォルダ）のnamelist.csvを参照する。
    """
    if not namelist_path.exists():
        # _dedup_masked → _dedup → 元フォルダ の順に探す
        fallback_dedup = Path(str(namelist_path).replace("_dedup_masked", "_dedup"))
        fallback_orig  = Path(str(namelist_path).replace("_dedup_masked", "").replace("_masked", ""))

        if fallback_dedup.exists():
            print(f"  [INFO] namelist.csvを_dedupフォルダから参照: {fallback_dedup}")
            namelist_path = fallback_dedup
        elif fallback_orig.exists():
            print(f"  [INFO] namelist.csvを元フォルダから参照: {fallback_orig}")
            namelist_path = fallback_orig
        else:
            raise FileNotFoundError(
                f"namelist.csv not found: {namelist_path}\n"
                f"_dedupにも見つかりません: {fallback_dedup}\n"
                f"元フォルダにも見つかりません: {fallback_orig}"
            )

    names = pd.read_csv(namelist_path, header=None)
    all_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy", "ifuku_id"])

    for _, row in names.iterrows():
        relative_csv = str(row[0])
        csv_path = PROJECT_ROOT / "datas" / relative_csv

        if not csv_path.exists():
            print(f"  [WARN] not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path, header=0)
        df.columns = ["img_path", "Fz", "Fx", "Fy"]
        df = df[pd.to_numeric(df["Fz"], errors="coerce") <= 100].copy()

        # img_pathをマスクパターンのパスに変換
        df["img_path"] = df["img_path"].apply(
            lambda p: str(resolve_img_path(p, mask_pattern))
        )
        df["ifuku_id"] = df["img_path"].apply(parse_ifuku_id)
        all_df = pd.concat([all_df, df], ignore_index=True)

    all_df = all_df.dropna(subset=["ifuku_id"]).copy()
    all_df["ifuku_id"] = all_df["ifuku_id"].astype(int)
    return all_df


# =========================================================
# Sequence（バッチごとに画像読み込み）
# =========================================================
class MaterialSequence(tf.keras.utils.Sequence):
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
                rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            x_list.append(rgb)

        x = np.array(x_list, dtype="float32") / 255.0
        y = batch_df[["Fz", "Fx", "Fy"]].values.astype("float32")
        y_norm = normalize_y(y)

        return x, [y_norm[:, 0], y_norm[:, 1], y_norm[:, 2]]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================================================
# モデル構築
# =========================================================
def build_model() -> Model:
    l2_alpha  = 0.001
    mid_units = 342
    optimizer = Adam(learning_rate=1e-4)

    inp  = Input(shape=(IMG_H, IMG_W, IMG_C), name="input_tensor")
    conv = VGG16(weights="imagenet",
                 input_shape=(IMG_H, IMG_W, IMG_C),
                 include_top=False)(inp)
    flat = GlobalMaxPooling2D(name="flatten")(conv)

    fz = Dense(mid_units, activation="relu",
               kernel_regularizer=regularizers.l2(l2_alpha))(flat)
    fz = Dropout(0.2)(fz)
    fz = Dense(1, activation="linear", name="Fz",
               kernel_regularizer=regularizers.l2(l2_alpha))(fz)

    fx = Dense(mid_units, activation="relu",
               kernel_regularizer=regularizers.l2(l2_alpha))(flat)
    fx = Dropout(0.2)(fx)
    fx = Dense(1, activation="linear", name="Fx",
               kernel_regularizer=regularizers.l2(l2_alpha))(fx)

    fy = Dense(mid_units, activation="relu",
               kernel_regularizer=regularizers.l2(l2_alpha))(flat)
    fy = Dropout(0.2)(fy)
    fy = Dense(1, activation="linear", name="Fy",
               kernel_regularizer=regularizers.l2(l2_alpha))(fy)

    model = Model(inp, [fz, fx, fy])
    model.compile(
        loss={"Fz": "mean_squared_error",
              "Fx": "mean_squared_error",
              "Fy": "mean_squared_error"},
        optimizer=optimizer
    )
    return model


# =========================================================
# 1モデル分の学習
# =========================================================
def train_one_model(df: pd.DataFrame, result_dir: Path, model_name: str):
    """
    df：学習に使う全データ（ifuku番号付き）
    result_dir：保存先フォルダ
    model_name：ログ表示用の名前
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "weights").mkdir(exist_ok=True)
    (result_dir / "csv").mkdir(exist_ok=True)

    weight_path = result_dir / "weights" / "weight_ifuku_for0-10.h5"

    # ── すでに重みがある場合はスキップ ────────────────────
    if weight_path.exists():
        print(f"  [SKIP] 重みが存在します: {weight_path}")
        return

    # ── ifuku番号単位でtrain/val分割 ─────────────────────
    all_ids = sorted(df["ifuku_id"].unique().tolist())
    random.Random(RANDOM_SEED).shuffle(all_ids)
    val_size  = max(1, int(len(all_ids) * VAL_RATIO))
    val_ids   = sorted(all_ids[:val_size])
    train_ids = sorted(all_ids[val_size:])

    train_df = df[df["ifuku_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["ifuku_id"].isin(val_ids)].reset_index(drop=True)

    print(f"  train: {len(train_df)} samples ({len(train_ids)} sessions)")
    print(f"  val  : {len(val_df)} samples ({len(val_ids)} sessions)")

    # ── Sequence作成 ──────────────────────────────────────
    train_seq = MaterialSequence(train_df, batch_size=BATCH_SIZE, shuffle=True)
    val_seq   = MaterialSequence(val_df,   batch_size=BATCH_SIZE, shuffle=False)

    # ── モデル構築 ────────────────────────────────────────
    model = build_model()

    # アーキテクチャ保存（1回だけ）
    json_path = result_dir / "for0-10.json"
    if not json_path.exists():
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(model.to_json())

    # ── 学習 ──────────────────────────────────────────────
    history = model.fit(
        train_seq,
        epochs=EPOCHS,
        validation_data=val_seq,
        verbose=1
    )

    model.save_weights(str(weight_path))
    print(f"  saved: {weight_path}")

    pd.DataFrame(history.history).to_csv(
        result_dir / "csv" / "learning_log.csv",
        index=False, encoding="utf-8-sig"
    )

    # ── val評価 ───────────────────────────────────────────
    val_pred_list = model.predict(val_seq, verbose=0)
    y_val_pred = unnormalize_y(
        np.concatenate(val_pred_list, axis=1)
    )
    y_val_true = val_df[["Fz", "Fx", "Fy"]].values.astype("float32")

    metrics = calc_metrics(y_val_true, y_val_pred, "val")
    pd.DataFrame([metrics]).to_csv(
        result_dir / "csv" / "val_metrics.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"  val Fz RMSE: {metrics['val_Fz_RMSE']:.3f} N")
    print(f"  val Fx RMSE: {metrics['val_Fx_RMSE']:.3f} N")
    print(f"  val Fy RMSE: {metrics['val_Fy_RMSE']:.3f} N")

    del model, train_seq, val_seq
    gc.collect()


# =========================================================
# メイン
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    for mask_pattern in MASK_PATTERNS:
        print(f"\n{'='*60}")
        print(f"マスクパターン: {mask_pattern}")
        print(f"{'='*60}")

        # ── 素材ごとの個別モデル ──────────────────────────
        for mat_name, mat_dir_name in MATERIAL_DIRS.items():
            namelist_path = DATA_ROOT / mat_dir_name / "namelist.csv"

            print(f"\n--- {mat_name} × {mask_pattern} ---")

            df = load_datalog(namelist_path, mask_pattern)
            if len(df) == 0:
                print("  [SKIP] データが空です")
                continue

            print(f"  全データ: {len(df)} samples")

            result_dir = RESULT_ROOT / f"{mat_name}_{mask_pattern}"
            train_one_model(df, result_dir, f"{mat_name}_{mask_pattern}")

        # ── 全素材混合モデル ──────────────────────────────
        print(f"\n--- all_materials × {mask_pattern} ---")

        all_dfs = []
        for mat_name, mat_dir_name in MATERIAL_DIRS.items():
            namelist_path = DATA_ROOT / mat_dir_name / "namelist.csv"
            df = load_datalog(namelist_path, mask_pattern)
            if len(df) > 0:
                df["material"] = mat_name
                all_dfs.append(df)

        if len(all_dfs) == 0:
            print("  [SKIP] データが空です")
            continue

        # ── 各素材から均等にサンプリングして合計を個別モデルと揃える ──
        # 個別モデル（1素材）と同じ合計枚数になるように
        # 各素材から均等に取る
        # 例：各素材6万枚 × 4素材 → 各素材から1.5万枚 → 合計6万枚
        total_samples = sum(len(df) for df in all_dfs)
        avg_per_mat   = total_samples // len(all_dfs)  # 平均サンプル数（個別モデルと同じスケール）
        per_mat_target = avg_per_mat // len(all_dfs)   # 各素材から取る枚数

        sizes = {df["material"].iloc[0]: len(df) for df in all_dfs}
        print(f"  各素材のサンプル数: {sizes}")
        print(f"  各素材から約{per_mat_target}枚サンプリング → 合計約{per_mat_target * len(all_dfs)}枚")

        sampled_dfs = []
        for df in all_dfs:
            mat_name = df["material"].iloc[0]
            if len(df) > per_mat_target:
                # ifuku番号単位でサンプリング
                all_ids = sorted(df["ifuku_id"].unique().tolist())
                random.Random(RANDOM_SEED).shuffle(all_ids)
                sampled_ids = []
                sampled_count = 0
                for id_ in all_ids:
                    id_count = len(df[df["ifuku_id"] == id_])
                    if sampled_count + id_count <= per_mat_target:
                        sampled_ids.append(id_)
                        sampled_count += id_count
                    if sampled_count >= int(per_mat_target * 0.95):
                        break
                sampled_df = df[df["ifuku_id"].isin(sampled_ids)].reset_index(drop=True)
            else:
                sampled_df = df
            print(f"  {mat_name}: {len(df)} → {len(sampled_df)} samples")
            sampled_dfs.append(sampled_df)

        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"  混合モデル合計: {len(combined_df)} samples（個別モデルと同スケール）")

        result_dir = RESULT_ROOT / f"all_materials_{mask_pattern}"
        train_one_model(combined_df, result_dir, f"all_materials_{mask_pattern}")

    print("\n\n=== 全モデル学習完了 ===")


if __name__ == "__main__":
    main()