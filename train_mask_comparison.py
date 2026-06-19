# -*- coding: utf-8 -*-
"""
3パターンの入力画像でVGG16ベースの3軸力推定モデルを学習・評価するスクリプト。

パターン:
  1. original      : 元画像(マスクなし)
  2. nail_only     : 爪のみマスク済み画像
  3. nail_and_tip  : 爪+指先マスク済み画像

分割:
  train: ifuku1  ~ ifuku166
  test : ifuku167 ~ ifuku180

出力:
  result/CNN_result/mask_comparison/{pattern}/
    weights/weight_ifuku_for0-10.h5
    csv/metrics_summary.csv
    csv/test_predictions.csv
    csv/learning_log.csv
"""

from __future__ import annotations

import gc
import math
import os
import random
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# 設定
# =========================================================
RANDOM_SEED = 42
IMG_H = 150
IMG_W = 290
IMG_C = 3
BATCH_SIZE = 32
EPOCHS = 10
VAL_RATIO = 0.15

NORMAL_FORCE_NORMALIZE = 10.0
SHEAR_FORCE_NORMALIZE  = 5.0

SUBJECT_NAME   = "ifuku"
TRAINVAL_START = 1
TRAINVAL_END   = 166
TEST_START     = 167
TEST_END       = 180

PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")
DATA_ROOT    = PROJECT_ROOT / "datas"

# 3パターンのデータルート
DATA_ROOTS = {
    "original"    : DATA_ROOT / "record0-10xyz",
    "nail_only"   : DATA_ROOT / "masked_nail_only",
    "nail_and_tip": DATA_ROOT / "masked_nail_and_tip",
}

RESULT_BASE  = PROJECT_ROOT / "result" / "CNN_result" / "mask_comparison"
NAMELIST_PATH = DATA_ROOT / "record0-10xyz" / "namelist.csv"


# =========================================================
# 共通関数
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def parse_ifuku_id(path_str: str) -> int | None:
    for part in Path(path_str).parts:
        if part.startswith("ifuku"):
            num = part.replace("ifuku", "")
            if num.isdigit():
                return int(num)
    return None


# =========================================================
# データ読み込み
# =========================================================
class DataLoader:
    def __init__(self, data_root: Path):
        self.data_root = data_root

    def personal_dataload(self) -> pd.DataFrame:
        names = pd.read_csv(NAMELIST_PATH, header=None)
        all_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy", "ifuku_id"])

        for _, row in names.iterrows():
            relative_csv_path = str(row[0])

            if SUBJECT_NAME not in relative_csv_path:
                continue

            # namelist.csv は record0-10xyz 基準のパスなのでそのまま使う
            csv_path = PROJECT_ROOT / "datas" / relative_csv_path
            if not csv_path.exists():
                print(f"[WARN] CSV not found: {csv_path}")
                continue

            csv_record = pd.read_csv(csv_path, header=0)
            csv_record.columns = ["img_path", "Fz", "Fx", "Fy"]
            csv_record = csv_record[
                pd.to_numeric(csv_record["Fz"], errors="coerce") <= 100
            ].copy()

            # パスのルートを置換(record0-10xyz → 今のdata_root)
            csv_record["img_path"] = csv_record["img_path"].apply(
                lambda p: self._replace_root(p)
            )
            csv_record["ifuku_id"] = csv_record["img_path"].apply(parse_ifuku_id)
            all_df = pd.concat([all_df, csv_record], ignore_index=True)

        all_df = all_df.dropna(subset=["ifuku_id"]).copy()
        all_df["ifuku_id"] = all_df["ifuku_id"].astype(int)
        return all_df

    def _replace_root(self, path_str: str) -> str:
        """
        元のパス(record0-10xyz/ifukuN/360deg/M.png)を
        現在のdata_root配下のパスに置換する。
        """
        p = Path(path_str)
        # パスの中からifukuN以降を取り出す
        parts = p.parts
        for i, part in enumerate(parts):
            if part.startswith("ifuku"):
                rel = Path(*parts[i:])  # ifukuN/360deg/M.png
                return str(self.data_root / rel)
        return path_str  # 変換できなければそのまま返す

    def load_image(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"image not found: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != IMG_H or rgb.shape[1] != IMG_W:
            rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        return rgb

    def df_to_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        x_list = [self.load_image(p) for p in df["img_path"].tolist()]
        x = np.array(x_list, dtype="float32") / 255.0
        y = df[["Fz", "Fx", "Fy"]].values.astype("float32")
        return x, y


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, loader: DataLoader, batch_size=32, shuffle=True):
        self.df         = df.reset_index(drop=True)
        self.loader     = loader
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.indices    = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df  = self.df.iloc[batch_idx]
        x_list    = [self.loader.load_image(p) for p in batch_df["img_path"].tolist()]
        x         = np.array(x_list, dtype="float32") / 255.0
        y         = batch_df[["Fz", "Fx", "Fy"]].values.astype("float32")
        y_norm    = normalize_y(y)
        return x, [y_norm[:, 0], y_norm[:, 1], y_norm[:, 2]]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# =========================================================
# モデル
# =========================================================
def build_model() -> Model:
    l2_alpha              = 0.001
    middle_class_recurrence = 342
    optimizer             = Adam(learning_rate=1e-4)

    input_tensor = Input(shape=(IMG_H, IMG_W, IMG_C), name="input_tensor")
    conv         = VGG16(weights="imagenet", input_shape=(IMG_H, IMG_W, IMG_C),
                         include_top=False)(input_tensor)
    flatten      = GlobalMaxPooling2D(name="flatten")(conv)

    def head(x, name):
        x = Dense(342, activation="relu",
                  kernel_regularizer=regularizers.l2(l2_alpha))(x)
        x = Dropout(0.2)(x)
        x = Dense(1, activation="linear", name=name,
                  kernel_regularizer=regularizers.l2(l2_alpha))(x)
        return x

    fz = head(flatten, "Fz")
    fx = head(flatten, "Fx")
    fy = head(flatten, "Fy")

    model = Model(input_tensor, [fz, fx, fy])
    model.compile(
        loss={"Fz": "mean_squared_error",
              "Fx": "mean_squared_error",
              "Fy": "mean_squared_error"},
        optimizer=optimizer
    )
    return model


# =========================================================
# 1パターン分の学習・評価
# =========================================================
def run_one_pattern(pattern_name: str, data_root: Path):
    print(f"\n{'='*60}")
    print(f"パターン: {pattern_name}")
    print(f"データルート: {data_root}")
    print(f"{'='*60}")

    result_dir = RESULT_BASE / pattern_name
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "weights").mkdir(exist_ok=True)
    (result_dir / "csv").mkdir(exist_ok=True)

    # データ読み込み
    loader  = DataLoader(data_root)
    full_df = loader.personal_dataload()
    print(f"全データ数: {len(full_df)}")

    # train/val/test 分割
    trainval_df = full_df[
        (full_df["ifuku_id"] >= TRAINVAL_START) &
        (full_df["ifuku_id"] <= TRAINVAL_END)
    ].copy()

    test_df = full_df[
        (full_df["ifuku_id"] >= TEST_START) &
        (full_df["ifuku_id"] <= TEST_END)
    ].copy().reset_index(drop=True)

    trainval_ids = list(range(TRAINVAL_START, TRAINVAL_END + 1))
    random.Random(RANDOM_SEED).shuffle(trainval_ids)
    val_size  = max(1, int(len(trainval_ids) * VAL_RATIO))
    val_ids   = sorted(trainval_ids[:val_size])
    train_ids = sorted(trainval_ids[val_size:])

    train_df = trainval_df[trainval_df["ifuku_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = trainval_df[trainval_df["ifuku_id"].isin(val_ids)].reset_index(drop=True)

    print(f"train: {len(train_df)} 枚 (ifuku {train_ids[:3]}...)")
    print(f"val  : {len(val_df)} 枚 (ifuku {val_ids[:3]}...)")
    print(f"test : {len(test_df)} 枚 (ifuku {TEST_START}~{TEST_END})")

    # Sequence
    train_seq = ImageSequence(train_df, loader, batch_size=BATCH_SIZE, shuffle=True)
    val_seq   = ImageSequence(val_df,   loader, batch_size=BATCH_SIZE, shuffle=False)

    # モデル構築
    model       = build_model()
    weight_path = result_dir / "weights" / f"weight_{SUBJECT_NAME}_for0-10.h5"

    if weight_path.exists():
        print(f"既存の重みを読み込みます: {weight_path}")
        model.load_weights(str(weight_path))
    else:
        print("学習開始...")
        history = model.fit(
            train_seq,
            epochs=EPOCHS,
            validation_data=val_seq,
            verbose=1
        )
        model.save_weights(str(weight_path))

        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(result_dir / "csv" / "learning_log.csv",
                       index=False, encoding="utf-8-sig")
        print(f"重みを保存: {weight_path}")

    # テスト評価
    print("テスト評価中...")
    x_test, y_test_true = loader.df_to_xy(test_df)
    test_pred_list = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
    y_test_pred_norm = np.concatenate(test_pred_list, axis=1)
    y_test_pred = unnormalize_y(y_test_pred_norm)

    metrics = calc_metrics(y_test_true, y_test_pred, "test")
    print(f"\n--- {pattern_name} テスト結果 ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存
    pd.DataFrame([metrics]).to_csv(
        result_dir / "csv" / "metrics_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    result_df = test_df.copy()
    result_df["pred_Fz"] = y_test_pred[:, 0]
    result_df["pred_Fx"] = y_test_pred[:, 1]
    result_df["pred_Fy"] = y_test_pred[:, 2]
    result_df["err_Fz"]  = result_df["pred_Fz"] - result_df["Fz"]
    result_df["err_Fx"]  = result_df["pred_Fx"] - result_df["Fx"]
    result_df["err_Fy"]  = result_df["pred_Fy"] - result_df["Fy"]
    result_df.to_csv(
        result_dir / "csv" / "test_predictions.csv",
        index=False, encoding="utf-8-sig"
    )

    # メモリ解放
    del model, x_test, y_test_true, y_test_pred
    gc.collect()
    tf.keras.backend.clear_session()

    return metrics


# =========================================================
# メイン
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    RESULT_BASE.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for pattern_name, data_root in DATA_ROOTS.items():
        if not data_root.exists():
            print(f"[WARN] データルートが見つかりません(スキップ): {data_root}")
            continue
        metrics = run_one_pattern(pattern_name, data_root)
        all_metrics[pattern_name] = metrics

    # 3パターンまとめて比較表を出力
    print(f"\n{'='*60}")
    print("=== 3パターン比較まとめ ===")
    summary_rows = []
    for pattern_name, metrics in all_metrics.items():
        row = {"pattern": pattern_name}
        row.update(metrics)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULT_BASE / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(summary_df.T.to_string())
    print(f"\n比較表を保存: {summary_path}")
    print("=== 全パターン完了 ===")


if __name__ == "__main__":
    main()
