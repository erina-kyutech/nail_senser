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

# ── train/val範囲 ──
# ifuku31以降は素材分類器と同じく完全ホールドアウト（重み更新に一切使わない）。
# ここでは train_val の元データを ifuku1~TRAINVAL_ID_END に制限する。
TRAINVAL_ID_END = 30


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
      元パス: felt_0-10xyz_dedup/ifuku1/360deg/0.png
      変換後: felt_0-10xyz_dedup_masked/ifuku1/360deg/nail_and_tip/0.png

    やること：
      1. 素材フォルダ名(xxx_0-10xyz_dedup)に_maskedを付ける
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

    # ① 素材フォルダ名（xxx_0-10xyz_dedup 等）に_maskedを付ける
    #    ※ endswith("_0-10xyz") だと "felt_0-10xyz_dedup" にマッチしないため
    #      部分一致(in)でチェックする（train_material_classifier.pyと同じロジック）
    for i, part in enumerate(parts):
        if "_0-10xyz" in part and "_masked" not in part:
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


def check_image_paths_sane(df: pd.DataFrame, n_check: int = 5) -> bool:
    """
    学習を始める前に、img_pathの一部が実在するか確認する。
    ここで全滅していたら resolve_img_path のパス変換がまだ間違っている合図。
    """
    sample_paths = df["img_path"].sample(min(n_check, len(df)), random_state=RANDOM_SEED).tolist()
    n_exist = sum(1 for p in sample_paths if Path(p).exists())
    print(f"  [CHECK] パス実在確認: {n_exist}/{len(sample_paths)} 件見つかりました")
    if n_exist == 0:
        print("  [ERROR] サンプルパスが1件も実在しません。resolve_img_path のロジックを確認してください。")
        for p in sample_paths:
            print(f"     見つからない例: {p}")
        return False
    return True


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

    # ── ifuku31以降を完全に除外（学習・valに一切混ざらないようにする） ──
    before_n = len(all_df)
    all_df = all_df[all_df["ifuku_id"] <= TRAINVAL_ID_END].reset_index(drop=True)
    excluded_n = before_n - len(all_df)
    if excluded_n > 0:
        print(f"  [INFO] ifuku{TRAINVAL_ID_END + 1}以降を{excluded_n}件除外"
              f"（完全ホールドアウト、train/valには一切使いません）")

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
def train_one_model(df: pd.DataFrame, result_dir: Path, model_name: str) -> dict:
    """
    df：学習に使う全データ（ifuku番号付き）
    result_dir：保存先フォルダ
    model_name：ログ表示用の名前

    戻り値: {"status": "success"/"skipped"/"aborted", "val_Fz_RMSE": float or None, ...}
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "weights").mkdir(exist_ok=True)
    (result_dir / "csv").mkdir(exist_ok=True)

    weight_path = result_dir / "weights" / "weight_ifuku_for0-10.h5"

    # ── すでに重みがある場合はスキップ ────────────────────
    # ★新しいマスク画像で再学習したい場合は、事前にこのweightsフォルダを
    #   削除 or リネームしておくこと。残したままだと古い重みのまま使われ続ける。
    if weight_path.exists():
        print(f"  [SKIP] 重みが既に存在するため学習をスキップ: {weight_path}")
        print(f"         新しいマスク画像で再学習したい場合はこのファイルを削除してから再実行してください。")
        return {"status": "skipped", "val_Fz_RMSE": None, "val_Fx_RMSE": None, "val_Fy_RMSE": None}

    # ── パスの実在チェック（黒画像学習の事故防止） ─────────
    if not check_image_paths_sane(df):
        print(f"  [ABORT] {model_name}: 画像パスが解決できないため学習を中止します。")
        return {"status": "aborted_bad_path", "val_Fz_RMSE": None, "val_Fx_RMSE": None, "val_Fy_RMSE": None}

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

    # ── GPUメモリ解放 ─────────────────────────────────────
    # 15モデルを1プロセス内で連続学習するため、これが無いと
    # VGG16の読み込みが積み重なってGPUメモリを圧迫し、途中でOOMする恐れがある。
    del model, train_seq, val_seq
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "status": "success",
        "val_Fz_RMSE": metrics["val_Fz_RMSE"],
        "val_Fx_RMSE": metrics["val_Fx_RMSE"],
        "val_Fy_RMSE": metrics["val_Fy_RMSE"],
    }


def run_model_safely(df: pd.DataFrame, result_dir: Path, model_name: str, progress_log_path: Path) -> None:
    """
    train_one_modelを例外から保護して実行し、結果を進捗ログに追記する。
    ここで例外を握りつぶすことで、1モデルの予期しない失敗
    （OOM、画像破損など）が残りのモデル学習を止めないようにする。
    """
    import datetime
    import csv as csv_module

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = {"status": "error", "val_Fz_RMSE": None, "val_Fx_RMSE": None, "val_Fy_RMSE": None}
    error_message = ""

    try:
        result = train_one_model(df, result_dir, model_name)
    except Exception as e:
        import traceback
        error_message = f"{type(e).__name__}: {e}"
        print(f"  [ERROR] {model_name}: {error_message}")
        traceback.print_exc()
        # 例外が起きてもGPUメモリだけは解放しておく
        try:
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception:
            pass

    # 進捗ログに追記（ファイルが無ければヘッダーから作る）
    write_header = not progress_log_path.exists()
    with open(progress_log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv_module.writer(f)
        if write_header:
            writer.writerow(["timestamp", "model_name", "status",
                             "val_Fz_RMSE", "val_Fx_RMSE", "val_Fy_RMSE", "error"])
        writer.writerow([timestamp, model_name, result["status"],
                         result.get("val_Fz_RMSE"), result.get("val_Fx_RMSE"),
                         result.get("val_Fy_RMSE"), error_message])


# =========================================================
# メイン
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    progress_log_path = RESULT_ROOT / "training_progress.csv"
    print(f"進捗ログ: {progress_log_path}")
    print("（放置して戻ってきたら、まずこのCSVを確認すれば全モデルの成否が一覧でわかります）\n")

    print(f"=== 実行予定のマスクパターン: {MASK_PATTERNS} ===\n")

    for mask_pattern in MASK_PATTERNS:
        print(f"\n{'='*60}")
        print(f"マスクパターン: {mask_pattern}")
        print(f"{'='*60}")

        # ── 素材ごとの個別モデル ──────────────────────────
        for mat_name, mat_dir_name in MATERIAL_DIRS.items():
            namelist_path = DATA_ROOT / mat_dir_name / "namelist.csv"
            model_name = f"{mat_name}_{mask_pattern}"

            print(f"\n--- {model_name} ---")

            try:
                df = load_datalog(namelist_path, mask_pattern)
            except Exception as e:
                print(f"  [ERROR] datalog読み込み失敗: {e}")
                continue

            if len(df) == 0:
                print("  [SKIP] データが空です")
                continue

            print(f"  全データ: {len(df)} samples")

            result_dir = RESULT_ROOT / model_name
            run_model_safely(df, result_dir, model_name, progress_log_path)

        # ── 全素材混合モデル ──────────────────────────────
        model_name = f"all_materials_{mask_pattern}"
        print(f"\n--- {model_name} ---")

        all_dfs = []
        for mat_name, mat_dir_name in MATERIAL_DIRS.items():
            namelist_path = DATA_ROOT / mat_dir_name / "namelist.csv"
            try:
                df = load_datalog(namelist_path, mask_pattern)
            except Exception as e:
                print(f"  [ERROR] {mat_name} datalog読み込み失敗: {e}")
                continue
            if len(df) > 0:
                df["material"] = mat_name
                all_dfs.append(df)

        if len(all_dfs) == 0:
            print("  [SKIP] データが空です")
            continue

        # ── 各素材から均等にサンプリングして合計を個別モデルと揃える ──
        total_samples = sum(len(df) for df in all_dfs)
        avg_per_mat   = total_samples // len(all_dfs)
        per_mat_target = avg_per_mat // len(all_dfs)

        sizes = {df["material"].iloc[0]: len(df) for df in all_dfs}
        print(f"  各素材のサンプル数: {sizes}")
        print(f"  各素材から約{per_mat_target}枚サンプリング → 合計約{per_mat_target * len(all_dfs)}枚")

        sampled_dfs = []
        for df in all_dfs:
            mat_name = df["material"].iloc[0]
            if len(df) > per_mat_target:
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

        result_dir = RESULT_ROOT / model_name
        run_model_safely(combined_df, result_dir, model_name, progress_log_path)

    print("\n\n=== 全モデル学習完了（進捗ログで成否を確認してください） ===")
    print(f"進捗ログ: {progress_log_path}")


if __name__ == "__main__":
    main()