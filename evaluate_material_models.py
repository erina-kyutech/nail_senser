# -*- coding: utf-8 -*-
"""
evaluate_material_models.py

学習済みモデルを使って素材内・素材間の全パターン（20通り×3マスク＝60通り）で評価する。

テストデータ：ifuku31〜35（学習に使っていないデータ）
出力：
  ① 1枚1枚の真値・推定値CSV（散布図用）
  ② RMSE集計CSV
  ③ Excelファイル（シート別に整理）

実行環境：TF環境（Tactile_sensors_20230929）
"""

from __future__ import annotations

import gc
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json

# =========================================================
# 設定
# =========================================================
IMG_H = 150
IMG_W = 290
IMG_C = 3
BATCH_SIZE = 128

NORMAL_FORCE_NORMALIZE = 10.0
SHEAR_FORCE_NORMALIZE  = 5.0

TEST_IFUKU_START = 31
TEST_IFUKU_END   = 35

PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")
DATA_ROOT    = PROJECT_ROOT / "datas"
RESULT_ROOT  = PROJECT_ROOT / "result" / "CNN_result" / "material_models"
EVAL_ROOT    = RESULT_ROOT / "evaluation"

MATERIAL_DIRS = {
    "felt":     "felt_0-10xyz_dedup_masked",
    "acrylic":  "acrylic_0-10xyz_dedup_masked",
    "paper":    "paper_0-10xyz_dedup_masked",
    "aluminum": "aluminum_0-10xyz_dedup_masked",
}

MASK_PATTERNS = ["nail_and_tip", "nail_only", "tip_only"]


# =========================================================
# ユーティリティ（train_material.pyと同じ実装）
# =========================================================
def parse_ifuku_id(path_str):
    m = re.search(r'ifuku(\d+)', str(path_str))
    return int(m.group(1)) if m else None


def resolve_img_path(path_str: str, mask_pattern: str) -> Path:
    """
    train_material.pyと同じ実装。
    datalog.csvのパスをマスク済みパターンのパスに変換する。
    例：
      元パス: felt_0-10xyz_dedup/ifuku1/360deg/0.png
      変換後: felt_0-10xyz_dedup_masked/ifuku1/360deg/nail_and_tip/0.png
    """
    p = Path(path_str)
    if not p.is_absolute():
        path_str2 = str(p).replace("./", "", 1).replace(".\\", "", 1)
        if path_str2.startswith("datas\\") or path_str2.startswith("datas/"):
            p = PROJECT_ROOT / path_str2
        else:
            p = DATA_ROOT / path_str2

    parts = list(p.parts)

    # 素材フォルダ名に_dedup_maskedを付ける
    # felt_0-10xyz_dedup → felt_0-10xyz_dedup_masked
    # felt_0-10xyz       → felt_0-10xyz_dedup_masked
    for i, part in enumerate(parts):
        if "_0-10xyz" in part and "_masked" not in part:
            if "_dedup" in part:
                parts[i] = part + "_masked"        # _dedup → _dedup_masked
            else:
                parts[i] = part + "_dedup_masked"  # _0-10xyz → _0-10xyz_dedup_masked
            break

    # 360degの直下にmask_patternを挿入
    try:
        deg_idx = parts.index("360deg")
        parts.insert(deg_idx + 1, mask_pattern)
    except ValueError:
        pass

    return Path(*parts)


def unnormalize_y(y: np.ndarray) -> np.ndarray:
    y = y.copy().astype("float32")
    y[:, 0] *= NORMAL_FORCE_NORMALIZE
    y[:, 1] *= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 2] *= (SHEAR_FORCE_NORMALIZE * 2.0)
    y[:, 1] -= SHEAR_FORCE_NORMALIZE
    y[:, 2] -= SHEAR_FORCE_NORMALIZE
    return y


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for i, name in enumerate(["Fz", "Fx", "Fy"]):
        rmse = math.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))
        out[f"RMSE_{name}"] = round(rmse, 4)
    return out


# =========================================================
# テストデータ読み込み（train_material.pyと同じ方式）
# =========================================================
def load_test_data(namelist_path: Path, mask_pattern: str) -> pd.DataFrame:
    """
    train_material.pyのload_datalogsと同じ方式でデータを読み込む。
    ifuku31〜35だけフィルタリングする。
    """
    if not namelist_path.exists():
        fallback_dedup = Path(str(namelist_path).replace("_dedup_masked", "_dedup"))
        fallback_orig  = Path(str(namelist_path).replace("_dedup_masked", "").replace("_masked", ""))
        if fallback_dedup.exists():
            namelist_path = fallback_dedup
        elif fallback_orig.exists():
            namelist_path = fallback_orig
        else:
            return pd.DataFrame()

    names = pd.read_csv(namelist_path, header=None)
    all_df = pd.DataFrame(columns=["img_path", "Fz", "Fx", "Fy", "ifuku_id"])

    for _, row in names.iterrows():
        relative_csv = str(row[0])

        # ifuku番号を確認してテスト範囲のみ処理
        ifuku_id = parse_ifuku_id(relative_csv)
        if ifuku_id is None:
            continue
        if not (TEST_IFUKU_START <= ifuku_id <= TEST_IFUKU_END):
            continue

        csv_path = PROJECT_ROOT / "datas" / relative_csv
        if not csv_path.exists():
            # _dedupのフォルダを探す
            csv_path = DATA_ROOT / relative_csv
            if not csv_path.exists():
                continue

        df = pd.read_csv(csv_path, header=0)
        df.columns = ["img_path", "Fz", "Fx", "Fy"]
        df = df[pd.to_numeric(df["Fz"], errors="coerce") <= 100].copy()

        # train_material.pyと同じresolve_img_pathで変換
        df["img_path"] = df["img_path"].apply(
            lambda p: str(resolve_img_path(p, mask_pattern))
        )
        df["ifuku_id"] = ifuku_id
        all_df = pd.concat([all_df, df], ignore_index=True)

    return all_df


# =========================================================
# モデル読み込み・予測
# =========================================================
def load_model(model_dir: Path):
    json_path   = model_dir / "for0-10.json"
    weight_path = model_dir / "weights" / "weight_ifuku_for0-10.h5"
    if not json_path.exists() or not weight_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        model = model_from_json(f.read())
    model.load_weights(str(weight_path))
    return model


def predict_batch(model, df: pd.DataFrame, chunk_size=512) -> np.ndarray:
    """チャンク方式で推論してOOMを防ぐ"""
    all_preds = []
    paths = df["img_path"].tolist()
    N = len(paths)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_list = []
        for p in paths[start:end]:
            bgr = cv2.imread(str(p))
            if bgr is None:
                x_list.append(np.zeros((IMG_H, IMG_W, IMG_C), dtype=np.uint8))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != IMG_H or rgb.shape[1] != IMG_W:
                rgb = cv2.resize(rgb, (IMG_W, IMG_H))
            x_list.append(rgb)

        x = np.array(x_list, dtype="float32") / 255.0
        pred_list = model.predict(x, batch_size=32, verbose=0)
        all_preds.append(np.concatenate(pred_list, axis=1))
        del x, pred_list

    return unnormalize_y(np.concatenate(all_preds, axis=0))


# =========================================================
# メイン
# =========================================================
def main():
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    all_rmse_rows   = []
    all_detail_rows = []

    for mask_pattern in MASK_PATTERNS:
        print(f"\n{'='*60}")
        print(f"マスクパターン: {mask_pattern}")
        print(f"{'='*60}")

        # テストデータ読み込み
        test_dfs = {}
        for mat_name, mat_dir_name in MATERIAL_DIRS.items():
            namelist_path = DATA_ROOT / mat_dir_name / "namelist.csv"
            df = load_test_data(namelist_path, mask_pattern)
            if len(df) > 0:
                test_dfs[mat_name] = df
                # サンプルパス確認用
                print(f"  {mat_name}: {len(df)} samples")
                print(f"    サンプルパス: {df['img_path'].iloc[0]}")
            else:
                print(f"  [WARN] テストデータなし: {mat_name}")

        # 個別モデル × 全テスト素材
        for train_mat in MATERIAL_DIRS.keys():
            model_dir = RESULT_ROOT / f"{train_mat}_{mask_pattern}"
            model = load_model(model_dir)
            if model is None:
                print(f"  [SKIP] モデルなし: {model_dir.name}")
                continue

            print(f"\n  学習素材: {train_mat}")

            for test_mat, test_df in test_dfs.items():
                eval_type = "素材内" if train_mat == test_mat else "素材間"
                y_true = test_df[["Fz", "Fx", "Fy"]].values.astype("float32")
                y_pred = predict_batch(model, test_df)
                rmse   = calc_rmse(y_true, y_pred)

                print(f"    → {test_mat} ({eval_type})  "
                      f"Fz={rmse['RMSE_Fz']:.3f}  "
                      f"Fx={rmse['RMSE_Fx']:.3f}  "
                      f"Fy={rmse['RMSE_Fy']:.3f}")

                all_rmse_rows.append({
                    "mask_pattern":   mask_pattern,
                    "model":          f"{train_mat}_{mask_pattern}",
                    "train_material": train_mat,
                    "test_material":  test_mat,
                    "eval_type":      eval_type,
                    "n_test":         len(test_df),
                    **rmse
                })

                for i in range(len(test_df)):
                    all_detail_rows.append({
                        "mask_pattern":   mask_pattern,
                        "train_material": train_mat,
                        "test_material":  test_mat,
                        "eval_type":      eval_type,
                        "ifuku_id":       int(test_df.iloc[i]["ifuku_id"]),
                        "Fz_true":  float(y_true[i, 0]),
                        "Fx_true":  float(y_true[i, 1]),
                        "Fy_true":  float(y_true[i, 2]),
                        "Fz_pred":  float(y_pred[i, 0]),
                        "Fx_pred":  float(y_pred[i, 1]),
                        "Fy_pred":  float(y_pred[i, 2]),
                        "Fz_error": float(y_pred[i, 0] - y_true[i, 0]),
                        "Fx_error": float(y_pred[i, 1] - y_true[i, 1]),
                        "Fy_error": float(y_pred[i, 2] - y_true[i, 2]),
                    })

            del model
            gc.collect()

        # 全素材混合モデル × 全テスト素材
        model_dir = RESULT_ROOT / f"all_materials_{mask_pattern}"
        model = load_model(model_dir)
        if model is not None:
            print(f"\n  学習素材: all_materials")
            for test_mat, test_df in test_dfs.items():
                y_true = test_df[["Fz", "Fx", "Fy"]].values.astype("float32")
                y_pred = predict_batch(model, test_df)
                rmse   = calc_rmse(y_true, y_pred)

                print(f"    → {test_mat} (混合)  "
                      f"Fz={rmse['RMSE_Fz']:.3f}  "
                      f"Fx={rmse['RMSE_Fx']:.3f}  "
                      f"Fy={rmse['RMSE_Fy']:.3f}")

                all_rmse_rows.append({
                    "mask_pattern":   mask_pattern,
                    "model":          f"all_materials_{mask_pattern}",
                    "train_material": "all_materials",
                    "test_material":  test_mat,
                    "eval_type":      "混合モデル",
                    "n_test":         len(test_df),
                    **rmse
                })

                for i in range(len(test_df)):
                    all_detail_rows.append({
                        "mask_pattern":   mask_pattern,
                        "train_material": "all_materials",
                        "test_material":  test_mat,
                        "eval_type":      "混合モデル",
                        "ifuku_id":       int(test_df.iloc[i]["ifuku_id"]),
                        "Fz_true":  float(y_true[i, 0]),
                        "Fx_true":  float(y_true[i, 1]),
                        "Fy_true":  float(y_true[i, 2]),
                        "Fz_pred":  float(y_pred[i, 0]),
                        "Fx_pred":  float(y_pred[i, 1]),
                        "Fy_pred":  float(y_pred[i, 2]),
                        "Fz_error": float(y_pred[i, 0] - y_true[i, 0]),
                        "Fx_error": float(y_pred[i, 1] - y_true[i, 1]),
                        "Fy_error": float(y_pred[i, 2] - y_true[i, 2]),
                    })

            del model
            gc.collect()

    # =========================================================
    # 保存
    # =========================================================
    rmse_df   = pd.DataFrame(all_rmse_rows)
    detail_df = pd.DataFrame(all_detail_rows)

    rmse_df.to_csv(EVAL_ROOT / "rmse_summary.csv",        index=False, encoding="utf-8-sig")
    detail_df.to_csv(EVAL_ROOT / "detail_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\nCSV保存: {EVAL_ROOT / 'rmse_summary.csv'}")
    print(f"CSV保存: {EVAL_ROOT / 'detail_predictions.csv'}")

    excel_path = EVAL_ROOT / "evaluation_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        rmse_df.to_excel(writer, sheet_name="RMSE_summary", index=False)
        rmse_df.groupby(["mask_pattern", "eval_type"])[
            ["RMSE_Fz", "RMSE_Fx", "RMSE_Fy"]
        ].mean().round(4).to_excel(writer, sheet_name="RMSE_by_eval_type")
        for mask_pattern in MASK_PATTERNS:
            df_mask = detail_df[detail_df["mask_pattern"] == mask_pattern].copy()
            df_mask.to_excel(writer, sheet_name=f"detail_{mask_pattern}"[:31], index=False)
        for mask_pattern in MASK_PATTERNS:
            df_mask = rmse_df[
                (rmse_df["mask_pattern"] == mask_pattern) &
                (rmse_df["train_material"] != "all_materials")
            ].copy()
            if len(df_mask) == 0:
                continue
            pivot = df_mask.pivot_table(
                index="train_material",
                columns="test_material",
                values="RMSE_Fz",
                aggfunc="mean"
            ).round(4)
            pivot.to_excel(writer, sheet_name=f"Fz_pivot_{mask_pattern}"[:31])

    print(f"Excel保存: {excel_path}")
    print("\n=== 評価完了 ===")
    print("\n=== RMSE サマリー ===")
    print(rmse_df.groupby(["mask_pattern", "eval_type"])[
        ["RMSE_Fz", "RMSE_Fx", "RMSE_Fy"]
    ].mean().round(3).to_string())


if __name__ == "__main__":
    main()