# -*- coding: utf-8 -*-
"""
train_material_classifier_all_patterns.py

素材分類器を学習・評価するスクリプト（nail_and_tip / nail_only / tip_only の3パターン統合版）。
VGG16ベースで4クラス（felt/acrylic/paper/aluminum）を分類する。

3パターンを順番に処理し、それぞれ:
  - 既に重みがあれば学習をスキップして読み込みのみ
  - なければ ifuku1〜30 で学習・val分割して学習
  - ifuku31〜35（完全ホールドアウト、重み更新なし）でテスト評価
を行い、最後に3パターンのテスト精度を比較する表とグラフを出力する。

保存先（パターンごとに分離）：
  result/CNN_result/material_classifier_nail_and_tip/
  result/CNN_result/material_classifier_nail_only/
  result/CNN_result/material_classifier_tip_only/
  result/CNN_result/material_classifier_comparison/   ← 3パターン比較結果
"""

from __future__ import annotations

import gc
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# 設定
# =========================================================
RANDOM_SEED = 42
IMG_H = 150
IMG_C = 3
BATCH_SIZE = 32
EPOCHS     = 10
VAL_RATIO  = 0.15

# 3パターンをここで順番に処理する
MASK_PATTERNS = ["nail_and_tip", "nail_only", "tip_only"]

# マスクパターンごとの画像幅（高さは共通150）
IMG_W_MAP = {
    "nail_and_tip": 290,
    "nail_only":    150,
    "tip_only":     140,
}

# 学習・バリデーションに使うifuku番号範囲
TRAIN_IFUKU_START = 1
TRAIN_IFUKU_END   = 30

# テスト（完全ホールドアウト、重み更新なし）に使うifuku番号範囲
TEST_IFUKU_START = 31
TEST_IFUKU_END   = 35

PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")
DATA_ROOT    = PROJECT_ROOT / "datas"
RESULT_BASE  = PROJECT_ROOT / "result" / "CNN_result"
COMPARISON_DIR = RESULT_BASE / "material_classifier_comparison"

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
MATERIAL_NAMES = list(LABEL_MAP.keys())


# =========================================================
# ユーティリティ
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_japanese_font():
    """Windows環境向けに日本語フォントを試行設定（無ければ既定のまま）"""
    for font_name in ["Yu Gothic", "MS Gothic", "Meiryo"]:
        try:
            plt.rcParams["font.family"] = font_name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def parse_ifuku_id(path_str):
    m = re.search(r'ifuku(\d+)', str(path_str))
    return int(m.group(1)) if m else None


def resolve_img_path(path_str: str, mask_pattern: str) -> Path:
    """datalog.csvのパスを_dedup_masked/{mask_pattern}のパスに変換"""
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

    # 360degの直下にmask_patternを挿入
    try:
        deg_idx = parts.index("360deg")
        parts.insert(deg_idx + 1, mask_pattern)
    except ValueError:
        pass

    return Path(*parts)


# =========================================================
# データ読み込み
# =========================================================
def load_material_data(mat_name: str, mat_dir_name: str,
                       ifuku_start: int, ifuku_end: int, mask_pattern: str) -> pd.DataFrame:
    """1素材分のデータを読み込んでラベルを付ける"""
    namelist_path = DATA_ROOT / mat_dir_name / "namelist.csv"

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
            lambda p: str(resolve_img_path(p, mask_pattern))
        )
        df["label"]    = LABEL_MAP[mat_name]
        df["ifuku_id"] = ifuku_id

        all_df = pd.concat([
            all_df,
            df[["img_path", "label", "ifuku_id"]]
        ], ignore_index=True)

    if len(all_df) > 0:
        # 空DataFrameとのconcatでobject型に引きずられるのを防ぐため明示的にキャスト
        all_df["label"] = all_df["label"].astype(int)
        all_df["ifuku_id"] = all_df["ifuku_id"].astype(int)

    return all_df


def load_all_materials(ifuku_start: int, ifuku_end: int, mask_pattern: str, tag: str = "") -> pd.DataFrame:
    """全素材分をまとめて読み込む（学習用・テスト用どちらにも使う共通関数）"""
    all_dfs = []
    for mat_name, mat_dir_name in MATERIAL_DIRS.items():
        df = load_material_data(mat_name, mat_dir_name, ifuku_start, ifuku_end, mask_pattern)
        if len(df) > 0:
            print(f"  [{tag}] {mat_name}: {len(df)} samples")
            all_dfs.append(df)
        else:
            print(f"  [WARN][{tag}] {mat_name}: 0 samples (ifuku{ifuku_start}~{ifuku_end})")

    if len(all_dfs) == 0:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


# =========================================================
# Sequence
# =========================================================
class ClassifierSequence(tf.keras.utils.Sequence):
    def __init__(self, df: pd.DataFrame, img_w: int, batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.img_w = img_w
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
                x_list.append(np.zeros((IMG_H, self.img_w, IMG_C), dtype=np.uint8))
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != IMG_H or rgb.shape[1] != self.img_w:
                rgb = cv2.resize(rgb, (self.img_w, IMG_H))
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
def build_classifier(img_w: int) -> Model:
    inp  = Input(shape=(IMG_H, img_w, IMG_C), name="input_tensor")
    conv = VGG16(weights="imagenet",
                 input_shape=(IMG_H, img_w, IMG_C),
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
# テスト評価（ifuku31〜35、完全ホールドアウト）
# =========================================================
def evaluate_on_test(model: Model, test_df: pd.DataFrame, result_root: Path,
                      mask_pattern: str, img_w: int) -> float:
    """テスト精度（accuracy）を返す。比較サマリーに使う。"""
    if len(test_df) == 0:
        print("[WARN] テストデータが空のため評価をスキップします")
        return float("nan")

    setup_japanese_font()

    test_seq = ClassifierSequence(test_df, img_w=img_w, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n=== テスト評価 (ifuku{TEST_IFUKU_START}~{TEST_IFUKU_END}, {len(test_df)} samples) ===")
    y_pred_proba = model.predict(test_seq, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    n_pred = len(y_pred)
    y_true = test_df["label"].astype(int).values[:n_pred]

    overall_acc = float((y_pred == y_true).mean())
    print(f"\ntest accuracy (ifuku{TEST_IFUKU_START}~{TEST_IFUKU_END}, {mask_pattern}): {overall_acc:.4f}")

    report_dict = classification_report(
        y_true, y_pred, target_names=MATERIAL_NAMES, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = result_root / "test_classification_report.csv"
    report_df.to_csv(report_csv_path, encoding="utf-8-sig")
    print(classification_report(y_true, y_pred, target_names=MATERIAL_NAMES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = np.divide(
        cm.astype("float"), cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype="float"),
        where=cm.sum(axis=1, keepdims=True) != 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title, fmt in [
        (axes[0], cm, "混同行列（件数）", "d"),
        (axes[1], cm_norm, "混同行列（正規化）", ".2f"),
    ]:
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(MATERIAL_NAMES, rotation=45)
        ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(MATERIAL_NAMES)
        ax.set_xlabel("予測"); ax.set_ylabel("真値")
        ax.set_title(title)
        vmax = mat.max() if mat.max() > 0 else 1
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt = f"{val:{fmt}}"
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"素材分類器 テスト混同行列 (ifuku{TEST_IFUKU_START}~{TEST_IFUKU_END}, input={mask_pattern})")
    plt.tight_layout()
    cm_fig_path = result_root / "test_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    detail_df = test_df.iloc[:n_pred].copy()
    detail_df["pred_label"] = y_pred
    detail_df["pred_material"] = [MATERIAL_NAMES[i] for i in y_pred]
    for i, m in enumerate(MATERIAL_NAMES):
        detail_df[f"proba_{m}"] = y_pred_proba[:, i]
    detail_csv_path = result_root / "test_predictions_detail.csv"
    detail_df.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")

    print(f"保存: {report_csv_path}")
    print(f"保存: {cm_fig_path}")
    print(f"保存: {detail_csv_path}")

    return overall_acc


# =========================================================
# 1マスクパターン分の学習〜評価を実行
# =========================================================
def run_for_mask_pattern(mask_pattern: str) -> float:
    print("\n" + "=" * 70)
    print(f"### マスクパターン: {mask_pattern} ###")
    print("=" * 70)

    img_w = IMG_W_MAP[mask_pattern]
    result_root = RESULT_BASE / f"material_classifier_{mask_pattern}"
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "weights").mkdir(exist_ok=True)

    weight_path = result_root / "weights" / "classifier_weights.h5"
    json_path = result_root / "classifier.json"

    model = build_classifier(img_w)

    if weight_path.exists():
        print(f"[SKIP] 既存の重みを読み込みます: {weight_path}")
        model.load_weights(str(weight_path))
    else:
        print(f"=== データ読み込み（学習/val: ifuku{TRAIN_IFUKU_START}~{TRAIN_IFUKU_END}） ===")
        combined_df = load_all_materials(TRAIN_IFUKU_START, TRAIN_IFUKU_END, mask_pattern, tag="train/val")

        if len(combined_df) == 0:
            print(f"[ERROR] {mask_pattern}: 学習データが読み込めませんでした。スキップします。")
            del model
            gc.collect()
            return float("nan")

        print(f"  合計: {len(combined_df)} samples")

        all_ids = sorted(combined_df["ifuku_id"].unique().tolist())
        random.Random(RANDOM_SEED).shuffle(all_ids)
        val_size  = max(1, int(len(all_ids) * VAL_RATIO))
        val_ids   = sorted(all_ids[:val_size])
        train_ids = sorted(all_ids[val_size:])

        train_df = combined_df[combined_df["ifuku_id"].isin(train_ids)].reset_index(drop=True)
        val_df   = combined_df[combined_df["ifuku_id"].isin(val_ids)].reset_index(drop=True)

        print(f"\n  train: {len(train_df)} samples ({len(train_ids)} sessions)")
        print(f"  val  : {len(val_df)} samples ({len(val_ids)} sessions)")

        train_seq = ClassifierSequence(train_df, img_w=img_w, batch_size=BATCH_SIZE, shuffle=True)
        val_seq   = ClassifierSequence(val_df,   img_w=img_w, batch_size=BATCH_SIZE, shuffle=False)

        print("\n=== 学習開始 ===")
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
            result_root / "learning_log.csv",
            index=False, encoding="utf-8-sig"
        )

        val_loss, val_acc = model.evaluate(val_seq, verbose=0)
        print(f"\nval accuracy: {val_acc:.4f}")
        print(f"val loss    : {val_loss:.4f}")

        label_info = {
            "label_map": LABEL_MAP,
            "id_to_material": {v: k for k, v in LABEL_MAP.items()},
            "mask_pattern": mask_pattern,
            "img_w": img_w,
        }
        with open(result_root / "label_info.json", "w", encoding="utf-8") as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)
        print(f"ラベル情報保存: {result_root / 'label_info.json'}")

        del train_seq, val_seq
        gc.collect()

    print(f"\n=== データ読み込み（テスト: ifuku{TEST_IFUKU_START}~{TEST_IFUKU_END}） ===")
    test_df = load_all_materials(TEST_IFUKU_START, TEST_IFUKU_END, mask_pattern, tag="test")
    test_acc = evaluate_on_test(model, test_df, result_root, mask_pattern, img_w)

    del model
    gc.collect()

    return test_acc


# =========================================================
# 3パターン比較サマリー
# =========================================================
def save_comparison(results: dict) -> None:
    setup_japanese_font()
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [{"mask_pattern": k, "test_accuracy": v} for k, v in results.items()]
    )
    summary_csv = COMPARISON_DIR / "mask_pattern_comparison.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print("\n=== 3パターン比較 ===")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#55A868", "#DD8452"]
    bars = ax.bar(summary_df["mask_pattern"], summary_df["test_accuracy"], color=colors)
    for b, v in zip(bars, summary_df["test_accuracy"]):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}",
                    ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("テスト精度 (ifuku31~35, accuracy)")
    ax.set_title("素材分類器: マスクパターン別 精度比較")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = COMPARISON_DIR / "mask_pattern_comparison.png"
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\n保存: {summary_csv}")
    print(f"保存: {fig_path}")


# =========================================================
# メイン
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    RESULT_BASE.mkdir(parents=True, exist_ok=True)

    results = {}
    for mask_pattern in MASK_PATTERNS:
        acc = run_for_mask_pattern(mask_pattern)
        results[mask_pattern] = acc

    save_comparison(results)

    print("\n=== 全パターン完了 ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if not np.isnan(v) else f"  {k}: 失敗/データなし")


if __name__ == "__main__":
    main()