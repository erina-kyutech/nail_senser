# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json


# =========================================================
# 設定
# =========================================================
PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")
SUBJECT_NAME = "ifuku"

# --- テスト画像フォルダ（表示に使う元画像） ---
IMAGE_DIR = PROJECT_ROOT / "datas" / "record0-10xyz" / "ifuku167" / "360deg"

# --- 画像選択元CSV（ここからランダムに選ぶ） ---
SELECT_CSV = (
    PROJECT_ROOT
    / "result"
    / "CNN_result"
    / "gradcam_確かめテスト"
    / "左端黒塗りバージョン_20px"
    / "csv"
    / "test_predictions.csv"
)

# --- 3つのモデル保存先 ---
MODEL_DIRS = {
    "original": PROJECT_ROOT / "result" / "CNN_result" / "gradcam_確かめテスト" / "何もしてないバージョン",
    "mean_replace": PROJECT_ROOT / "result" / "CNN_result" / "gradcam_確かめテスト" / "左側平均画像バージョン",
    "black_mask": PROJECT_ROOT / "result" / "CNN_result" / "gradcam_確かめテスト" / "左端黒塗りバージョン_20px",
}

# --- 保存先 ---
SAVE_DIR = PROJECT_ROOT / "result" / "CNN_result" / "gradcam_確かめテスト" / "gradcam_from_testcsv_random10_fixed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- 入力サイズ ---
IMG_H = 150
IMG_W = 290

# --- 左側処理幅 ---
LEFT_PIXELS = 20

# --- 平均左端を作る train 側 ifuku 範囲 ---
TRAIN_IFUKU_START = 1
TRAIN_IFUKU_END = 166

# --- ランダム抽出 ---
N_SAMPLES = 10
RANDOM_SEED = 42

# --- Grad-CAM対象 ---
# vgg16出力を flatten が受け取っているので、flatten.input を使う
FEATURE_TENSOR_LAYER_NAME = "flatten"

OUTPUT_NAMES = ["Fz", "Fx", "Fy"]


# =========================================================
# 日本語パス対応
# =========================================================
def imread_japanese(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_japanese(path: Path, img: np.ndarray):
    ext = path.suffix if path.suffix else ".png"
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"failed to encode image: {path}")
    enc.tofile(str(path))


# =========================================================
# 画像読み込み
# =========================================================
def build_input_image_from_path(img_path: Path) -> np.ndarray:
    bgr = imread_japanese(img_path)
    if bgr is None:
        raise FileNotFoundError(f"image not found: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if rgb.shape[0] != IMG_H or rgb.shape[1] != IMG_W:
        rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    return rgb


def preprocess_for_model(img_rgb: np.ndarray) -> np.ndarray:
    x = img_rgb.astype(np.float32) / 255.0
    return x[None, ...]


# =========================================================
# 左側処理
# =========================================================
def apply_black_mask(img_rgb: np.ndarray, left_pixels: int = 20) -> np.ndarray:
    out = img_rgb.copy()
    out[:, :left_pixels, :] = 0
    return out


def apply_mean_left_replace(img_rgb: np.ndarray, mean_left_patch: np.ndarray, left_pixels: int = 20) -> np.ndarray:
    out = img_rgb.copy()
    out[:, :left_pixels, :] = mean_left_patch
    return out


# =========================================================
# 平均左端パッチ作成
# =========================================================
def collect_train_image_paths(base_dir: Path, start_id: int = 1, end_id: int = 166):
    image_paths = []
    for ifuku_id in range(start_id, end_id + 1):
        one_dir = base_dir / f"ifuku{ifuku_id}" / "360deg"
        if not one_dir.exists():
            print(f"[WARN] not found: {one_dir}")
            continue

        paths = sorted(one_dir.glob("*.png"))
        image_paths.extend(paths)

    return image_paths


def build_mean_left_patch_from_paths(image_paths, left_pixels: int = 20, max_samples: int | None = None):
    if len(image_paths) == 0:
        raise FileNotFoundError("no training images found for mean-left patch")

    if max_samples is not None and len(image_paths) > max_samples:
        random.seed(RANDOM_SEED)
        image_paths = random.sample(image_paths, max_samples)

    acc = np.zeros((IMG_H, left_pixels, 3), dtype=np.float64)

    for i, p in enumerate(image_paths):
        img = build_input_image_from_path(p)
        acc += img[:, :left_pixels, :].astype(np.float64)

        if (i + 1) % 1000 == 0:
            print(f"[mean left patch] {i + 1}/{len(image_paths)}")

    mean_left_patch = np.clip(acc / len(image_paths), 0, 255).astype(np.uint8)
    return mean_left_patch


# =========================================================
# モデル読み込み
# =========================================================
def find_weight_path(model_dir: Path, subject_name: str) -> Path:
    candidates = [
        model_dir / "weights" / f"weight_{subject_name}_for0-10.h5",
        model_dir / "weight" / f"weight_{subject_name}_for0-10.h5",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "weight file not found. checked:\n" + "\n".join(str(p) for p in candidates)
    )


def load_model_from_dir(model_dir: Path, subject_name: str):
    json_path = model_dir / "for0-10.json"
    weight_path = find_weight_path(model_dir, subject_name)

    if not json_path.exists():
        raise FileNotFoundError(f"json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        model_json_string = f.read()

    model = model_from_json(model_json_string)
    model.load_weights(str(weight_path))
    return model


# =========================================================
# 予測値（逆正規化）
# =========================================================
def predict_force_physical(model, x: np.ndarray):
    y_list = model(x, training=False)

    y = np.concatenate([
        y_list[0].numpy(),  # Fz
        y_list[1].numpy(),  # Fx
        y_list[2].numpy(),  # Fy
    ], axis=1)

    y[:, 0] *= 10.0
    y[:, 1] = y[:, 1] * 10.0 - 5.0
    y[:, 2] = y[:, 2] * 10.0 - 5.0

    return y[0, 0], y[0, 1], y[0, 2]


# =========================================================
# Grad-CAM
# =========================================================
def build_grad_model(model, feature_tensor_layer_name: str):
    """
    Graph disconnected回避のため、
    nestedなvgg16出力ではなく flatten.input を使う
    """
    layer = model.get_layer(feature_tensor_layer_name)
    feature_tensor = layer.input   # ← flatten の入力 = (None, 4, 9, 512)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[feature_tensor, model.outputs]
    )
    return grad_model


def make_gradcam_heatmap(grad_model, img_array, output_index: int):
    with tf.GradientTape() as tape:
        feature_maps, preds = grad_model(img_array, training=False)

        # preds = [Fz, Fx, Fy]
        target = preds[output_index]
        target = tf.reduce_sum(target)  # スカラー化

    grads = tape.gradient(target, feature_maps)

    feature_maps = feature_maps[0]   # (4, 9, 512)
    grads = grads[0]                 # (4, 9, 512)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))             # (512,)
    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1) # (4, 9)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap_on_image(img_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45):
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# =========================================================
# CSVから画像解決
# =========================================================
def resolve_image_path_from_row(row, image_dir: Path) -> Path:
    if "img_path" in row.index:
        p = Path(str(row["img_path"]))
        return image_dir / p.name

    if "image_index" in row.index:
        return image_dir / f"{int(row['image_index'])}.png"

    raise KeyError("CSVに img_path も image_index もありません。")


# =========================================================
# メイン
# =========================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=== check paths ===")
    print("IMAGE_DIR :", IMAGE_DIR)
    print("SELECT_CSV:", SELECT_CSV)
    print("exists IMAGE_DIR :", IMAGE_DIR.exists())
    print("exists SELECT_CSV:", SELECT_CSV.exists())

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {IMAGE_DIR}")
    if not SELECT_CSV.exists():
        raise FileNotFoundError(f"SELECT_CSV not found: {SELECT_CSV}")

    # -------------------------------
    # CSV読み込み
    # -------------------------------
    df = pd.read_csv(SELECT_CSV)

    print("=== csv columns ===")
    print(df.columns.tolist())

    if len(df) == 0:
        raise ValueError("test_predictions.csv が空です。")

    n_pick = min(N_SAMPLES, len(df))
    selected_df = df.sample(n=n_pick, random_state=RANDOM_SEED).reset_index(drop=True)

    print("selected image count:", len(selected_df))

    selected_txt = SAVE_DIR / "selected_images.txt"
    with open(selected_txt, "w", encoding="utf-8") as f:
        for i in range(len(selected_df)):
            img_path = resolve_image_path_from_row(selected_df.loc[i], IMAGE_DIR)
            f.write(str(img_path) + "\n")

    # -------------------------------
    # ifuku1〜166 の左端平均パッチを作る
    # -------------------------------
    print("=== collect train images for mean-left patch (ifuku1-166) ===")
    train_image_paths = collect_train_image_paths(
        PROJECT_ROOT / "datas" / "record0-10xyz",
        start_id=TRAIN_IFUKU_START,
        end_id=TRAIN_IFUKU_END
    )
    print("train image count for mean-left:", len(train_image_paths))

    mean_left_cache = SAVE_DIR / f"mean_left{LEFT_PIXELS}_ifuku1_166.png"

    if mean_left_cache.exists():
        print("=== load cached mean-left patch ===")
        mean_left_bgr = imread_japanese(mean_left_cache)
        if mean_left_bgr is None:
            raise FileNotFoundError(f"cannot read cached mean-left patch: {mean_left_cache}")
        mean_left_patch = cv2.cvtColor(mean_left_bgr, cv2.COLOR_BGR2RGB)
    else:
        print("=== build mean-left patch from ifuku1-166 ===")
        mean_left_patch = build_mean_left_patch_from_paths(
            train_image_paths,
            left_pixels=LEFT_PIXELS,
            max_samples=None
        )
        imwrite_japanese(mean_left_cache, cv2.cvtColor(mean_left_patch, cv2.COLOR_RGB2BGR))
        print(f"saved mean-left patch -> {mean_left_cache}")

    # -------------------------------
    # モデル読み込み
    # -------------------------------
    models = {}
    grad_models = {}

    for key, model_dir in MODEL_DIRS.items():
        print(f"loading model: {key}")
        model = load_model_from_dir(model_dir, SUBJECT_NAME)
        models[key] = model
        grad_models[key] = build_grad_model(model, FEATURE_TENSOR_LAYER_NAME)

    row_keys = ["original", "mean_replace", "black_mask"]
    row_titles = {
        "original": "何もしてない版",
        "mean_replace": "左側平均画像版",
        "black_mask": "左端黒塗り版_20px",
    }

    # -------------------------------
    # 1枚ずつGrad-CAM
    # -------------------------------
    for idx in range(len(selected_df)):
        row = selected_df.loc[idx]
        img_path = resolve_image_path_from_row(row, IMAGE_DIR)

        print("=" * 60)
        print(f"[{idx+1}/{len(selected_df)}] processing: {img_path.name}")

        original_img = build_input_image_from_path(img_path)

        input_images = {
            "original": original_img,
            "mean_replace": apply_mean_left_replace(original_img, mean_left_patch, left_pixels=LEFT_PIXELS),
            "black_mask": apply_black_mask(original_img, left_pixels=LEFT_PIXELS),
        }

        true_fz = row["Fz"] if "Fz" in row.index else np.nan
        true_fx = row["Fx"] if "Fx" in row.index else np.nan
        true_fy = row["Fy"] if "Fy" in row.index else np.nan

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))

        for r, key in enumerate(row_keys):
            model = models[key]
            grad_model = grad_models[key]
            img_rgb = input_images[key]
            x = preprocess_for_model(img_rgb)

            pred_fz, pred_fx, pred_fy = predict_force_physical(model, x)
            preds = [pred_fz, pred_fx, pred_fy]
            trues = [true_fz, true_fx, true_fy]

            # 入力画像
            ax0 = axes[r, 0]
            ax0.imshow(img_rgb)
            ax0.axis("off")
            ax0.set_title(f"{row_titles[key]}\ninput", fontsize=11)

            # Fz/Fx/Fy
            for c, out_name in enumerate(OUTPUT_NAMES, start=1):
                heatmap = make_gradcam_heatmap(
                    grad_model=grad_model,
                    img_array=x,
                    output_index=c - 1
                )

                overlay = overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.45)

                ax = axes[r, c]
                ax.imshow(overlay)
                ax.axis("off")
                ax.set_title(
                    f"{out_name}\ntrue={trues[c-1]:.3f}, pred={preds[c-1]:.3f}",
                    fontsize=10
                )

        plt.suptitle(f"{img_path.name}", fontsize=16)
        plt.tight_layout()

        save_path = SAVE_DIR / f"gradcam_compare_{img_path.stem}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"saved: {save_path}")

    print("done.")


if __name__ == "__main__":
    main()