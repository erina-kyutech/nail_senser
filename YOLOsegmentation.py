# -*- coding: utf-8 -*-
"""
LabelMe (finger_tip / nail ポリゴン, 点数は画像ごとに可変) の JSONを
YOLOv11-seg 用のデータセット形式に変換するスクリプト。

入力:
  PROJECT_ROOT/datas/LabelMe/finger_tip_and_nail/*.json
    各JSONの imagePath が ../../record0-10xyz/ifukuN/360deg/M.png を指す

出力:
  OUTPUT_ROOT/
    images/train/*.png
    images/val/*.png
    labels/train/*.txt
    labels/val/*.txt
    data.yaml

使い方:
  python labelme_to_yoloseg.py
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

# =========================================================
# 設定
# =========================================================
PROJECT_ROOT = Path(r"C:\Users\Owner\PycharmProjects")

# LabelMe JSONが入っているフォルダ
LABELME_DIR = PROJECT_ROOT / "datas" / "LabelMe" / "finger_tip_and_nail"

# 出力先(このスクリプトを置く場所の下に作る)
OUTPUT_ROOT = PROJECT_ROOT / "Tactile_sensors_20230929" / "yolo_seg_dataset"

# クラス定義 (順番がそのままクラスID 0,1,... になる)
CLASS_NAMES = ["finger_tip", "nail"]
CLASS_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# train/val 分割比率 (val側の割合)
VAL_RATIO = 0.15
RANDOM_SEED = 42

# 画像サイズ (LabelMeのJSON内 imageWidth/imageHeight と一致する想定)
EXPECTED_W = 290
EXPECTED_H = 150


# =========================================================
# ユーティリティ
# =========================================================
def resolve_image_path(json_path: Path, image_path_in_json: str) -> Path:
    """
    LabelMe JSON内の imagePath (相対パス、json基準) から
    実際の画像の絶対パスを解決する。
    """
    # JSON内のパスは json ファイルが置かれている場所からの相対パス
    rel = Path(image_path_in_json.replace("\\", "/"))
    img_path = (json_path.parent / rel).resolve()
    return img_path


def polygon_to_yolo_line(class_id: int, points: list[list[float]], img_w: int, img_h: int) -> str:
    """
    1つのshape(ポリゴン)を YOLO-seg の1行のフォーマットに変換する。
    フォーマット: class_id x1 y1 x2 y2 ... xn yn (0~1正規化)
    """
    coords = []
    for x, y in points:
        nx = min(max(x / img_w, 0.0), 1.0)
        ny = min(max(y / img_h, 0.0), 1.0)
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    return f"{class_id} " + " ".join(coords)


def collect_samples(labelme_dir: Path):
    """
    LabelMe JSON一覧を読み込み、(json_path, image_path, shapes, w, h) のリストを返す。
    """
    samples = []
    json_files = sorted(
        labelme_dir.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
    )

    if not json_files:
        raise FileNotFoundError(f"JSONファイルが見つかりません: {labelme_dir}")

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth", EXPECTED_W)
        img_h = data.get("imageHeight", EXPECTED_H)

        if img_w != EXPECTED_W or img_h != EXPECTED_H:
            print(f"[WARN] {json_path.name}: imageWidth/Height が想定と異なります "
                  f"({img_w}x{img_h}, 期待値 {EXPECTED_W}x{EXPECTED_H})")

        image_path_in_json = data.get("imagePath", "")
        img_path = resolve_image_path(json_path, image_path_in_json)

        if not img_path.exists():
            print(f"[WARN] 画像が見つかりません(スキップ): {img_path} (from {json_path.name})")
            continue

        shapes = data.get("shapes", [])

        # 未知のラベルがあれば警告
        valid_shapes = []
        for s in shapes:
            label = s.get("label")
            if label not in CLASS_ID:
                print(f"[WARN] {json_path.name}: 未知のラベル '{label}' をスキップします")
                continue
            if s.get("shape_type") != "polygon":
                print(f"[WARN] {json_path.name}: shape_type='{s.get('shape_type')}' "
                      f"(polygon以外)はスキップします")
                continue
            if len(s.get("points", [])) < 3:
                print(f"[WARN] {json_path.name}: ラベル '{label}' の点数が3未満のためスキップ")
                continue
            valid_shapes.append(s)

        if not valid_shapes:
            print(f"[WARN] {json_path.name}: 有効なポリゴンが無いためスキップ")
            continue

        samples.append({
            "json_path": json_path,
            "img_path": img_path,
            "shapes": valid_shapes,
            "img_w": img_w,
            "img_h": img_h,
        })

    return samples


def write_label_file(label_path: Path, shapes: list[dict], img_w: int, img_h: int):
    lines = []
    for s in shapes:
        class_id = CLASS_ID[s["label"]]
        line = polygon_to_yolo_line(class_id, s["points"], img_w, img_h)
        lines.append(line)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_unique_stem(json_path: Path, img_path: Path) -> str:
    """
    train/val内でファイル名が重複しないように、
    'json番号_ifukuID_画像番号' の形式にする。
    例: 0_ifuku1_0
    """
    # img_path: .../ifukuN/360deg/M.png
    try:
        ifuku_name = img_path.parts[-3]  # ifukuN
    except IndexError:
        ifuku_name = "unknown"

    img_stem = img_path.stem  # M
    json_stem = json_path.stem  # 0,1,2,...

    return f"{json_stem}_{ifuku_name}_{img_stem}"


def ensure_dirs():
    for split in ("train", "val"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))

    content = (
        f"path: {OUTPUT_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n"
    )

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[INFO] data.yaml を書き出しました -> {yaml_path}")


# =========================================================
# メイン
# =========================================================
def main():
    random.seed(RANDOM_SEED)

    print("=== LabelMe JSON読み込み ===")
    samples = collect_samples(LABELME_DIR)
    print(f"有効サンプル数: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("有効なサンプルが1件もありません。パス設定を確認してください。")

    # シャッフルしてtrain/val分割
    indices = list(range(len(samples)))
    random.shuffle(indices)

    val_size = max(1, int(len(indices) * VAL_RATIO))
    val_indices = set(indices[:val_size])

    ensure_dirs()

    train_count = 0
    val_count = 0

    for i, sample in enumerate(samples):
        split = "val" if i in val_indices else "train"

        stem = make_unique_stem(sample["json_path"], sample["img_path"])

        # 画像コピー(拡張子は元画像のものを使う)
        ext = sample["img_path"].suffix
        out_img_path = OUTPUT_ROOT / "images" / split / f"{stem}{ext}"
        shutil.copy2(sample["img_path"], out_img_path)

        # ラベル書き出し
        out_label_path = OUTPUT_ROOT / "labels" / split / f"{stem}.txt"
        write_label_file(out_label_path, sample["shapes"], sample["img_w"], sample["img_h"])

        if split == "train":
            train_count += 1
        else:
            val_count += 1

    write_data_yaml()

    print("=== 完了 ===")
    print(f"train: {train_count} 枚")
    print(f"val  : {val_count} 枚")
    print(f"出力先: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()