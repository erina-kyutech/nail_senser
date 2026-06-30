# -*- coding: utf-8 -*-
"""
check_duplicate_frames.py

撮影された連番画像（0.png, 1.png, 2.png, ...）について、
隣り合うフレーム同士が完全に同じ内容かどうかを調べる。

カメラのFPS上限を超えてループが回っていると、
同じカメラフレームが複数回保存されている可能性がある。
このスクリプトはその重複がどれくらいあるかを定量的に確認する。
"""

import os
import cv2
import numpy as np
import hashlib
import csv

# =========================================================
# 設定（ここを調べたいフォルダに変更）
# =========================================================
TARGET_DIR = r"C:\Users\Owner\PycharmProjects\datas\record0-10xyz\ifuku3\360deg"
OUTPUT_CSV = os.path.join(TARGET_DIR, "duplicate_check_result.csv")


def get_image_hash(img):
    """画像の内容からハッシュ値を計算する（完全一致判定用）"""
    return hashlib.md5(img.tobytes()).hexdigest()


def list_numbered_images(image_dir):
    """0.png, 1.png, ... の順にソートして返す"""
    files = []
    for name in os.listdir(image_dir):
        if name.lower().endswith(".png"):
            stem = os.path.splitext(name)[0]
            if stem.isdigit():
                files.append((int(stem), os.path.join(image_dir, name)))
    files.sort(key=lambda x: x[0])
    return files


def main():
    print("=== 対象フォルダ ===")
    print(TARGET_DIR)

    if not os.path.exists(TARGET_DIR):
        raise FileNotFoundError(f"フォルダが見つかりません: {TARGET_DIR}")

    images = list_numbered_images(TARGET_DIR)
    print(f"画像枚数: {len(images)}")

    if len(images) < 2:
        print("画像が少なすぎて比較できません。")
        return

    prev_hash = None
    prev_idx = None
    duplicate_count = 0
    duplicate_pairs = []

    rows = []

    for idx, path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 読み込み失敗: {path}")
            continue

        h = get_image_hash(img)

        is_duplicate = (h == prev_hash)
        if is_duplicate:
            duplicate_count += 1
            duplicate_pairs.append((prev_idx, idx))

        rows.append({
            "index": idx,
            "hash": h,
            "is_duplicate_of_prev": is_duplicate
        })

        prev_hash = h
        prev_idx = idx

        if (idx + 1) % 500 == 0:
            print(f"  チェック中... {idx + 1}/{len(images)}")

    # ── 結果まとめ ──────────────────────────────────────
    total = len(images)
    dup_rate = duplicate_count / total * 100

    print("\n=== 結果 ===")
    print(f"総画像枚数:       {total}")
    print(f"重複フレーム数:    {duplicate_count}")
    print(f"重複率:           {dup_rate:.1f}%")
    print(f"ユニーク画像枚数:  {total - duplicate_count}")

    if duplicate_pairs:
        print("\n=== 重複ペアの例（最初の10件） ===")
        for prev_i, cur_i in duplicate_pairs[:10]:
            print(f"  {prev_i}.png  ==  {cur_i}.png")

    # ── CSV保存 ──────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "hash", "is_duplicate_of_prev"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n詳細結果を保存しました: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()