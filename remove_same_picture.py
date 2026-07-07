# -*- coding: utf-8 -*-
"""
remove_duplicate_frames.py

撮影済みの連番画像（0.png, 1.png, ...）から、
隣接フレームと完全に同じ画像（重複フレーム）を検出し、
重複を除いた新しいデータセットを作成する。

元のフォルダは変更せず、別フォルダに「重複除去版」を出力する。
datalog.csvも重複を除いた行だけを抽出して作り直す。

複数素材をまとめて処理できる。
"""

import os
import shutil
import hashlib
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# 設定（ここを編集する）
# =========================================================
# 処理する素材フォルダのリスト（増えたら追加する）
SOURCE_ROOTS = [
    r"C:\Users\Owner\PycharmProjects\datas\felt_0-10xyz",
    r"C:\Users\Owner\PycharmProjects\datas\acrylic_0-10xyz",
    r"C:\Users\Owner\PycharmProjects\datas\paper_0-10xyz",
    r"C:\Users\Owner\PycharmProjects\datas\aluminum_0-10xyz",
    # r"C:\Users\Owner\PycharmProjects\datas\record0-10xyz",  # 元データも処理したい場合はコメントアウトを外す
]

# 対象とするifuku IDの範囲（Noneなら全部処理）
ID_START = None
ID_END   = None


# =========================================================
# ユーティリティ
# =========================================================
def get_image_hash(img):
    """画像の内容からハッシュ値を計算する（完全一致判定用）"""
    return hashlib.md5(img.tobytes()).hexdigest()


def list_numbered_images(image_dir: Path):
    """0.png, 1.png, ... の順にソートして返す"""
    files = []
    for name in os.listdir(image_dir):
        if name.lower().endswith(".png"):
            stem = os.path.splitext(name)[0]
            if stem.isdigit():
                files.append((int(stem), image_dir / name))
    files.sort(key=lambda x: x[0])
    return files


def find_session_dirs(source_root: Path):
    """
    SOURCE_ROOT 直下の ifuku1, ifuku2, ... のような
    撮影セッションフォルダ（360degを含むもの）を列挙する
    """
    session_dirs = []
    if not source_root.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {source_root}")

    for name in sorted(os.listdir(source_root)):
        session_path = source_root / name
        img_dir = session_path / "360deg"
        if img_dir.is_dir():
            # ifuku番号を取得してID範囲フィルタを適用
            digits = "".join(c for c in name if c.isdigit())
            if digits:
                num = int(digits)
                if ID_START is not None and num < ID_START:
                    continue
                if ID_END is not None and num > ID_END:
                    continue
            session_dirs.append((name, img_dir))

    return session_dirs


# =========================================================
# 1セッション分の重複除去
# =========================================================
def dedup_one_session(session_name: str, img_dir: Path, output_root: Path):
    """
    1セッション分（例: ifuku1/360deg）の重複除去を行う。
    重複していない画像だけを出力先にコピーし、
    新しいdatalog.csvを作る。
    """
    datalog_path = img_dir / "datalog.csv"
    if not datalog_path.exists():
        print(f"  [WARN] datalog.csv が見つかりません: {datalog_path}")
        return 0, 0

    df = pd.read_csv(datalog_path, header=0)
    df.columns = ["path", "Fz", "Fr", "Ff"]

    images = list_numbered_images(img_dir)
    if len(images) == 0:
        print(f"  [WARN] 画像が見つかりません: {img_dir}")
        return 0, 0

    out_session_dir = output_root / session_name / "360deg"
    out_session_dir.mkdir(parents=True, exist_ok=True)

    prev_hash = None
    kept_rows = []
    new_idx   = 0
    total     = len(images)

    for idx, path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue

        h = get_image_hash(img)

        if h == prev_hash:
            # 直前と完全一致 → スキップ
            prev_hash = h
            continue

        # ユニーク画像として採用 → 連番を振り直してコピー
        new_filename = f"{new_idx}.png"
        new_path = out_session_dir / new_filename
        shutil.copy2(str(path), str(new_path))

        # datalog.csvから対応する行を探す
        matched = df[df["path"].astype(str).str.endswith(f"/{idx}.png")]
        if matched.empty:
            matched = df[df["path"].astype(str).str.endswith(f"\\{idx}.png")]

        if not matched.empty:
            row = matched.iloc[0]
            kept_rows.append([str(new_path), row["Fz"], row["Fr"], row["Ff"]])
        else:
            print(f"  [WARN] datalog.csvに対応行なし: index={idx}")

        prev_hash = h
        new_idx += 1

    # 新しいdatalog.csvを保存
    out_datalog_path = out_session_dir / "datalog.csv"
    with open(out_datalog_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "Fz", "Fr", "Ff"])
        writer.writerows(kept_rows)

    kept = new_idx
    return total, kept


# =========================================================
# 1素材分の処理
# =========================================================
def process_one_material(source_root: Path):
    output_root = Path(str(source_root).rstrip("\\/") + "_dedup")

    print(f"\n=== {source_root.name} ===")
    print(f"入力元: {source_root}")
    print(f"出力先: {output_root}")

    try:
        session_dirs = find_session_dirs(source_root)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return []

    print(f"対象セッション数: {len(session_dirs)}")

    summary_rows = []
    total_all  = 0
    kept_all   = 0

    for session_name, img_dir in session_dirs:
        total, kept = dedup_one_session(session_name, img_dir, output_root)
        dup  = total - kept
        rate = (dup / total * 100) if total > 0 else 0

        print(f"  {session_name}: {total}枚 → {kept}枚 (除去率 {rate:.1f}%)")

        summary_rows.append({
            "material":       source_root.name,
            "session":        session_name,
            "total":          total,
            "kept":           kept,
            "removed":        dup,
            "removed_rate(%)": round(rate, 2)
        })

        total_all += total
        kept_all  += kept

    overall_rate = ((total_all - kept_all) / total_all * 100) if total_all > 0 else 0
    print(f"  合計: {total_all}枚 → {kept_all}枚 (全体除去率 {overall_rate:.1f}%)")

    return summary_rows


# =========================================================
# メイン
# =========================================================
def main():
    print("=== 重複フレーム除去処理 開始 ===")

    all_summary = []

    for source_root_str in SOURCE_ROOTS:
        source_root = Path(source_root_str)
        rows = process_one_material(source_root)
        all_summary.extend(rows)

    # 全体サマリーをCSV保存
    if all_summary:
        summary_df = pd.DataFrame(all_summary)

        # 最初の素材の_dedupフォルダにサマリーを保存
        first_output = Path(SOURCE_ROOTS[0].rstrip("\\/") + "_dedup")
        first_output.mkdir(parents=True, exist_ok=True)
        summary_path = first_output.parent / "dedup_summary_all.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print(f"\n=== 全体サマリー ===")
        print(summary_df.groupby("material")[["total", "kept", "removed"]].sum().to_string())
        print(f"\nサマリー保存先: {summary_path}")


if __name__ == "__main__":
    main()