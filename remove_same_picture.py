# -*- coding: utf-8 -*-
"""
remove_duplicate_frames.py

撮影済みの連番画像（0.png, 1.png, ...）から、
隣接フレームと完全に同じ画像（重複フレーム）を検出し、
重複を除いた新しいデータセットを作成する。

元のフォルダは変更せず、別フォルダに「重複除去版」を出力する。
datalog.csvも重複を除いた行だけを抽出して作り直す。
"""

import os
import shutil
import hashlib
import csv
import cv2
import pandas as pd

# =========================================================
# 設定
# =========================================================
# 重複チェック対象のルートフォルダ（force_path単位）
# 例: felt_0-10xyz, acrylic_0-10xyz, record0-10xyz など
SOURCE_ROOT = r"C:\Users\Owner\PycharmProjects\datas\felt_0-10xyz"

# 出力先（_dedup を付けて別フォルダに保存する）
OUTPUT_ROOT = SOURCE_ROOT.rstrip("\\/") + "_dedup"

# 対象とするifuku/felt等のID範囲（Noneなら全部処理）
ID_START = None
ID_END = None


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


def find_session_dirs(source_root):
    """
    SOURCE_ROOT 直下の felt1, felt2, ... のような
    撮影セッションフォルダ（360degを含むもの）を列挙する
    """
    session_dirs = []
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"フォルダが見つかりません: {source_root}")

    for name in sorted(os.listdir(source_root)):
        session_path = os.path.join(source_root, name)
        img_dir = os.path.join(session_path, "360deg")
        if os.path.isdir(img_dir):
            session_dirs.append((name, img_dir))

    return session_dirs


def dedup_one_session(session_name, img_dir, output_root):
    """
    1セッション分（例: felt1/360deg）の重複除去を行う。
    重複していない画像だけを出力先にコピーし、
    新しいdatalog.csvを作る。
    """
    datalog_path = os.path.join(img_dir, "datalog.csv")
    if not os.path.exists(datalog_path):
        print(f"  [WARN] datalog.csv が見つかりません: {datalog_path}")
        return 0, 0

    df = pd.read_csv(datalog_path, header=0)
    df.columns = ["path", "Fz", "Fr", "Ff"]

    images = list_numbered_images(img_dir)
    if len(images) == 0:
        print(f"  [WARN] 画像が見つかりません: {img_dir}")
        return 0, 0

    out_session_dir = os.path.join(output_root, session_name, "360deg")
    os.makedirs(out_session_dir, exist_ok=True)

    prev_hash = None
    kept_rows = []
    new_idx = 0
    total = len(images)
    kept = 0

    for idx, path in images:
        img = cv2.imread(path)
        if img is None:
            continue

        h = get_image_hash(img)

        if h == prev_hash:
            # 直前と完全一致 → スキップ（重複として除外）
            prev_hash = h
            continue

        # ユニーク画像として採用 → 連番を振り直してコピー
        new_filename = f"{new_idx}.png"
        new_path = os.path.join(out_session_dir, new_filename)
        shutil.copy2(path, new_path)

        # 元のdatalog.csvから対応する行を探す
        matched = df[df["path"].astype(str).str.endswith(f"/{idx}.png")]
        if matched.empty:
            matched = df[df["path"].astype(str).str.endswith(f"\\{idx}.png")]

        if not matched.empty:
            row = matched.iloc[0]
            kept_rows.append([new_path, row["Fz"], row["Fr"], row["Ff"]])
        else:
            print(f"  [WARN] datalog.csvに対応行なし: index={idx}")

        prev_hash = h
        new_idx += 1
        kept += 1

    # 新しいdatalog.csvを保存
    out_datalog_path = os.path.join(out_session_dir, "datalog.csv")
    with open(out_datalog_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "Fz", "Fr", "Ff"])
        writer.writerows(kept_rows)

    return total, kept


def main():
    print("=== 重複除去処理 開始 ===")
    print("入力元:", SOURCE_ROOT)
    print("出力先:", OUTPUT_ROOT)
    print()

    session_dirs = find_session_dirs(SOURCE_ROOT)
    print(f"対象セッション数: {len(session_dirs)}")

    summary_rows = []
    total_all = 0
    kept_all = 0

    for session_name, img_dir in session_dirs:
        # ID範囲フィルタ（必要な場合）
        if ID_START is not None or ID_END is not None:
            digits = "".join(c for c in session_name if c.isdigit())
            if digits.isdigit():
                num = int(digits)
                if ID_START is not None and num < ID_START:
                    continue
                if ID_END is not None and num > ID_END:
                    continue

        print(f"[{session_name}] 処理中...")
        total, kept = dedup_one_session(session_name, img_dir, OUTPUT_ROOT)
        dup = total - kept
        dup_rate = (dup / total * 100) if total > 0 else 0

        print(f"  元枚数: {total}  →  重複除去後: {kept}  (除去率 {dup_rate:.1f}%)")

        summary_rows.append({
            "session": session_name,
            "total": total,
            "kept": kept,
            "removed": dup,
            "removed_rate(%)": round(dup_rate, 2)
        })

        total_all += total
        kept_all += kept

    # 全体サマリーを保存
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUTPUT_ROOT, "dedup_summary.csv")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    print("\n=== 全体結果 ===")
    print(f"元の合計枚数:     {total_all}")
    print(f"重複除去後の枚数: {kept_all}")
    if total_all > 0:
        print(f"全体除去率:       {(total_all - kept_all) / total_all * 100:.1f}%")
    print(f"\nサマリーを保存しました: {summary_csv_path}")
    print(f"重複除去済みデータ: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()