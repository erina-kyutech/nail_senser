# -*- coding: utf-8 -*-
"""
generate_dedup_namelist.py

_dedupフォルダをスキャンしてnamelist.csvを生成する。
generate_masked_images.pyの実行前に実行する必要はなく、
train_material.pyやevaluate_material_models.pyが
正しい_dedupのdatalog.csvを参照できるようにするためのスクリプト。

実行環境：TF環境でもYOLO環境でもどちらでもOK
"""

import csv
from pathlib import Path

# =========================================================
# 設定
# =========================================================
DATA_ROOT = Path(r"C:\Users\Owner\PycharmProjects\datas")

# _dedupフォルダのリスト
DEDUP_DIRS = [
    "felt_0-10xyz_dedup",
    "acrylic_0-10xyz_dedup",
    "paper_0-10xyz_dedup",
    "aluminum_0-10xyz_dedup",
]


def generate_namelist(dedup_dir: Path):
    """
    _dedupフォルダをスキャンしてnamelist.csvを生成する。
    """
    namelist_path = dedup_dir / "namelist.csv"

    # ifukuフォルダを番号順に列挙
    session_dirs = sorted(
        [d for d in dedup_dir.iterdir()
         if d.is_dir() and d.name.startswith("ifuku")],
        key=lambda d: int(d.name.replace("ifuku", ""))
    )

    if len(session_dirs) == 0:
        print(f"  [WARN] ifukuフォルダが見つかりません: {dedup_dir}")
        return 0

    rows = []
    for session_dir in session_dirs:
        datalog_path = session_dir / "360deg" / "datalog.csv"
        if datalog_path.exists():
            # 相対パス形式で書く（PROJECT_ROOT/datasからの相対パス）
            rel_path = f"{dedup_dir.name}/{session_dir.name}/360deg/datalog.csv"
            rows.append([rel_path, "360"])

    with open(namelist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"  生成完了: {namelist_path} ({len(rows)}件)")
    return len(rows)


def main():
    print("=== _dedup namelist.csv 生成 ===\n")

    for dir_name in DEDUP_DIRS:
        dedup_dir = DATA_ROOT / dir_name
        print(f"--- {dir_name} ---")

        if not dedup_dir.exists():
            print(f"  [SKIP] フォルダなし: {dedup_dir}")
            continue

        generate_namelist(dedup_dir)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()