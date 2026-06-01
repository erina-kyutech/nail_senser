# compare_val_loss.py
# 3方式（G / RGB / HS）の学習ログ(learning_log*.csv)を読み込み、
# val_loss と各タスク(val_Fz_loss/val_Fx_loss/val_Fy_loss)を比較・可視化・集計します。

import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- ここはあなたの保存先に合わせて調整してね ----
DEFAULT_BASE = "./result/CNN_result/compare_G_RGB_HS"
SUBDIRS = {
    "G":   "vgg16_G",
    "RGB": "vgg16_RGB",
    "HS":  "vgg16_HS",
}
CSV_PATTERN = "**/learning_log*.csv"   # vgg16_*/ の下にある CSV を再帰で探索
OUTDIR = "./result/CNN_result/compare_G_RGB_HS/_compare"  # まとめ出力先
# ----------------------------------------------------

NEEDED_COLS = [
    "loss", "Fz_loss", "Fx_loss", "Fy_loss",
    "val_loss", "val_Fz_loss", "val_Fx_loss", "val_Fy_loss"
]

def find_latest_csv(dirpath: str):
    files = glob.glob(os.path.join(dirpath, CSV_PATTERN), recursive=True)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p))  # 最終更新時刻でソート
    return files[-1]

def load_log(csv_path: str):
    df = pd.read_csv(csv_path)
    # 列名の余分な空白や大文字小文字のゆらぎを吸収
    df.columns = [c.strip() for c in df.columns]
    # 期待列が全部あるかチェック（古いコードでも動くよう軽く吸収）
    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が見つかりません: {missing} in {csv_path}")
    # epoch列を付ける（0始まり）
    df.insert(0, "epoch", range(1, len(df)+1))
    return df

def main():
    parser = argparse.ArgumentParser(
        description="G / RGB / HS の val_loss を比較＆可視化"
    )
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help=f"ログのベースディレクトリ（既定: {DEFAULT_BASE}）")
    parser.add_argument("--out", default=OUTDIR,
                        help=f"結果出力先（既定: {OUTDIR}）")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    logs = {}
    for key, sub in SUBDIRS.items():
        d = os.path.join(args.base, sub)
        csv_path = find_latest_csv(d)
        if not csv_path:
            print(f"[WARN] {key}: {d} に learning_log*.csv が見つかりません。スキップします。")
            continue
        try:
            df = load_log(csv_path)
            logs[key] = {"df": df, "path": csv_path}
            print(f"[OK] {key}: {csv_path}")
        except Exception as e:
            print(f"[ERROR] {key}: {csv_path}: {e}")

    if not logs:
        print("有効なログが見つかりませんでした。パス設定を確認してください。")
        return

    # ---------------- 可視化: val_loss 全体 ----------------
    plt.figure(figsize=(8,5))
    for key, item in logs.items():
        df = item["df"]
        plt.plot(df["epoch"], df["val_loss"], label=key)
    plt.xlabel("epoch")
    plt.ylabel("val_loss (overall)")
    plt.title("Validation Loss (overall)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(args.out, "val_loss_overall.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[SAVE] {out_path}")

    # ---------------- 可視化: 各タスク ----------------
    for col, title in [
        ("val_Fz_loss", "Validation Loss (Fz)"),
        ("val_Fx_loss", "Validation Loss (Fx)"),
        ("val_Fy_loss", "Validation Loss (Fy)"),
    ]:
        plt.figure(figsize=(8,5))
        for key, item in logs.items():
            df = item["df"]
            plt.plot(df["epoch"], df[col], label=key)
        plt.xlabel("epoch")
        plt.ylabel(col)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(args.out, f"{col}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[SAVE] {out_path}")

    # ---------------- 集計: 最終epoch と最良epoch ----------------
    rows = []
    for key, item in logs.items():
        df = item["df"]
        last = df.iloc[-1]
        best_idx = df["val_loss"].idxmin()
        best = df.loc[best_idx]

        rows.append({
            "variant": key,
            "csv_path": item["path"],
            "last_epoch": int(last["epoch"]),
            "last_val_loss": float(last["val_loss"]),
            "last_val_Fz_loss": float(last["val_Fz_loss"]),
            "last_val_Fx_loss": float(last["val_Fx_loss"]),
            "last_val_Fy_loss": float(last["val_Fy_loss"]),
            "best_epoch": int(best["epoch"]),
            "best_val_loss": float(best["val_loss"]),
            "best_val_Fz_loss": float(best["val_Fz_loss"]),
            "best_val_Fx_loss": float(best["val_Fx_loss"]),
            "best_val_Fy_loss": float(best["val_Fy_loss"]),
        })

    summary = pd.DataFrame(rows).sort_values("best_val_loss")
    out_csv = os.path.join(args.out, "summary_val_loss.csv")
    summary.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
