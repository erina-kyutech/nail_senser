# -*- coding: utf-8 -*-
"""
ontime_dualpath_material_ensemble.py
リアルタイム3軸力推定（2パス比較版）。

ontime_rgb_only_concat_150x290_light_threaded.py のカメラスレッド構成はそのまま流用し、
以下を追加する:

  1. YOLO(ultralytics, CPU推論)で爪(nail)/指腹(finger_tip)をセグメンテーションし、
     それ以外の領域を黒塗りした画像を作る（nail_and_tipマスク画像、学習データと同じ処理）。
  2. 素材分類器（nail_and_tip, テスト精度0.9941）でマスク済み画像から4素材の確率(重み)を算出。
  3. 2つの力推定パスを同時に実行して比較する:
       パス1: 素材混合モデル（all_materials_nail_and_tip）に直接入力
       パス2: 4つの個別素材モデル（felt/acrylic/paper/aluminum_nail_and_tip）にそれぞれ入力し、
              分類器の重みで加重平均

追加機能:
  - u/d キーで左半分（指先が映っている側, n_cy）のROIを上下に移動できる。
    U/D（大文字）は5px刻み。※既存モデルは固定ROIで学習しているので、
    既存モデル評価時は初期値(N_CY=250)のまま使うこと。
  - 表示ウィンドウの左半分にガイド縦線5本（中央のみ緑）を描画。
    表示専用で、モデル入力・保存画像には入らない。

保存先モデルパスは環境依存が大きいため、weights フォルダ内の *.h5 を自動検出する。
起動時に「どのファイルを使ったか」を必ず表示するので、想定と違う場合はログを確認すること。
"""

import os
import glob
import json
import csv
import time
import math
import threading
import queue
import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process, Value
from tensorflow.keras.models import model_from_json

from ultralytics import YOLO

import axis_satsuei_material as axis_satuei_4houkou

# ----------------------------- Config (EDIT HERE) -----------------------------
SUBJECT_NAME = "ifuku"

# concat出力サイズ（撮影・学習と同じ）
OUT_H = 150
OUT_W_LEFT = 150
OUT_W_RIGHT = 140
OUT_W = OUT_W_LEFT + OUT_W_RIGHT  # 290

# カメラindex（環境で0/1逆なら入れ替えて）
CAM_LEFT_INDEX = 1   # 左半分ソース（実機では指先が映っている）
CAM_RIGHT_INDEX = 0  # 右半分ソース（実機では爪が映っている）

# ---- ROI（生値、撮影プログラムと同じ）----
# 注意: 名前は N_* だが、実機では左半分に「指先」が映っている。
#       既存データセット・学習済みモデルとの整合のため並びは変えない。
N_CX, N_CY = 499, 250
N_W0, N_H0 = 282, 409

T_CX, T_CY = 324, 550
T_W0, T_H0 = 182, 136

N_W_SCALE, N_H_SCALE = 1.3, 0.9
T_W_SCALE, T_H_SCALE = 1.7, 1.0

# ---- 指先ROI(左半分)手動調整 ----
# カメラが指先を横向きに見ているため、実世界の「高さ」は
# 生画像上では「左右」(n_cx) に対応する。u/d で n_cx を動かす。
TIP_STEP = 1        # u/d 1回の移動量[px]
TIP_STEP_BIG = 5    # U/D（大文字）の移動量[px]
TIP_DIR = 1         # uを押したときの符号。逆に動くと感じたら -1 に
N_GUIDE = 5         # ガイド縦線の本数

# 調整したn_cxを保存して次回起動時に自動復元する
TIP_ROI_STATE_PATH = "./tip_roi_state_realtime.txt"

# ---- YOLOセグメンテーション ----
YOLO_WEIGHT_PATH = r"C:\Users\Owner\PycharmProjects\YOLO_nail_seg\runs\segment\runs\segment\nail_seg_v1\weights\best.onnx"
YOLO_IMGSZ = [160, 320]
YOLO_CONF = 0.1
YOLO_DEVICE = "cpu"

# ---- モデルパス ----
PROJECT_ROOT = r"C:\Users\Owner\PycharmProjects"
MATERIAL_MODELS_ROOT = os.path.join(PROJECT_ROOT, "result", "CNN_result", "material_models")
CLASSIFIER_ROOT = os.path.join(PROJECT_ROOT, "result", "CNN_result", "material_classifier_nail_and_tip")

MATERIALS = ["felt", "acrylic", "paper", "aluminum"]
MIXED_MODEL_DIR = os.path.join(MATERIAL_MODELS_ROOT, "all_materials_nail_and_tip")
MATERIAL_MODEL_DIRS = {
    m: os.path.join(MATERIAL_MODELS_ROOT, f"{m}_nail_and_tip") for m in MATERIALS
}

# 軽量化用
PLOT_EVERY_N_FRAMES = 5
SHOW_SEC = 10.0

# ---- 推定モード ----
# "both"           : 混合モデル・重み付きアンサンブル両方を毎フレーム推定（比較用、一番重い）
# "mixed_only"     : 混合モデルのみ（分類器・個別4モデルは読み込まない、一番軽い）
# "weighted_only"  : 分類器+重み付きアンサンブルのみ（混合モデルは読み込まない）
MODE = "both"

# 分類器は毎フレームではなく間引いて実行し、結果を次に更新されるまで使い回す。
# 1回の測定(recording)中は素材が変わらない前提なので、これで精度を落とさずに軽くできる。
CLASSIFY_EVERY_N_FRAMES = 15

# ---- リアルタイム画像の保存 ----
SAVE_IMAGES = False
# 保存先はCSVと同じ回ごとのフォルダに自動で分けられる:
#   ./realtime_logs/<素材名>/images_<連番2桁>/0.png, 1.png, ...
# （固定フォルダだと次の回に0.pngから上書きされてしまうため）
SAVE_IMAGE_EXT = ".png"   # 速度優先なら ".jpg" もあり（画質より速度が欲しい場合）

# ---- Excel(.xlsx)への定期エクスポート ----
# xlsxは追記型フォーマットではないため、毎フレーム書き込むとデータが増えるほど遅くなる。
# そのため一定間隔でバックグラウンドスレッドが全データをスナップショット保存する方式にする。
# 常に最新のCSVは毎フレーム即時記録されているので、
# 万一Excel保存が間に合わなくてもデータが失われることはない。
# 保存先はCSVと同名（拡張子だけ.xlsx）に自動で揃える。
EXCEL_EXPORT = False
EXCEL_SAVE_INTERVAL_SEC = 5.0

# ---- データログの保存先 ----
# 起動時に素材名を入力すると、
#   ./realtime_logs/<素材名>/datalog_fr_<素材名>_<連番2桁>.csv
# が自動生成される。連番はフォルダ内の既存ファイルから自動で+1。
# 上書きは絶対に起きない。1行も記録せず終了した場合は空ファイルを自動削除する。
# 失敗した回はファイルを手で消せば、次回同じ番号が再利用される。
LOG_ROOT = "./realtime_logs"
# -----------------------------------------------------------------------------


# =========================================================
# 画像処理（撮影プログラムと同じ）
# =========================================================
def crop_with_center_wh_safe(img, cx, cy, w, h):
    H, W = img.shape[:2]
    w = int(max(1, min(w, W)))
    h = int(max(1, min(h, H)))
    cx = int(max(w // 2, min(cx, W - w // 2)))
    cy = int(max(h // 2, min(cy, H - h // 2)))
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = x1 + w
    y2 = y1 + h
    return img[y1:y2, x1:x2], (x1, y1, w, h)


def _resize_no_pad_center_crop(img, out_w, out_h):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    target = out_w / out_h
    cur = w / h

    if cur > target:
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        cropped = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        cropped = img[y0:y0 + new_h, :]

    interp = cv2.INTER_AREA if (cropped.shape[0] > out_h or cropped.shape[1] > out_w) else cv2.INTER_LINEAR
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)


def make_concat_bgr(img_left_bgr, img_right_bgr, n_cx=N_CX):
    """n_cx: 左半分（指先側）ROIの中心x。u/dキーで動かせるよう引数化。
    （カメラが横向きのため、実世界の高さ調整 = 画像上の左右移動）"""
    n_w = int(N_W0 * N_W_SCALE)
    n_h = int(N_H0 * N_H_SCALE)
    t_w = int(T_W0 * T_W_SCALE)
    t_h = int(T_H0 * T_H_SCALE)

    roi_l, _ = crop_with_center_wh_safe(img_left_bgr, n_cx, N_CY, n_w, n_h)
    roi_r, _ = crop_with_center_wh_safe(img_right_bgr, T_CX, T_CY, t_w, t_h)

    roi_l = _resize_no_pad_center_crop(roi_l, OUT_W_LEFT, OUT_H)
    roi_r = _resize_no_pad_center_crop(roi_r, OUT_W_RIGHT, OUT_H)

    return cv2.hconcat([roi_l, roi_r])


def preprocess_for_mode(concat_bgr):
    rgb = cv2.cvtColor(concat_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[None, ...] / 255.0


def data_unnormalize(Y):
    """力推定モデルの正規化を戻す（rgb_only_training.py等と同じ規約）"""
    normal_force_normalize = 10.0
    thear_force_normalize = 5.0
    Y = Y.copy()
    Y[:, 0] *= normal_force_normalize
    Y[:, 1] *= (thear_force_normalize * 2)
    Y[:, 2] *= (thear_force_normalize * 2)
    Y[:, 1] -= thear_force_normalize
    Y[:, 2] -= thear_force_normalize
    return Y


# =========================================================
# データログのファイル名管理
# =========================================================
MATERIAL_PRESETS = {
    "1": "felt",
    "2": "acrylic",
    "3": "paper",
    "4": "aluminum",
}


def select_material_name() -> str:
    """起動時に素材名を入力させる。番号(1-4)か自由入力（例: hinoki）。"""
    prompt = (
        "素材を選んでください:\n"
        "  1: felt (フェルト)\n"
        "  2: acrylic (アクリル)\n"
        "  3: paper (紙)\n"
        "  4: aluminum (アルミ)\n"
        "  それ以外の素材は名前を直接入力（例: hinoki）\n"
        "番号 または 素材名（英字半角） > "
    )
    while True:
        raw = input(prompt).strip()
        if raw in MATERIAL_PRESETS:
            name = MATERIAL_PRESETS[raw]
        else:
            name = raw.lower()
        # ファイル名に使えない文字を除外
        if name and all(c.isalnum() or c in "-_" for c in name):
            print(f"[INFO] 素材: {name}")
            return name
        print(f'[WARN] "{raw}" はファイル名に使えません。英数字で入力してください。\n')


def next_run_number(log_dir: str, material: str) -> int:
    """log_dir内の datalog_fr_<material>_NN.csv をスキャンして次の連番を返す。"""
    import re
    # 旧形式（タイムスタンプ付き）も一応マッチさせて番号被りを防ぐ
    pattern = re.compile(rf"^datalog_fr_{re.escape(material)}_(\d+)(_.*)?\.csv$")
    max_run = 0
    if os.path.isdir(log_dir):
        for fname in os.listdir(log_dir):
            m = pattern.match(fname)
            if m:
                max_run = max(max_run, int(m.group(1)))
    return max_run + 1


def build_datalog_path(material: str):
    """素材別フォルダに、連番付きのCSVパスを作る。上書きは起きない。"""
    log_dir = os.path.join(LOG_ROOT, material)
    os.makedirs(log_dir, exist_ok=True)
    run = next_run_number(log_dir, material)
    csv_path = os.path.join(log_dir, f"datalog_fr_{material}_{run:02d}.csv")
    return csv_path, run


# =========================================================
# YOLOセグメンテーション → マスク適用
# =========================================================
def extract_masks(result, orig_h, orig_w):
    """ultralyticsのresultからnail/finger_tipのマスクを取得する（test_yolo_ultarlytics.pyと同じ）"""
    nail_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    tip_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if result.masks is None:
        return nail_mask, tip_mask

    names = result.names
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    masks_data = result.masks.data.cpu().numpy()

    for i, cls_id in enumerate(cls_ids):
        m = masks_data[i]
        m_resized = cv2.resize(m.astype(np.float32), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
        m_bin = (m_resized > 0.5).astype(np.uint8)

        class_name = names[cls_id]
        if class_name == "nail":
            nail_mask = np.logical_or(nail_mask, m_bin).astype(np.uint8)
        elif class_name == "finger_tip":
            tip_mask = np.logical_or(tip_mask, m_bin).astype(np.uint8)

    return nail_mask, tip_mask


def apply_nail_and_tip_mask(concat_bgr, nail_mask, tip_mask):
    """爪+指腹の領域以外を黒塗りする（学習データのnail_and_tipマスクと同じ処理）"""
    combined_mask = np.logical_or(nail_mask, tip_mask)
    out = concat_bgr.copy()
    out[combined_mask == 0] = 0
    return out, combined_mask


# =========================================================
# モデル読み込み（フォルダ内の.json/.h5を自動検出）
# =========================================================
def find_json_file(model_dir: str) -> str:
    candidates = [
        os.path.join(model_dir, "for0-10.json"),
        os.path.join(model_dir, "classifier.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    json_files = glob.glob(os.path.join(model_dir, "*.json"))
    json_files = [j for j in json_files if os.path.basename(j) != "label_info.json"]
    if len(json_files) == 1:
        return json_files[0]
    if len(json_files) > 1:
        raise FileNotFoundError(
            f"jsonファイルが複数見つかりました。どれを使うか特定できません: {json_files}"
        )
    raise FileNotFoundError(f"モデルの.jsonが見つかりません: {model_dir}")


def find_weight_file(model_dir: str) -> str:
    weights_dir = os.path.join(model_dir, "weights")
    if not os.path.isdir(weights_dir):
        weights_dir = model_dir  # weightsサブフォルダが無い場合はそのまま探す

    candidates = [
        os.path.join(weights_dir, f"weight_{SUBJECT_NAME}_for0-10.h5"),
        os.path.join(weights_dir, "classifier_weights.h5"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    h5_files = glob.glob(os.path.join(weights_dir, "*.h5"))
    if len(h5_files) == 1:
        return h5_files[0]
    if len(h5_files) > 1:
        raise FileNotFoundError(
            f"重みファイル(.h5)が複数見つかりました。どれを使うか特定できません: {h5_files}"
        )
    raise FileNotFoundError(f"重みファイル(.h5)が見つかりません: {weights_dir}")


def load_keras_model(model_dir: str, label: str):
    json_path = find_json_file(model_dir)
    weight_path = find_weight_file(model_dir)
    print(f"[MODEL:{label}] json  : {json_path}")
    print(f"[MODEL:{label}] weight: {weight_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        model_json_string = f.read()

    model = model_from_json(model_json_string, custom_objects={})
    model.load_weights(weight_path)
    return model


# -------- sensors (from axis_satsuei_material) --------
gf2000 = axis_satuei_4houkou.gf2000
SC800IM700_1 = axis_satuei_4houkou.SC800IM700_1
SC800IM700_2 = axis_satuei_4houkou.SC800IM700_2
SC800IM700_3 = axis_satuei_4houkou.SC800IM700_3
SC800IM700_4 = axis_satuei_4houkou.SC800IM700_4


class RealTime:
    def __init__(self):
        # ---- YOLO(セグメンテーション、CPU) ----
        print("=== YOLOモデル読み込み ===")
        self.yolo = YOLO(YOLO_WEIGHT_PATH)
        print("YOLO読み込み: OK")

        # ---- 指先ROI(左半分)の手動調整状態 ----
        self.n_cx = N_CX
        self.N_GUIDE = N_GUIDE

        # 前回調整したn_cxを復元
        if os.path.exists(TIP_ROI_STATE_PATH):
            try:
                with open(TIP_ROI_STATE_PATH, "r", encoding="utf-8") as f:
                    self.n_cx = int(f.read().strip())
                print(f"[INFO] 前回の指先ROI中心を復元: n_cx={self.n_cx}"
                      f"（既存モデル評価時は初期値{N_CX}に戻すこと）")
            except (ValueError, OSError):
                pass

        # ---- 素材分類器（weighted_only / both のときだけ必要） ----
        self.classifier = None
        self.id_to_material = {0: "felt", 1: "acrylic", 2: "paper", 3: "aluminum"}
        self.classifier_material_order = [self.id_to_material[i] for i in range(4)]

        if MODE in ("both", "weighted_only"):
            print("=== 素材分類器読み込み ===")
            self.classifier = load_keras_model(CLASSIFIER_ROOT, "classifier")

            label_info_path = os.path.join(CLASSIFIER_ROOT, "label_info.json")
            if os.path.exists(label_info_path):
                with open(label_info_path, "r", encoding="utf-8") as f:
                    label_info = json.load(f)
                self.id_to_material = {int(k): v for k, v in label_info["id_to_material"].items()}
            self.classifier_material_order = [self.id_to_material[i] for i in range(len(self.id_to_material))]
            print(f"classifier出力の並び: {self.classifier_material_order}")

        # ---- 力推定モデル（MODEに応じて必要なものだけ読み込む） ----
        self.mixed_model = None
        if MODE in ("both", "mixed_only"):
            print("=== 力推定モデル読み込み(混合) ===")
            self.mixed_model = load_keras_model(MIXED_MODEL_DIR, "mixed")

        self.material_models = {}
        if MODE in ("both", "weighted_only"):
            print("=== 力推定モデル読み込み(個別素材) ===")
            for m in MATERIALS:
                self.material_models[m] = load_keras_model(MATERIAL_MODEL_DIRS[m], m)

        print(f"\n=== 推定モード: {MODE} ===")

        # 分類器の間引き実行用キャッシュ
        self._last_class_weights = {m: 1.0 / len(MATERIALS) for m in MATERIALS}
        self._classify_counter = 0

        # ---- データログ（素材別フォルダ・連番自動）----
        self.material_name = select_material_name()
        self.datalog_path, run_no = build_datalog_path(self.material_name)
        self.excel_path = os.path.splitext(self.datalog_path)[0] + ".xlsx"
        print(f"[INFO] この回のログ: {self.datalog_path}（{self.material_name} {run_no}回目）")

        # ---- 画像の非同期保存（CSVと同じ回ごとのフォルダに保存）----
        self._save_queue = None
        self._save_thread = None
        self._save_frame_idx = 0
        self.save_image_dir = os.path.join(
            os.path.dirname(self.datalog_path), f"images_{run_no:02d}"
        )
        if SAVE_IMAGES:
            os.makedirs(self.save_image_dir, exist_ok=True)
            self._save_queue = queue.Queue()
            self._save_thread = threading.Thread(target=self._save_worker, daemon=False)
            self._save_thread.start()
            print(f"画像保存: 有効 -> {self.save_image_dir}")

        # ---- camera ----
        self.cap_l = cv2.VideoCapture(CAM_LEFT_INDEX)
        self.cap_r = cv2.VideoCapture(CAM_RIGHT_INDEX)
        if (not self.cap_l.isOpened()) or (not self.cap_r.isOpened()):
            raise RuntimeError("Camera open failed. (index 0/1 reversed?)")

        self._frame_l = None
        self._frame_r = None
        self._lock_l = threading.Lock()
        self._lock_r = threading.Lock()
        self._cam_running = True

        self._thread_l = threading.Thread(target=self._read_l, daemon=True)
        self._thread_r = threading.Thread(target=self._read_r, daemon=True)
        self._thread_l.start()
        self._thread_r.start()
        time.sleep(0.5)

        cv2.namedWindow("concat", cv2.WINDOW_NORMAL)
        cv2.moveWindow("concat", 50, 50)
        cv2.setWindowProperty("concat", cv2.WND_PROP_TOPMOST, 1)

        # FPS
        self.prev_time = time.perf_counter()
        self.fps_ema = 0.0
        self.fps_alpha = 0.1

        self.N2gf = 101.972  # g -> N

        # csv（パスは__init__冒頭で決定済み）
        self.data_csv = open(self.datalog_path, "w", newline="")
        self.w = csv.writer(self.data_csv)
        header = (
            ["Time", "Fx_True", "Fy_True", "Fz_True",
             "Fx_Pred_mixed", "Fy_Pred_mixed", "Fz_Pred_mixed",
             "Fx_Pred_weighted", "Fy_Pred_weighted", "Fz_Pred_weighted",
             "Fx_Err_mixed", "Fy_Err_mixed", "Fz_Err_mixed",
             "Fx_Err_weighted", "Fy_Err_weighted", "Fz_Err_weighted"]
            + [f"proba_{m}" for m in self.classifier_material_order]
            + (["Image_Path"] if SAVE_IMAGES else [])
        )
        self.header = header
        self.w.writerow(header)

        # ---- Excel定期エクスポート用 ----
        self._all_rows = []            # 記録した全行（辞書ではなくheader順のlist）
        self._last_excel_save_time = time.perf_counter()
        self._excel_save_thread = None

        # plot init
        plt.ion()
        self.force_names = ["Fz", "Fx", "Fy"]
        self.fig, self.axes = plt.subplots(1, 4, figsize=(17, 4),
                                           gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
        force_axes_2d = np.expand_dims(self.axes[:3], axis=0)
        self.axes = force_axes_2d
        self.ax_material = self.fig.axes[3]  # 4番目：素材確率の棒グラフ

        self.show_sec = SHOW_SEC
        self.plot_counter = 0
        self.hist_time = []
        self.hist_true = {k: [] for k in self.force_names}
        self.hist_mixed = {k: [] for k in self.force_names}
        self.hist_weighted = {k: [] for k in self.force_names}

        self.lines_true = {}
        self.lines_mixed = {}
        self.lines_weighted = {}
        for c, k in enumerate(self.force_names):
            ax = self.axes[0, c]
            lt, = ax.plot([], [], ls=":", color="black", label="true")
            lm, = ax.plot([], [], ls="-", color="tab:blue", label="mixed")
            lw, = ax.plot([], [], ls="--", color="tab:orange", label="weighted")
            self.lines_true[k] = lt
            self.lines_mixed[k] = lm
            self.lines_weighted[k] = lw
            ax.set_title(k)
            if c == 0:
                ax.legend(fontsize=8, loc="upper left")

        # ---- 素材確率パネルの初期化 ----
        self.ax_material.set_title("material proba", fontsize=10)
        self.ax_material.set_ylim(0, 1.0)
        if MODE in ("both", "weighted_only"):
            init_probs = [self._last_class_weights[m] for m in self.classifier_material_order]
            self._material_bars = self.ax_material.bar(
                self.classifier_material_order, init_probs, color="tab:gray"
            )
            self.ax_material.set_xticks(range(len(self.classifier_material_order)))
            self.ax_material.set_xticklabels(self.classifier_material_order, rotation=30, fontsize=8)
        else:
            self._material_bars = None
            self.ax_material.text(0.5, 0.5, "N/A\n(mixed_only)", ha="center", va="center",
                                   transform=self.ax_material.transAxes, fontsize=9, color="gray")

        self.fig.tight_layout()

        # control
        self.is_recording = False
        self.start_time = None
        self.Fr0 = 0.0
        self.Ff0 = 0.0

        print("\n準備OK：u/dで指先ROI(左半分)の高さ調整（U/Dは5px刻み）。"
              "指の位置を確認して 'r' を押すと測定開始（'ESC'で終了）")

    def _read_l(self):
        while self._cam_running:
            ret, frame = self.cap_l.read()
            if ret and frame is not None:
                with self._lock_l:
                    self._frame_l = frame

    def _read_r(self):
        while self._cam_running:
            ret, frame = self.cap_r.read()
            if ret and frame is not None:
                with self._lock_r:
                    self._frame_r = frame

    def _save_worker(self):
        """画像保存専用スレッド。メインループをディスクI/Oでブロックしないためのキュー処理。"""
        while True:
            item = self._save_queue.get()
            if item is None:  # 終了シグナル
                self._save_queue.task_done()
                break
            filepath, img = item
            try:
                cv2.imwrite(filepath, img)
            except Exception as e:
                print(f"\n[WARN] 画像保存失敗: {filepath} ({e})")
            self._save_queue.task_done()

    def _export_excel_async(self, rows_snapshot):
        """rows_snapshot（その時点までの全行のコピー）をバックグラウンドでxlsxに書き出す"""
        def _worker(rows, header, path):
            try:
                df = pd.DataFrame(rows, columns=header)
                df.to_excel(path, index=False, engine="openpyxl")
            except Exception as e:
                print(f"\n[WARN] Excel保存失敗: {e}")

        t = threading.Thread(target=_worker, args=(rows_snapshot, self.header, self.excel_path), daemon=True)
        t.start()
        return t

    def _maybe_export_excel(self):
        if not EXCEL_EXPORT:
            return
        now = time.perf_counter()
        if (now - self._last_excel_save_time) < EXCEL_SAVE_INTERVAL_SEC:
            return
        # 既に前回のエクスポートが走ってる場合は重複起動しない
        if self._excel_save_thread is not None and self._excel_save_thread.is_alive():
            return
        self._last_excel_save_time = now
        rows_snapshot = list(self._all_rows)  # スナップショットコピー
        self._excel_save_thread = self._export_excel_async(rows_snapshot)

    def _save_tip_roi(self):
        try:
            with open(TIP_ROI_STATE_PATH, "w", encoding="utf-8") as f:
                f.write(str(self.n_cx))
        except OSError:
            pass

    def _move_tip_roi(self, direction, big=False):
        """左半分（指先側）ROIの中心xを移動する。表示・入力ともに反映される。
        （カメラが横向きのため、実世界の高さ調整 = 画像上の左右移動）"""
        step = TIP_STEP_BIG if big else TIP_STEP
        self.n_cx += step * direction * TIP_DIR
        with self._lock_l:
            frame = self._frame_l
        if frame is not None:
            W = frame.shape[1]
            n_w = int(N_W0 * N_W_SCALE)
            self.n_cx = int(max(n_w // 2, min(self.n_cx, W - n_w // 2)))
        print(f"\n[ROI] 指先(左半分)中心x = {self.n_cx}")
        self._save_tip_roi()

    def _predict_force(self, model, x):
        y_list = model(x, training=False)
        y = np.concatenate([arr.numpy() for arr in y_list], axis=1)
        y = data_unnormalize(y)
        return y[0]  # [Fz, Fx, Fy]

    def _predict_classifier_weights(self, x):
        proba = self.classifier(x, training=False).numpy()[0]  # (4,)
        return {mat: float(proba[i]) for i, mat in enumerate(self.classifier_material_order)}

    def _update_plot(self):
        if len(self.hist_time) < 2:
            return

        t1 = self.hist_time[-1]
        t0 = max(0.0, t1 - self.show_sec)
        if t1 == t0:
            t1 = t0 + 1e-6

        for c, k in enumerate(self.force_names):
            ax = self.axes[0, c]
            self.lines_true[k].set_data(self.hist_time, self.hist_true[k])
            self.lines_mixed[k].set_data(self.hist_time, self.hist_mixed[k])
            self.lines_weighted[k].set_data(self.hist_time, self.hist_weighted[k])
            ax.set_xlim(t0, t1)

            y_all = np.array(
                self.hist_true[k] + self.hist_mixed[k] + self.hist_weighted[k], dtype=np.float32
            )
            if y_all.size > 0:
                ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                pad = 0.12 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

        # ---- 素材確率パネルの更新 ----
        if self._material_bars is not None:
            top_material = max(self._last_class_weights, key=self._last_class_weights.get)
            for bar, mat in zip(self._material_bars, self.classifier_material_order):
                prob = self._last_class_weights[mat]
                bar.set_height(prob)
                bar.set_color("tab:orange" if mat == top_material else "tab:gray")
            self.ax_material.set_title(f"material: {top_material} ({self._last_class_weights[top_material]*100:.0f}%)",
                                       fontsize=9)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def loop(self):
        try:
            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord('r') and not self.is_recording:
                    print("=== 測定開始 ===")
                    time.sleep(1.0)
                    self.Fr0 = shear_force1.value - shear_force3.value
                    self.Ff0 = shear_force2.value - shear_force4.value
                    self.start_time = time.perf_counter()
                    self.is_recording = True

                if key == 27:
                    break

                # ---- 指先ROI(左半分)の高さ調整（記録中は誤操作防止のため無効） ----
                if key in (ord('u'), ord('U'), ord('d'), ord('D')):
                    if self.is_recording:
                        print("\n[ROI] 記録中はROI調整できません")
                    else:
                        if key in (ord('u'), ord('U')):
                            self._move_tip_roi(-1, big=(key == ord('U')))
                        else:
                            self._move_tip_roi(+1, big=(key == ord('D')))

                # FPS
                now_fps_time = time.perf_counter()
                dt = now_fps_time - self.prev_time
                self.prev_time = now_fps_time
                if dt > 0:
                    fps_inst = 1.0 / dt
                    self.fps_ema = (1 - self.fps_alpha) * self.fps_ema + self.fps_alpha * fps_inst

                # capture
                with self._lock_l:
                    base_l = self._frame_l
                with self._lock_r:
                    base_r = self._frame_r
                if base_l is None or base_r is None:
                    continue

                concat_bgr = make_concat_bgr(base_l, base_r, self.n_cx)

                # ---- YOLOセグメンテーション → マスク適用 ----
                yolo_t0 = time.perf_counter()
                results = self.yolo.predict(
                    source=concat_bgr, device=YOLO_DEVICE, verbose=False,
                    imgsz=YOLO_IMGSZ, conf=YOLO_CONF
                )
                yolo_ms = (time.perf_counter() - yolo_t0) * 1000
                nail_mask, tip_mask = extract_masks(results[0], OUT_H, OUT_W)
                masked_bgr, combined_mask = apply_nail_and_tip_mask(concat_bgr, nail_mask, tip_mask)

                if self.is_recording:
                    t = time.perf_counter() - self.start_time

                    # true forces
                    Fz_true = float(normal_force.value / self.N2gf)
                    Fx_true = (shear_force1.value - shear_force3.value) - self.Fr0
                    Fy_true = (shear_force2.value - shear_force4.value) - self.Ff0

                    X = preprocess_for_mode(masked_bgr)

                    Fz_m = Fx_m = Fy_m = np.nan
                    Fz_w = Fx_w = Fy_w = np.nan
                    class_weights = self._last_class_weights

                    # ---- パス1: 混合モデル ----
                    if MODE in ("both", "mixed_only"):
                        Fz_m, Fx_m, Fy_m = self._predict_force(self.mixed_model, X)

                    # ---- パス2: 分類器（間引き実行）+ 個別モデルの加重平均 ----
                    if MODE in ("both", "weighted_only"):
                        if self._classify_counter % CLASSIFY_EVERY_N_FRAMES == 0:
                            class_weights = self._predict_classifier_weights(X)
                            self._last_class_weights = class_weights
                        self._classify_counter += 1

                        weighted_sum = np.zeros(3, dtype=np.float64)
                        for mat in MATERIALS:
                            pred = self._predict_force(self.material_models[mat], X)
                            weighted_sum += class_weights[mat] * pred
                        Fz_w, Fx_w, Fy_w = weighted_sum

                    # ---- 画像の非同期保存（キューに投げるだけでメインループはブロックしない） ----
                    if SAVE_IMAGES:
                        img_filename = f"{self._save_frame_idx}{SAVE_IMAGE_EXT}"
                        img_path = os.path.join(self.save_image_dir, img_filename)
                        self._save_queue.put((img_path, masked_bgr.copy()))
                        self._save_frame_idx += 1
                    else:
                        img_path = ""

                    # history
                    self.hist_time.append(t)
                    self.hist_true["Fz"].append(Fz_true)
                    self.hist_true["Fx"].append(Fx_true)
                    self.hist_true["Fy"].append(Fy_true)
                    self.hist_mixed["Fz"].append(Fz_m)
                    self.hist_mixed["Fx"].append(Fx_m)
                    self.hist_mixed["Fy"].append(Fy_m)
                    self.hist_weighted["Fz"].append(Fz_w)
                    self.hist_weighted["Fx"].append(Fx_w)
                    self.hist_weighted["Fy"].append(Fy_w)

                    while len(self.hist_time) > 0 and (self.hist_time[-1] - self.hist_time[0]) > self.show_sec:
                        self.hist_time.pop(0)
                        for k in self.force_names:
                            self.hist_true[k].pop(0)
                            self.hist_mixed[k].pop(0)
                            self.hist_weighted[k].pop(0)

                    # csv（使ってないパスはNaNのまま記録される）
                    row = (
                        [t, Fx_true, Fy_true, Fz_true,
                         Fx_m, Fy_m, Fz_m,
                         Fx_w, Fy_w, Fz_w,
                         Fx_m - Fx_true, Fy_m - Fy_true, Fz_m - Fz_true,
                         Fx_w - Fx_true, Fy_w - Fy_true, Fz_w - Fz_true]
                        + [class_weights[m] for m in self.classifier_material_order]
                        + ([img_path] if SAVE_IMAGES else [])
                    )
                    self.w.writerow(row)
                    self._all_rows.append(row)
                    self._maybe_export_excel()

                    # 画面表示（コンソール）
                    status = f" t={t:5.1f}s  Fz(true"
                    if MODE in ("both", "mixed_only"):
                        status += f"/mix={Fz_m:5.2f}"
                    if MODE in ("both", "weighted_only"):
                        status += f"/w={Fz_w:5.2f}"
                        weight_str = " ".join(f"{m}:{class_weights[m]:.2f}" for m in self.classifier_material_order)
                        status += f")  [{weight_str}]"
                    else:
                        status += f")={Fz_true:5.2f}"
                    print(f"\r{status}", end="")

                    self.plot_counter += 1
                    if self.plot_counter % PLOT_EVERY_N_FRAMES == 0:
                        self._update_plot()

                # ---- display（マスク適用後の画像 + ガイド縦線 + FPS） ----
                disp_bgr = masked_bgr.copy()

                # 表示専用ガイド縦線（左半分＝指先側、モデル入力・保存画像には入らない）
                xs = np.linspace(0, OUT_W_LEFT - 1, self.N_GUIDE + 2).astype(int)[1:-1]
                mid = xs[len(xs) // 2]
                for x in xs:
                    color = (0, 255, 0) if x == mid else (0, 0, 255)  # BGRなので赤=(0,0,255)
                    cv2.line(disp_bgr, (int(x), 0), (int(x), OUT_H - 1), color, 1)

                disp_bgr = cv2.resize(disp_bgr, (OUT_W * 2, OUT_H * 2))
                cv2.putText(disp_bgr, f"FPS: {self.fps_ema:.1f}  YOLO: {yolo_ms:.0f}ms  n_cx: {self.n_cx}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("concat", disp_bgr)

        finally:
            self._cam_running = False
            if self._save_queue is not None:
                print("\n画像保存キューの残りを書き出しています...")
                self._save_queue.put(None)
                self._save_thread.join()
            if EXCEL_EXPORT and len(self._all_rows) > 0:
                print("最終Excel保存中...")
                # 終了時は同期的に保存して確実にデータを残す
                if self._excel_save_thread is not None and self._excel_save_thread.is_alive():
                    self._excel_save_thread.join()
                df = pd.DataFrame(self._all_rows, columns=self.header)
                try:
                    df.to_excel(self.excel_path, index=False, engine="openpyxl")
                    print(f"Excel保存完了: {self.excel_path}")
                except Exception as e:
                    print(f"[WARN] 最終Excel保存失敗: {e}")
            try:
                self.data_csv.close()
            except Exception:
                pass

            # 1行も記録せず終了した場合はヘッダだけの空CSVを自動削除
            # （センサ不調ですぐやり直すときにファイルが溜まらない）
            if len(self._all_rows) == 0:
                try:
                    os.remove(self.datalog_path)
                    print(f"\n[INFO] データ未記録のため空ログを削除: {self.datalog_path}")
                except OSError:
                    pass
                if SAVE_IMAGES:
                    try:
                        os.rmdir(self.save_image_dir)  # 空のときだけ消える
                    except OSError:
                        pass
            else:
                print(f"\n[INFO] ログ保存済み: {self.datalog_path}（{len(self._all_rows)}行）")
            self.cap_l.release()
            self.cap_r.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.close("all")


if __name__ == "__main__":

    # -------- shared memory vars --------
    normal_force = Value("f", 0.00)
    shear_force1 = Value("f", 0.00)
    shear_force2 = Value("f", 0.00)
    shear_force3 = Value("f", 0.00)
    shear_force4 = Value("f", 0.00)
    ser_flag = Value("b", True)

    # --- ports (あなたの環境) ---
    xy_port_1 = "COM8"
    xy_port_2 = "COM9"
    xy_port_3 = "COM10"
    xy_port_4 = "COM12"
    xy_address = 0x2A

    z_port = "COM21"

    # --- sensor init ---
    shear_loadcell_1 = SC800IM700_1(xy_port_1, xy_address); shear_loadcell_1.power_on(); shear_loadcell_1.sub_ready()
    shear_loadcell_2 = SC800IM700_2(xy_port_2, xy_address); shear_loadcell_2.power_on(); shear_loadcell_2.sub_ready()
    shear_loadcell_3 = SC800IM700_3(xy_port_3, xy_address); shear_loadcell_3.power_on(); shear_loadcell_3.sub_ready()
    shear_loadcell_4 = SC800IM700_4(xy_port_4, xy_address); shear_loadcell_4.power_on(); shear_loadcell_4.sub_ready()

    normal_loadcell = gf2000(z_port); normal_loadcell.sub_ready()

    # --- subprocess start ---
    sub_z = Process(target=gf2000.sub_loop, args=[z_port, ser_flag, normal_force])
    sub_z.start()

    count1 = Value("i", 0); count2 = Value("i", 0); count3 = Value("i", 0); count4 = Value("i", 0)

    sub_xy1 = Process(target=SC800IM700_1.sub_loop, args=[xy_port_1, xy_address, ser_flag, shear_force1, count1])
    sub_xy2 = Process(target=SC800IM700_2.sub_loop, args=[xy_port_2, xy_address, ser_flag, shear_force2, count2])
    sub_xy3 = Process(target=SC800IM700_3.sub_loop, args=[xy_port_3, xy_address, ser_flag, shear_force3, count3])
    sub_xy4 = Process(target=SC800IM700_4.sub_loop, args=[xy_port_4, xy_address, ser_flag, shear_force4, count4])

    sub_xy1.start(); sub_xy2.start(); sub_xy3.start(); sub_xy4.start()

    RealTime().loop()