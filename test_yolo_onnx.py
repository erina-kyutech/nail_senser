# -*- coding: utf-8 -*-
"""
test_yolo_onnx.py

TensorFlow環境でONNX版YOLOが動くか確認するテストスクリプト。
カメラ画像に対してセグメンテーションを実行し、
爪マスクと指先マスクが正しく取れているか可視化する。

確認できること：
  - onnxruntimeがTF環境で動くか
  - best.onnxが正しく読み込めるか
  - 入力サイズ（160×320）へのリサイズが正しく動くか
  - 爪・指先のマスクが正しく抽出できるか
"""

import cv2
import numpy as np
import onnxruntime as ort
import time

# =========================================================
# 設定
# =========================================================
ONNX_PATH = r"C:\Users\Owner\PycharmProjects\YOLO_nail_seg\runs\segment\runs\segment\nail_seg_v1\weights\best.onnx"

# YOLOの入力サイズ（変換時に32の倍数に自動調整されたサイズ）
YOLO_H = 160
YOLO_W = 320

# concat画像のサイズ（VGG16用、変えない）
ORIG_H = 150
ORIG_W = 290

# クラス定義（nail_seg_v1の学習時と同じ順番）
CLASS_NAMES = {0: "nail", 1: "finger_tip"}

# カメラ
CAM_NAIL = 1
CAM_TIP  = 0

# ROI（撮影スクリプトと同じ）
N_CX, N_CY = 499, 250
N_W0, N_H0 = 282, 409
T_CX, T_CY = 324, 550
T_W0, T_H0 = 182, 136
N_W_SCALE, N_H_SCALE = 1.3, 0.9
T_W_SCALE, T_H_SCALE = 1.7, 1.0
OUT_W_LEFT  = 150
OUT_W_RIGHT = 140


def crop_with_center_wh_safe(img, cx, cy, w, h):
    H, W = img.shape[:2]
    w = int(max(1, min(w, W)))
    h = int(max(1, min(h, H)))
    cx = int(max(w // 2, min(cx, W - w // 2)))
    cy = int(max(h // 2, min(cy, H - h // 2)))
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    return img[y1:y1+h, x1:x1+w], (cx, cy, w, h)


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


def make_concat_bgr(img_nail_bgr, img_tip_bgr):
    n_w = int(N_W0 * N_W_SCALE)
    n_h = int(N_H0 * N_H_SCALE)
    t_w = int(T_W0 * T_W_SCALE)
    t_h = int(T_H0 * T_H_SCALE)
    roi_n, _ = crop_with_center_wh_safe(img_nail_bgr, N_CX, N_CY, n_w, n_h)
    roi_t, _ = crop_with_center_wh_safe(img_tip_bgr,  T_CX, T_CY, t_w, t_h)
    roi_n = _resize_no_pad_center_crop(roi_n, OUT_W_LEFT,  ORIG_H)
    roi_t = _resize_no_pad_center_crop(roi_t, OUT_W_RIGHT, ORIG_H)
    return cv2.hconcat([roi_n, roi_t])


def letterbox(img, new_h, new_w, color=(114, 114, 114)):
    """
    アスペクト比を保ったままリサイズしてパディングする。
    ultralyticsのpredict()が内部でやってるのと同じ処理。
    """
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    dw = (new_w - new_unpad_w) / 2
    dh = (new_h - new_unpad_h) / 2
    resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


def preprocess_for_yolo(concat_bgr):
    """
    concat画像（150×290）をYOLO入力用（160×320）に変換する。
    単純リサイズではなくletterbox（アスペクト比を保ってパディング）を使う。
    これがultralyticsのpredict()と同じ前処理になる。
    """
    lb = letterbox(concat_bgr, YOLO_H, YOLO_W)   # BGR, (160, 320, 3)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))   # HWC → CHW
    x = np.expand_dims(x, axis=0)    # CHW → BCHW
    return x


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolo_onnx_output(output0, output1, conf_threshold=0.5):
    """
    YOLOv11-segのONNX出力をデコードして、
    クラスID・信頼度・マスク係数を取得する。

    output0: (1, 38, 1050) → 検出結果（座標・クラス・マスク係数）
    output1: (1, 32, 40, 80) → マスクのprototype

    YOLOv11-segの出力フォーマット：
      output0[:, 0:4, :]  → x, y, w, h（正規化座標）
      output0[:, 4:6, :]  → クラスのスコア（クラス数=2）
      output0[:, 6:38, :] → マスク係数（32個）
    """
    pred = output0[0]           # (38, 1050)
    proto = output1[0]          # (32, 40, 80)

    # クラススコアと信頼度
    cls_scores = pred[4:6, :]   # (2, 1050)
    cls_ids = np.argmax(cls_scores, axis=0)          # (1050,)
    confidences = cls_scores[cls_ids, np.arange(cls_scores.shape[1])]  # (1050,)

    # 信頼度でフィルタリング
    keep = confidences > conf_threshold
    if not np.any(keep):
        return [], [], []

    cls_ids_kept = cls_ids[keep]
    conf_kept = confidences[keep]
    mask_coefs = pred[6:38, keep].T   # (N, 32)

    return cls_ids_kept, conf_kept, mask_coefs, proto


def generate_masks(cls_ids, mask_coefs, proto, orig_h, orig_w, yolo_h, yolo_w):
    """
    マスク係数とprototypeからマスクを生成して
    元画像サイズ（orig_h × orig_w）にリサイズして返す

    nail_mask:     (orig_h, orig_w) uint8
    tip_mask:      (orig_h, orig_w) uint8
    """
    nail_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    tip_mask  = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if len(cls_ids) == 0:
        return nail_mask, tip_mask

    # proto: (32, ph, pw)
    proto_h, proto_w = proto.shape[1], proto.shape[2]
    proto_flat = proto.reshape(32, -1)  # (32, ph*pw)

    for i, cls_id in enumerate(cls_ids):
        coef = mask_coefs[i]                           # (32,)
        mask_flat = sigmoid(coef @ proto_flat)         # (ph*pw,)
        mask_2d = mask_flat.reshape(proto_h, proto_w)  # (ph, pw)

        # 元画像サイズにリサイズ
        mask_resized = cv2.resize(mask_2d, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_resized > 0.5).astype(np.uint8)

        if CLASS_NAMES[cls_id] == "nail":
            nail_mask = np.logical_or(nail_mask, mask_bin).astype(np.uint8)
        elif CLASS_NAMES[cls_id] == "finger_tip":
            tip_mask = np.logical_or(tip_mask, mask_bin).astype(np.uint8)

    return nail_mask, tip_mask


def main():
    # ── ONNXセッション作成 ───────────────────────────────
    print("=== ONNX読み込み ===")
    print("ONNX_PATH:", ONNX_PATH)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_PATH, providers=providers)

    print("入力名:", [inp.name for inp in session.get_inputs()])
    print("入力shape:", [inp.shape for inp in session.get_inputs()])
    print("出力名:", [out.name for out in session.get_outputs()])
    print("出力shape:", [out.shape for out in session.get_outputs()])
    print()

    input_name = session.get_inputs()[0].name

    # ── カメラ初期化 ──────────────────────────────────────
    cap_nail = cv2.VideoCapture(CAM_NAIL, cv2.CAP_MSMF)
    time.sleep(0.8)
    cap_tip = cv2.VideoCapture(CAM_TIP, cv2.CAP_MSMF)

    if (not cap_nail.isOpened()) or (not cap_tip.isOpened()):
        raise RuntimeError("カメラを開けませんでした")

    print("カメラ起動OK。'q'キーで終了")

    fps_ema = 0.0
    prev_time = time.perf_counter()

    while True:
        ret_n, base_n = cap_nail.read()
        ret_t, base_t = cap_tip.read()
        if not ret_n or not ret_t:
            continue

        concat_bgr = make_concat_bgr(base_n, base_t)

        # ── YOLO推論 ─────────────────────────────────────
        t0 = time.perf_counter()
        x = preprocess_for_yolo(concat_bgr)
        output0, output1 = session.run(None, {input_name: x})
        t1 = time.perf_counter()

        # ── デコード ─────────────────────────────────────
        result = decode_yolo_onnx_output(output0, output1, conf_threshold=0.5)

        if len(result) == 4:
            cls_ids, conf_kept, mask_coefs, proto = result
            nail_mask, tip_mask = generate_masks(
                cls_ids, mask_coefs, proto,
                ORIG_H, ORIG_W, YOLO_H, YOLO_W
            )
        else:
            nail_mask = np.zeros((ORIG_H, ORIG_W), dtype=np.uint8)
            tip_mask  = np.zeros((ORIG_H, ORIG_W), dtype=np.uint8)

        # ── FPS ──────────────────────────────────────────
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_ema = fps_ema * 0.9 + (1.0 / dt) * 0.1

        yolo_ms = (t1 - t0) * 1000

        # ── 可視化 ───────────────────────────────────────
        # 爪マスク：赤色
        # 指先マスク：青色
        disp = concat_bgr.copy()
        disp[nail_mask == 1] = (0, 0, 180)     # 赤（BGR）
        disp[tip_mask  == 1] = (180, 0, 0)     # 青（BGR）

        disp_large = cv2.resize(disp, (ORIG_W * 3, ORIG_H * 3))
        cv2.putText(disp_large, f"FPS: {fps_ema:.1f}  YOLO: {yolo_ms:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp_large, f"nail={int(nail_mask.sum())}px  tip={int(tip_mask.sum())}px",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO ONNX test (red=nail, blue=tip)", disp_large)

        print(f"\rFPS: {fps_ema:.1f}  YOLO: {yolo_ms:.1f}ms  "
              f"nail={int(nail_mask.sum())}px  tip={int(tip_mask.sum())}px", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_nail.release()
    cap_tip.release()
    cv2.destroyAllWindows()
    print("\n終了")


if __name__ == "__main__":
    main()