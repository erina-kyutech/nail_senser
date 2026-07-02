# -*- coding: utf-8 -*-
"""
test_yolo_ultralytics_tf_env.py

TensorFlow環境でultralyticsが動くか確認するテスト。
YOLOはCPUで推論、VGG16はGPUで推論、という構成が成立するか確認する。
"""

import cv2
import time
import threading
import numpy as np

# ── まずultralyticsだけimportして確認 ────────────────────
print("=== ultralytics import テスト ===")
from ultralytics import YOLO
print("ultralytics: OK")

# ── 次にTensorFlowをimport ────────────────────────────────
print("=== TensorFlow import テスト ===")
import tensorflow as tf
print("TensorFlow: OK")
print("GPU利用可能:", tf.config.list_physical_devices('GPU'))

# =========================================================
# 設定
# =========================================================
YOLO_WEIGHT_PATH = r"C:\Users\Owner\PycharmProjects\YOLO_nail_seg\runs\segment\runs\segment\nail_seg_v1\weights\best.pt"

CAM_NAIL = 1
CAM_TIP  = 0

OUT_H = 150
OUT_W_LEFT  = 150
OUT_W_RIGHT = 140

N_CX, N_CY = 499, 250
N_W0, N_H0 = 282, 409
T_CX, T_CY = 324, 550
T_W0, T_H0 = 182, 136
N_W_SCALE, N_H_SCALE = 1.3, 0.9
T_W_SCALE, T_H_SCALE = 1.7, 1.0


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
    roi_n = _resize_no_pad_center_crop(roi_n, OUT_W_LEFT,  OUT_H)
    roi_t = _resize_no_pad_center_crop(roi_t, OUT_W_RIGHT, OUT_H)
    return cv2.hconcat([roi_n, roi_t])


def extract_masks(result, orig_h, orig_w):
    """ultralyticsのresultからnail/finger_tipのマスクを取得する"""
    nail_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    tip_mask  = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if result.masks is None:
        return nail_mask, tip_mask

    names = result.names
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    masks_data = result.masks.data.cpu().numpy()  # (N, mh, mw)

    for i, cls_id in enumerate(cls_ids):
        m = masks_data[i]
        if m.shape[0] != orig_h or m.shape[1] != orig_w:
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m > 0.5).astype(np.uint8)

        if names[cls_id] == "nail":
            nail_mask = np.logical_or(nail_mask, m_bin).astype(np.uint8)
        elif names[cls_id] == "finger_tip":
            tip_mask = np.logical_or(tip_mask, m_bin).astype(np.uint8)

    return nail_mask, tip_mask


def main():
    print("\n=== YOLOモデル読み込み（CPUで動かす） ===")
    yolo = YOLO(YOLO_WEIGHT_PATH)
    print("YOLO読み込み: OK")

    cap_nail = cv2.VideoCapture(CAM_NAIL, cv2.CAP_MSMF)
    time.sleep(0.8)
    cap_tip = cv2.VideoCapture(CAM_TIP, cv2.CAP_MSMF)
    if (not cap_nail.isOpened()) or (not cap_tip.isOpened()):
        raise RuntimeError("カメラを開けませんでした")

    # カメラスレッド
    _frame_nail = [None]
    _frame_tip  = [None]
    _lock_nail  = threading.Lock()
    _lock_tip   = threading.Lock()
    _running    = [True]

    def read_nail():
        while _running[0]:
            ret, f = cap_nail.read()
            if ret and f is not None:
                with _lock_nail:
                    _frame_nail[0] = f

    def read_tip():
        while _running[0]:
            ret, f = cap_tip.read()
            if ret and f is not None:
                with _lock_tip:
                    _frame_tip[0] = f

    threading.Thread(target=read_nail, daemon=True).start()
    threading.Thread(target=read_tip,  daemon=True).start()
    time.sleep(0.5)

    fps_ema = 0.0
    prev_time = time.perf_counter()

    print("準備OK。'q'キーで終了")
    print("赤=爪(nail)  青=指先(finger_tip)")

    while True:
        with _lock_nail:
            base_n = _frame_nail[0]
        with _lock_tip:
            base_t = _frame_tip[0]
        if base_n is None or base_t is None:
            continue

        concat_bgr = make_concat_bgr(base_n, base_t)

        # ★ YOLOはCPUで推論（device="cpu"）
        t0 = time.perf_counter()
        results = yolo.predict(source=concat_bgr, device="cpu", verbose=False)
        t1 = time.perf_counter()

        nail_mask, tip_mask = extract_masks(results[0], OUT_H, OUT_W_LEFT + OUT_W_RIGHT)

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_ema = fps_ema * 0.9 + (1.0 / dt) * 0.1

        yolo_ms = (t1 - t0) * 1000

        # 可視化
        disp = concat_bgr.copy()
        disp[nail_mask == 1] = (0, 0, 180)    # 赤 = 爪
        disp[tip_mask  == 1] = (180, 0, 0)    # 青 = 指先

        disp_large = cv2.resize(disp, ((OUT_W_LEFT + OUT_W_RIGHT) * 3, OUT_H * 3))
        cv2.putText(disp_large, f"FPS: {fps_ema:.1f}  YOLO(CPU): {yolo_ms:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp_large, f"nail={int(nail_mask.sum())}px  tip={int(tip_mask.sum())}px",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLO ultralytics (CPU) test", disp_large)

        print(f"\rFPS: {fps_ema:.1f}  YOLO(CPU): {yolo_ms:.1f}ms  "
              f"nail={int(nail_mask.sum())}px  tip={int(tip_mask.sum())}px", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    _running[0] = False
    cap_nail.release()
    cap_tip.release()
    cv2.destroyAllWindows()
    print("\n終了")


if __name__ == "__main__":
    main()