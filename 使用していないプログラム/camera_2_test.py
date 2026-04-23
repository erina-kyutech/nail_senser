# roi_2cam_tuner.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time

# =========================
# 設定（ここだけ変えればOK）
# =========================
CAM_NAIL = 0
CAM_TIP  = 1

BACKEND = cv2.CAP_DSHOW   # WindowsならDSHOW推奨。だめなら cv2.CAP_MSMF に変更
FPS = 30
BUF = 1

# 初期ROI（中心cx,cy + w,h）※最初はだいたいでOK
n_cx, n_cy, n_w, n_h = 497, 803, 209, 301
t_cx, t_cy, t_w, t_h = 776, 653, 836, 1080

# 画面表示の縮小率（高解像度カメラだと重いので）
DISPLAY_SCALE = 0.5  # 0.5なら半分表示（ROI座標は自動で元解像度に換算）

WIN = "ROI_2CAM_TUNER (click:select / drag:move / wheel:resize / r:reset / s:print / q:quit)"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_roi(img, cx, cy, w, h, color=(0,255,0), thick=2):
    x1 = int(cx - w/2); y1 = int(cy - h/2)
    x2 = int(cx + w/2); y2 = int(cy + h/2)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)
    return (x1,y1,x2,y2)

def safe_center(img_shape, cx, cy, w, h):
    H, W = img_shape[:2]
    w = int(max(2, min(w, W)))
    h = int(max(2, min(h, H)))
    cx = int(max(w//2, min(cx, W - w//2)))
    cy = int(max(h//2, min(cy, H - h//2)))
    return cx, cy, w, h

def crop_center(img, cx, cy, w, h):
    cx, cy, w, h = safe_center(img.shape, cx, cy, w, h)
    x1 = int(cx - w/2); y1 = int(cy - h/2)
    x2 = x1 + w; y2 = y1 + h
    return img[y1:y2, x1:x2], (cx,cy,w,h)

def open_cam(idx):
    cap = cv2.VideoCapture(idx, BACKEND)
    time.sleep(0.3)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUF)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap

# ====== マウス操作用グローバル ======
state = {
    "active": "nail",     # "nail" or "tip"
    "drag": False,
    "last_xy": (0,0),
}

def on_mouse(event, x, y, flags, param):
    # param: dict with current frames sizes + roi vars
    global n_cx, n_cy, n_w, n_h, t_cx, t_cy, t_w, t_h

    # 表示座標 -> 元解像度座標に変換
    sx = 1.0 / DISPLAY_SCALE
    sy = 1.0 / DISPLAY_SCALE
    X = int(x * sx)
    Y = int(y * sy)

    nail_w = param["nail_w"]  # 元解像度
    # クリック位置がどっち側かでactive切り替え（左右に並べて表示）
    if X < nail_w:
        which = "nail"
        lx, ly = X, Y
    else:
        which = "tip"
        lx, ly = X - nail_w, Y

    # ホイールでサイズ変更
    if event == cv2.EVENT_MOUSEWHEEL:
        state["active"] = which
        delta = 1 if flags > 0 else -1
        scale = 1.08 if delta > 0 else (1/1.08)

        if which == "nail":
            n_w = int(max(20, n_w * scale))
            n_h = int(max(20, n_h * scale))
        else:
            t_w = int(max(20, t_w * scale))
            t_h = int(max(20, t_h * scale))
        return

    # 左クリックで選択 + ドラッグ開始
    if event == cv2.EVENT_LBUTTONDOWN:
        state["active"] = which
        state["drag"] = True
        state["last_xy"] = (lx, ly)
        # クリックした点を中心にしたい場合はここで更新
        if which == "nail":
            n_cx, n_cy = lx, ly
        else:
            t_cx, t_cy = lx, ly
        return

    # ドラッグ中：中心移動
    if event == cv2.EVENT_MOUSEMOVE and state["drag"]:
        dx = lx - state["last_xy"][0]
        dy = ly - state["last_xy"][1]
        state["last_xy"] = (lx, ly)

        if state["active"] == "nail":
            n_cx += dx
            n_cy += dy
        else:
            t_cx += dx
            t_cy += dy
        return

    if event == cv2.EVENT_LBUTTONUP:
        state["drag"] = False
        return


def print_roi():
    # コピペしやすい形で出す
    print("\n===== ROI RESULT =====")
    print(f"# Nail（爪カメラ）")
    print(f"self.n_cx, self.n_cy = {n_cx}, {n_cy}")
    print(f"self.n_w,  self.n_h  = {n_w}, {n_h}")
    print(f"# Tip（指先カメラ）")
    print(f"self.t_cx, self.t_cy = {t_cx}, {t_cy}")
    print(f"self.t_w,  self.t_h  = {t_w}, {t_h}")
    print("======================\n")


def main():
    global n_cx, n_cy, n_w, n_h, t_cx, t_cy, t_w, t_h

    cap_n = open_cam(CAM_NAIL)
    cap_t = open_cam(CAM_TIP)

    if not cap_n.isOpened() or not cap_t.isOpened():
        print("Camera open failed.")
        print("BACKEND を変える（DSHOW/MSMF）、IDを入れ替える、USBを変える、を試して。")
        return

    # 1フレーム取ってサイズ取得
    ret1, f1 = cap_n.read()
    ret2, f2 = cap_t.read()
    if not ret1 or not ret2:
        print("Initial read failed.")
        return

    nail_h, nail_w = f1.shape[:2]
    tip_h, tip_w   = f2.shape[:2]

    # 表示用キャンバス（左右に連結）
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, int((nail_w + tip_w)*DISPLAY_SCALE), int(max(nail_h, tip_h)*DISPLAY_SCALE))

    param = {"nail_w": nail_w}
    cv2.setMouseCallback(WIN, on_mouse, param)

    # reset用に初期値保存
    init = (n_cx,n_cy,n_w,n_h,t_cx,t_cy,t_w,t_h)

    while True:
        ret1, f1 = cap_n.read()
        ret2, f2 = cap_t.read()
        if not ret1 or not ret2:
            # 途切れる場合は少し待つ
            time.sleep(0.01)
            continue

        # ROIを画面内に収める
        n_cx2, n_cy2, n_w2, n_h2 = safe_center(f1.shape, n_cx, n_cy, n_w, n_h)
        t_cx2, t_cy2, t_w2, t_h2 = safe_center(f2.shape, t_cx, t_cy, t_w, t_h)
        n_cx, n_cy, n_w, n_h = n_cx2, n_cy2, n_w2, n_h2
        t_cx, t_cy, t_w, t_h = t_cx2, t_cy2, t_w2, t_h2

        # ROI枠描画
        col_n = (0,255,0) if state["active"]=="nail" else (200,200,200)
        col_t = (0,255,0) if state["active"]=="tip"  else (200,200,200)

        draw_roi(f1, n_cx, n_cy, n_w, n_h, color=col_n, thick=2)
        draw_roi(f2, t_cx, t_cy, t_w, t_h, color=col_t, thick=2)

        # ROI切り出しプレビュー（右上に小さく出す）
        roi_n, _ = crop_center(f1, n_cx, n_cy, n_w, n_h)
        roi_t, _ = crop_center(f2, t_cx, t_cy, t_w, t_h)
        # 小さくして貼る
        pv_h = int(240)  # プレビュー高さ（固定）
        def resize_keep(img, hh):
            h,w = img.shape[:2]
            ww = int(w * (hh / h))
            return cv2.resize(img, (ww, hh), interpolation=cv2.INTER_AREA)

        pv_n = resize_keep(roi_n, pv_h)
        pv_t = resize_keep(roi_t, pv_h)

        # 連結（高さ合わせ）
        H = max(f1.shape[0], f2.shape[0])
        if f1.shape[0] != H:
            f1p = cv2.copyMakeBorder(f1, 0, H-f1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        else:
            f1p = f1
        if f2.shape[0] != H:
            f2p = cv2.copyMakeBorder(f2, 0, H-f2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        else:
            f2p = f2

        canvas = np.concatenate([f1p, f2p], axis=1)

        # 文字情報
        cv2.putText(canvas, f"Nail ROI: cx,cy={n_cx},{n_cy}  w,h={n_w},{n_h}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Tip  ROI: cx,cy={t_cx},{t_cy}  w,h={t_w},{t_h}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(canvas, "Keys: s=print ROI / r=reset / q=quit",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # プレビューを左上に貼る（重なるけどOK）
        y0 = 120
        x0 = 10
        canvas[y0:y0+pv_n.shape[0], x0:x0+pv_n.shape[1]] = pv_n
        x1 = x0 + pv_n.shape[1] + 10
        canvas[y0:y0+pv_t.shape[0], x1:x1+pv_t.shape[1]] = pv_t
        cv2.rectangle(canvas, (x0,y0), (x0+pv_n.shape[1], y0+pv_n.shape[0]), (255,255,255), 2)
        cv2.rectangle(canvas, (x1,y0), (x1+pv_t.shape[1], y0+pv_t.shape[0]), (255,255,255), 2)

        # 表示
        disp = cv2.resize(canvas, (int(canvas.shape[1]*DISPLAY_SCALE), int(canvas.shape[0]*DISPLAY_SCALE)),
                          interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN, disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
        if k == ord('s'):
            print_roi()
        if k == ord('r'):
            n_cx,n_cy,n_w,n_h,t_cx,t_cy,t_w,t_h = init
            print("reset.")

    cap_n.release()
    cap_t.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
