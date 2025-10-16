# live_compare_G_H_S_V_HS_readable.py
import cv2
import numpy as np
import time

# ==== 固定ROI ====
ROI_X, ROI_Y = 260, 315
ROI_W, ROI_H = 140, 155

# ==== レイアウト調整 ====
TILE_W   = 320      # タイル横幅（入らなければ 300/280 に）
LABEL_H  = 44       # ラベル帯の高さ（文字が切れないよう広め）
GAP      = 10       # タイル間の余白
FOOTER_H = 96       # 統計帯の高さ
MAX_W    = 1280     # 最終出力の最大幅
MAX_H    = 720      # 最終出力の最大高さ

# ==== 共通フィルタ ====
def denoise(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    return clahe.apply(blur)

# ==== ラベル付きタイル（切れない＆読みやすい）====
def make_tile_gray(gray_u8, label, color=(255,255,255)):
    h0, w0 = gray_u8.shape[:2]
    scale = TILE_W / float(w0)
    h1 = int(round(h0 * scale))
    img = cv2.resize(gray_u8, (TILE_W, h1))
    # ラベル帯
    band = np.zeros((LABEL_H, TILE_W), dtype=np.uint8)
    band_bgr = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
    # 太字＋縁取りで可読性UP
    cv2.putText(band_bgr, label, (10, LABEL_H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(band_bgr, label, (10, LABEL_H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.vstack([band_bgr, bgr])

def make_tile_color(bgr_u8, label, color=(255,255,255)):
    h0, w0 = bgr_u8.shape[:2]
    scale = TILE_W / float(w0)
    h1 = int(round(h0 * scale))
    img = cv2.resize(bgr_u8, (TILE_W, h1))
    band = np.zeros((LABEL_H, TILE_W, 3), dtype=np.uint8)
    cv2.putText(band, label, (10, LABEL_H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(band, label, (10, LABEL_H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return np.vstack([band, img])

# ==== 横結合（高さそろえ）====
def hstack_same_height(a, b, gap=GAP):
    h = max(a.shape[0], b.shape[0])
    if a.shape[0] < h:
        a = cv2.copyMakeBorder(a, 0, h-a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    if b.shape[0] < h:
        b = cv2.copyMakeBorder(b, 0, h-b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    sep = np.zeros((h, gap, 3), dtype=np.uint8)
    return np.hstack([a, sep, b])

# ==== フッター（重ならない3カラム）====
def draw_footer(H, S, V, HS, width):
    def ms(x): return float(np.mean(x)), float(np.std(x))
    Hm,Hs = ms(H); Sm,Ss = ms(S); Vm,Vs = ms(V); HSm,HSs = ms(HS)
    foot = np.zeros((FOOTER_H, width, 3), dtype=np.uint8)
    cv2.putText(foot, "=== HSV Stats (ROI) ===", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2, cv2.LINE_AA)
    col = width//3
    def put(x,y,txt,colr):
        cv2.putText(foot, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, colr, 2, cv2.LINE_AA)
    put(10,         56, f"H  mean:{Hm:6.2f} std:{Hs:5.2f}", (0,215,255))
    put(10,         86, f"S  mean:{Sm:6.2f} std:{Ss:5.2f}", (0,255,0))
    put(col+10,     56, f"V  mean:{Vm:6.2f} std:{Vs:5.2f}", (200,200,200))
    put(2*col+10,   56, f"H+S mean:{HSm:6.2f} std:{HSs:5.2f}", (255,180,100))
    return foot

# ==== 最終サイズを画面にフィット ====
def fit_to_screen(img, max_w=MAX_W, max_h=MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    recording = False

    print("q:終了, s:スナップ保存, r:録画ON/OFF")
    while True:
        ok, frame = cap.read()
        if not ok: break

        # ROI
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W].copy()

        # 分解
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        G  = roi[:,:,1]
        gray0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 表示用
        Hn = cv2.convertScaleAbs(Hc, alpha=255/30.0)
        HS  = cv2.addWeighted(Hn, 0.5, Sc, 0.5, 0)

        # フィルタ
        Orig = denoise(gray0); Gv = denoise(G)
        Hv = denoise(Hn); Sv = denoise(Sc); Vv = denoise(Vc); HSv = denoise(HS)

        # タイル（3×2）
        t1 = make_tile_gray(Orig, "Original",      (0,255,255))
        t2 = make_tile_gray(Gv,   "RGB G",         (0,255,0))
        t3 = make_tile_gray(Hv,   "HSV H",         (0,215,255))
        t4 = make_tile_gray(Sv,   "HSV S",         (200,200,200))
        t5 = make_tile_gray(Vv,   "HSV V",         (255,255,255))
        t6 = make_tile_gray(HSv,  "HSV H+S",       (255,180,100))

        row1 = hstack_same_height(hstack_same_height(t1, t2), t3)
        row2 = hstack_same_height(hstack_same_height(t4, t5), t6)

        # 同幅に
        width_inner = max(row1.shape[1], row2.shape[1])
        def pad_w(img, w):
            if img.shape[1] < w:
                pad = w - img.shape[1]
                return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
            return img
        row1 = pad_w(row1, width_inner)
        row2 = pad_w(row2, width_inner)

        footer = draw_footer(Hc, Sc, Vc, HS, width_inner)
        sep = np.zeros((GAP, width_inner, 3), dtype=np.uint8)
        panel = np.vstack([row1, sep, row2, sep, footer])

        # 画面にフィット
        panel = fit_to_screen(panel, MAX_W, MAX_H)

        cv2.imshow("HSV Comparison (G/H/S/V/H+S)", panel)

        # 録画
        if recording:
            if writer is None:
                Hcvs, Wcvs = panel.shape[:2]
                writer = cv2.VideoWriter(f"nail_HSV_HS_readable_{int(time.time())}.mp4",
                                         fourcc, 30.0, (Wcvs, Hcvs))
            writer.write(panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fn = f"nail_HSV_HS_panel_{int(time.time())}.png"
            cv2.imwrite(fn, panel)
            print("saved:", fn)
        elif key == ord('r'):
            recording = not recording
            if not recording and writer is not None:
                writer.release(); writer = None
            print("Recording:", "ON" if recording else "OFF")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
