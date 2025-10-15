# live_compare_G_H_S_fixedROI_gray.py
import cv2
import numpy as np
import time

# ==== 固定ROI（左上座標とサイズ）====
ROI_X, ROI_Y = 260, 315   # 左上座標
ROI_W, ROI_H = 140, 155   # 幅,高さ

S_MIN_MASK = 25  # H表示で低彩度を無効化する下限（0-255）

# ==== 共通フィルタ（全チャンネルに同条件で適用）====
def denoise(gray):
    """全チャンネルで共通のGaussian + CLAHE"""
    blur = cv2.GaussianBlur(gray, (5,5), 1)  # σ=1
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    return clahe.apply(blur)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    recording = False

    print("q: 終了, s: スナップ保存, r: 録画ON/OFF")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        Hf, Wf = frame.shape[:2]
        x = max(0, min(ROI_X, Wf-1))
        y = max(0, min(ROI_Y, Hf-1))
        w = max(1, min(ROI_W, Wf - x))
        h = max(1, min(ROI_H, Hf - y))
        roi = frame[y:y+h, x:x+w].copy()

        # === 各チャンネル ===
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        G = roi[:,:,1]

        # === H/Sを0〜255スケールのグレースケールに変換 ===
        H_gray = cv2.convertScaleAbs(Hc, alpha=255/40.0)
        S_gray = Sc.copy()

        # === 共通ノイズ除去 ===
        G_dn = denoise(G)
        H_dn = denoise(H_gray)
        S_dn = denoise(S_gray)
        Orig_gray = denoise(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

        # === ラベル付け ===
        def put_label(img, text, color=(255,255,255)):
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2, cv2.LINE_AA)
            return bgr

        p1 = put_label(Orig_gray, "Original Gray (denoised)", (0,255,255))
        p2 = put_label(G_dn, "RGB G (denoised)", (0,255,0))
        p3 = put_label(H_dn, f"HSV H (denoised, gray)", (0,215,255))
        p4 = put_label(S_dn, "HSV S (denoised, gray)", (200,200,200))

        # === 2x2パネル ===
        target_w = 400
        scale = target_w / float(p1.shape[1])
        target_h = int(p1.shape[0] * scale)
        rz = lambda x: cv2.resize(x, (target_w, target_h))

        grid_top = np.hstack([rz(p1), rz(p2)])
        grid_bot = np.hstack([rz(p3), rz(p4)])
        panel = np.vstack([grid_top, grid_bot])

        cv2.imshow("Nail: G / H / S (grayscale & CLAHE, fixed ROI)", panel)

        # === 録画処理 ===
        if recording:
            if writer is None:
                Hp, Wp = panel.shape[:2]
                writer = cv2.VideoWriter(f"nail_G_H_S_gray_{int(time.time())}.mp4",
                                         fourcc, 30.0, (Wp, Hp))
            writer.write(panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"nail_gray_panel_{ts}.png", panel)
            print(f"saved: nail_gray_panel_{ts}.png")
        elif key == ord('r'):
            recording = not recording
            if not recording and writer is not None:
                writer.release()
                writer = None
            print("Recording:", "ON" if recording else "OFF")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
