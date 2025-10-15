# live_compare_G_H_S_V_fixedROI_gray.py
import cv2
import numpy as np
import time

# ==== 固定ROI ====
ROI_X, ROI_Y = 260, 315   # 左上座標
ROI_W, ROI_H = 140, 155   # 幅,高さ

S_MIN_MASK = 25  # H表示で低彩度を無効化する下限（0-255）

# ==== 共通フィルタ ====
def denoise(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    return clahe.apply(blur)

# ==== 数値情報をまとめて描画 ====
def draw_info_panel(H, S, V):
    H_mean, S_mean, V_mean = np.mean(H), np.mean(S), np.mean(V)
    H_std,  S_std,  V_std  = np.std(H),  np.std(S),  np.std(V)
    H_min,  S_min,  V_min  = np.min(H),  np.min(S),  np.min(V)
    H_max,  S_max,  V_max  = np.max(H),  np.max(S),  np.max(V)

    panel = np.zeros((250, 400, 3), dtype=np.uint8)
    cv2.putText(panel, "=== HSV Stats (ROI) ===", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"H mean:{H_mean:6.2f} std:{H_std:6.2f} min:{H_min:5.1f} max:{H_max:5.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"S mean:{S_mean:6.2f} std:{S_std:6.2f} min:{S_min:5.1f} max:{S_max:5.1f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(panel, f"V mean:{V_mean:6.2f} std:{V_std:6.2f} min:{V_min:5.1f} max:{V_max:5.1f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
    cv2.putText(panel, f"H range: {np.max(H)-np.min(H):6.2f}", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
    return panel

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

        # === HSV分解 ===
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        G = roi[:,:,1]

        # === H/S/Vをグレースケールに変換 ===
        H_gray = cv2.convertScaleAbs(Hc, alpha=255/40.0)  # 赤〜黄を拡大
        S_gray = Sc.copy()
        V_gray = Vc.copy()

        # === 共通フィルタ ===
        Orig_gray = denoise(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        G_dn = denoise(G)
        H_dn = denoise(H_gray)
        S_dn = denoise(S_gray)
        V_dn = denoise(V_gray)

        # === ラベル付け ===
        def put_label(img, text, color=(255,255,255)):
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2, cv2.LINE_AA)
            return bgr

        p1 = put_label(Orig_gray, "Original Gray", (0,255,255))
        p2 = put_label(G_dn, "RGB G (denoised)", (0,255,0))
        p3 = put_label(H_dn, "HSV H (denoised)", (0,215,255))
        p4 = put_label(S_dn, "HSV S (denoised)", (200,200,200))
        p5 = put_label(V_dn, "HSV V (denoised)", (255,255,255))

        # === 数値情報パネル ===
        info_panel = draw_info_panel(Hc, Sc, Vc)

        # === 3x2パネル配置 ===
        target_w = 400
        scale = target_w / float(p1.shape[1])
        target_h = int(p1.shape[0] * scale)
        rz = lambda x: cv2.resize(x, (target_w, target_h))

        row1 = np.hstack([rz(p1), rz(p2)])
        row2 = np.hstack([rz(p3), rz(p4)])
        row3 = np.hstack([rz(p5), rz(info_panel)])
        panel = np.vstack([row1, row2, row3])

        cv2.imshow("Nail: G / H / S / V (grayscale + stats, fixed ROI)", panel)

        # === 録画処理 ===
        if recording:
            if writer is None:
                Hp, Wp = panel.shape[:2]
                writer = cv2.VideoWriter(f"nail_G_H_S_V_gray_{int(time.time())}.mp4",
                                         fourcc, 30.0, (Wp, Hp))
            writer.write(panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"nail_gray_HSV_{ts}.png", panel)
            print(f"saved: nail_gray_HSV_{ts}.png")
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
