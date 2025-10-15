# live_compare_G_H_S_V_fixedROI_gray_diff_fixed.py
import cv2
import numpy as np
import time

# ==== 固定ROI ====
ROI_X, ROI_Y = 260, 315   # 左上座標
ROI_W, ROI_H = 140, 155   # 幅,高さ

# ==== 共通フィルタ ====
def denoise(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    return clahe.apply(blur)

# ==== 数値情報パネル ====
def draw_info_panel(H, S, V, gray0):
    H_mean, S_mean, V_mean = np.mean(H), np.mean(S), np.mean(V)
    H_std,  S_std,  V_std  = np.std(H),  np.std(S),  np.std(V)
    H_min,  S_min,  V_min  = np.min(H),  np.min(S),  np.min(V)
    H_max,  S_max,  V_max  = np.max(H),  np.max(S),  np.max(V)

    # Gray と V の関係
    g = gray0.astype(np.float32).ravel()
    v = V.astype(np.float32).ravel()
    g -= g.mean(); v -= v.mean()
    denom = (np.linalg.norm(g) * np.linalg.norm(v) + 1e-9)
    corr = float(np.dot(g, v) / denom)
    mse  = float(np.mean((gray0.astype(np.float32) - V.astype(np.float32))**2))

    panel = np.zeros((260, 420, 3), dtype=np.uint8)
    put = lambda y, text, color: cv2.putText(panel, text, (10, y),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    put(28,  "=== HSV Stats (ROI) ===", (0,255,255))
    put(65,  f"H mean:{H_mean:6.2f}  std:{H_std:6.2f}  min:{H_min:5.1f}  max:{H_max:5.1f}", (0,215,255))
    put(100, f"S mean:{S_mean:6.2f}  std:{S_std:6.2f}  min:{S_min:5.1f}  max:{S_max:5.1f}", (0,255,0))
    put(135, f"V mean:{V_mean:6.2f}  std:{V_std:6.2f}  min:{V_min:5.1f}  max:{V_max:5.1f}", (220,220,220))
    put(170, f"H range: {H_max-H_min:6.2f}", (0,255,255))
    put(205, f"corr(Gray,V): {corr:.4f}   MSE(Gray,V): {mse:.2f}", (255,255,255))
    put(240, "Press 'd' to toggle V/Diff", (180,180,255))
    return panel

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    recording = False
    show_diff = False   # 'd' で V と Diff の切替

    print("q:終了, s:スナップ保存, r:録画ON/OFF, d:V/Diff切替")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        Hf, Wf = frame.shape[:2]
        x = max(0, min(ROI_X, Wf-1)); y = max(0, min(ROI_Y, Hf-1))
        w = max(1, min(ROI_W, Wf - x)); h = max(1, min(ROI_H, Hf - y))
        roi = frame[y:y+h, x:x+w].copy()

        # HSV分解 & 各チャンネル
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        G  = roi[:,:,1]
        gray0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 生Gray（相関用）

        # 表示用グレイスケール（Hは赤〜黄を拡大）
        H_gray = cv2.convertScaleAbs(Hc, alpha=255/40.0)
        S_gray = Sc.copy()
        V_gray = Vc.copy()

        # 統一フィルタ
        Orig_gray = denoise(gray0)
        G_dn  = denoise(G)
        H_dn  = denoise(H_gray)
        S_dn  = denoise(S_gray)
        V_dn  = denoise(V_gray)

        # ラベル
        def put_label(img, text, color=(255,255,255)):
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            return bgr

        p1 = put_label(Orig_gray, "Original Gray", (0,255,255))
        p2 = put_label(G_dn,      "RGB G (denoised)", (0,255,0))
        p3 = put_label(H_dn,      "HSV H (denoised)", (0,215,255))
        p4 = put_label(S_dn,      "HSV S (denoised)", (200,200,200))

        # 3段目左：V か Diff
        if show_diff:
            diff = cv2.absdiff(gray0, Vc)                  # 生の差
            diff = cv2.convertScaleAbs(diff, alpha=3.0)    # 見やすさ×3
            diff_dn = denoise(diff)
            p5 = put_label(diff_dn, "Diff |Gray - V| x3", (255,255,255))
        else:
            p5 = put_label(V_dn,     "HSV V (denoised)", (255,255,255))

        # 右下の統計パネル
        info_panel = draw_info_panel(Hc, Sc, Vc, gray0)

        # === レイアウト ===
        target_w = 400
        def rz(x):
            h0,w0 = x.shape[:2]
            th = int(h0 * (target_w/float(w0)))
            return cv2.resize(x, (target_w, th))

        row1 = np.hstack([rz(p1), rz(p2)])
        row2 = np.hstack([rz(p3), rz(p4)])

        rp5 = rz(p5)
        info_resized = cv2.resize(info_panel, (info_panel.shape[1], rp5.shape[0]))
        row3 = np.hstack([rp5, info_resized])

        # --- 👇ここを追加：幅を統一してから結合 ---
        min_w = min(row1.shape[1], row2.shape[1], row3.shape[1])
        row1 = cv2.resize(row1, (min_w, row1.shape[0]))
        row2 = cv2.resize(row2, (min_w, row2.shape[0]))
        row3 = cv2.resize(row3, (min_w, row3.shape[0]))

        panel = np.vstack([row1, row2, row3])


        cv2.imshow("Nail: G / H / S / V (grayscale + stats + diff)", panel)

        # 録画
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
            ts = int(time.time()); cv2.imwrite(f"nail_gray_HSV_{ts}.png", panel)
            print(f"saved: nail_gray_HSV_{ts}.png")
        elif key == ord('r'):
            recording = not recording
            if not recording and writer is not None:
                writer.release(); writer = None
            print("Recording:", "ON" if recording else "OFF")
        elif key == ord('d'):
            show_diff = not show_diff
            print("Display:", "Diff" if show_diff else "V")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
