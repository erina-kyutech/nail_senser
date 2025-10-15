# live_compare_G_H_S_V_grid_all_in_one.py
import cv2
import numpy as np
import time

# ===== 固定ROI =====
ROI_X, ROI_Y = 260, 315
ROI_W, ROI_H = 140, 155

# ===== レイアウト設定（ここだけ触れば全体が整う）=====
TILE_W = 320          # 1タイルの横幅（全体は 3*TILE_W + 余白）
GAP = 10              # タイル間の余白(px)
LABEL_H = 34          # 各タイル上部のラベル帯の高さ
FOOTER_H = 76         # 画面下の統計情報帯の高さ
BORDER = 12           # 外枠余白

# ===== 共通フィルタ =====
def denoise(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    return clahe.apply(blur)

# ===== タイル生成（サイズ/ラベル/余白を一括で整える）=====
def make_tile(gray_u8, label, color=(255,255,255)):
    # 中身をTILE_Wに合わせてリサイズ（アスペクト保持）
    h0, w0 = gray_u8.shape[:2]
    scale = TILE_W / float(w0)
    h1 = int(round(h0 * scale))
    img = cv2.resize(gray_u8, (TILE_W, h1))
    # ラベル帯を上に追加
    band = np.zeros((LABEL_H, TILE_W), dtype=np.uint8)
    band_bgr = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
    cv2.putText(band_bgr, label, (10, LABEL_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(band_bgr, label, (10, LABEL_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    # グレー→BGR化して結合
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    tile = np.vstack([band_bgr, bgr])
    return tile  # (LABEL_H + h1, TILE_W, 3)

# ===== フッター（数値）=====
def draw_footer(H, S, V, gray0, width):
    # 統計（生のH/S/V/Grayで計算）
    H_mean, S_mean, V_mean = float(np.mean(H)), float(np.mean(S)), float(np.mean(V))
    H_std,  S_std,  V_std  = float(np.std(H)),  float(np.std(S)),  float(np.std(V))
    H_min,  S_min,  V_min  = float(np.min(H)),  float(np.min(S)),  float(np.min(V))
    H_max,  S_max,  V_max  = float(np.max(H)),  float(np.max(S)),  float(np.max(V))
    # GrayとVの相関/誤差
    g = gray0.astype(np.float32).ravel(); v = V.astype(np.float32).ravel()
    g -= g.mean(); v -= v.mean()
    corr = float(np.dot(g, v) / (np.linalg.norm(g)*np.linalg.norm(v) + 1e-9))
    mse  = float(np.mean((gray0.astype(np.float32)-V.astype(np.float32))**2))
    # 描画
    foot = np.zeros((FOOTER_H, width, 3), dtype=np.uint8)
    def put(x, y, txt, col): cv2.putText(foot, txt, (x, y),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    put(10, 26, "=== HSV Stats (ROI) ===", (0,255,255))
    put(10, 54, f"H mean:{H_mean:6.2f} std:{H_std:5.2f} min:{H_min:5.1f} max:{H_max:5.1f}", (0,215,255))
    put(350,54, f"S mean:{S_mean:6.2f} std:{S_std:5.2f} min:{S_min:5.1f} max:{S_max:5.1f}", (0,255,0))
    put(690,54, f"V mean:{V_mean:6.2f} std:{V_std:5.2f} min:{V_min:5.1f} max:{V_max:5.1f}", (220,220,220))
    put(980,26, f"corr(Gray,V): {corr:.4f}  MSE: {mse:.1f}", (255,255,255))
    return foot

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

        Hf, Wf = frame.shape[:2]
        x = max(0, min(ROI_X, Wf-1)); y = max(0, min(ROI_Y, Hf-1))
        w = max(1, min(ROI_W, Wf-x));  h = max(1, min(ROI_H, Hf-y))
        roi = frame[y:y+h, x:x+w].copy()

        # --- チャンネル ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vc = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        G  = roi[:,:,1]
        gray0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 表示用グレー（Hは赤〜黄を強調）
        H_gray = cv2.convertScaleAbs(Hc, alpha=255/40.0)
        S_gray = Sc.copy()
        V_gray = Vc.copy()

        # 統一フィルタ（見やすさ）
        Orig = denoise(gray0)
        Gv   = denoise(G)
        Hv   = denoise(H_gray)
        Sv   = denoise(S_gray)
        Vv   = denoise(V_gray)
        # Diff（見やすいように×3 → denoise）
        diff = cv2.convertScaleAbs(cv2.absdiff(gray0, Vc), alpha=3.0)
        diffv = denoise(diff)

        # タイル化（各タイルは同じ幅＆上部ラベル帯つき）
        t1 = make_tile(Orig, "Original Gray", (0,255,255))
        t2 = make_tile(Gv,   "RGB G (denoised)", (0,255,0))
        t3 = make_tile(Hv,   "HSV H (denoised)", (0,215,255))
        t4 = make_tile(Sv,   "HSV S (denoised)", (200,200,200))
        t5 = make_tile(Vv,   "HSV V (denoised)", (255,255,255))
        t6 = make_tile(diffv,"Diff |Gray - V| x3", (255,255,255))

        # 各行を横結合（高さを揃えるため、短い方に下パディング）
        def hstack_same_height(a, b, gap=GAP):
            h = max(a.shape[0], b.shape[0])
            pad_a = h - a.shape[0]
            pad_b = h - b.shape[0]
            if pad_a>0: a = cv2.copyMakeBorder(a, 0, pad_a, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            if pad_b>0: b = cv2.copyMakeBorder(b, 0, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            sep = np.zeros((h, gap, 3), dtype=np.uint8)
            return np.hstack([a, sep, b])

        row_left  = hstack_same_height(t1, t2)         # [t1 | t2]
        row_right = hstack_same_height(t3, t4)         # [t3 | t4]
        row1 = hstack_same_height(row_left, row_right) # [t1|t2 | t3|t4]

        row_left2  = hstack_same_height(t5, t6)        # [t5 | t6]
        # ダミー（上段と同じ幅にするためのスペーサ）→ 2x3にするので右側は空にしない
        # ここでは 2x3 を作る：上段(3つ)・下段(3つ)
        # row1 は t1|t2|t3、row2 は t4|t5|t6 に並べ替える方が自然なので並べ替え実施
        row1 = hstack_same_height(hstack_same_height(t1, t2), t3)
        row2 = hstack_same_height(hstack_same_height(t4, t5), t6)

        # 幅揃え＆外枠
        width = max(row1.shape[1], row2.shape[1]) + BORDER*2
        # 行の幅を合わせる
        def pad_to_width(img, w):
            if img.shape[1] < w:
                pad = w - img.shape[1]
                return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
            return img
        row1 = pad_to_width(row1, width - BORDER*2)
        row2 = pad_to_width(row2, width - BORDER*2)

        # フッター作成
        footer = draw_footer(Hc, Sc, Vc, gray0, width - BORDER*2)

        # 全面キャンバス
        sep_h = np.zeros((GAP, width - BORDER*2, 3), dtype=np.uint8)
        inner = np.vstack([row1, sep_h, row2, sep_h, footer])
        panel = cv2.copyMakeBorder(inner, BORDER, BORDER, BORDER, BORDER,
                                   cv2.BORDER_CONSTANT, value=(0,0,0))

        cv2.imshow("Nail: G / H / S / V (all-in-one grid + footer)", panel)

        # 録画
        if recording:
            if writer is None:
                Hcvs, Wcvs = panel.shape[:2]
                writer = cv2.VideoWriter(f"nail_G_H_S_V_all_{int(time.time())}.mp4",
                                         fourcc, 30.0, (Wcvs, Hcvs))
            writer.write(panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(f"nail_HSV_all_{ts}.png", panel)
            print("saved:", f"nail_HSV_all_{ts}.png")
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
