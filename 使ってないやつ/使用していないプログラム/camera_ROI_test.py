import cv2
import numpy as np
import os

# =========================
# 設定（ここだけ自分の環境に合わせる）
# =========================
NAIL_IMG_PATH = r"nail_sample.jpg"
TIP_IMG_PATH  = r"tip_sample.jpg"

OUT_H = 100                 # ★左右とも高さはこれに揃える
OUT_W_LEFT  = 100            # 左(爪)の幅
OUT_W_RIGHT = 100           # 右(指先)の幅
DO_CONCAT = True
SAVE_DIR = "roi_debug"

# =========================
# 便利関数
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_with_center_wh(img, cx, cy, w, h):
    H, W = img.shape[:2]
    w = max(2, int(w))
    h = max(2, int(h))

    x1 = clamp(int(cx - w/2), 0, W - 1)
    y1 = clamp(int(cy - h/2), 0, H - 1)
    x2 = clamp(int(cx + w/2), 1, W)
    y2 = clamp(int(cy + h/2), 1, H)

    roi = img[y1:y2, x1:x2]
    rect = (x1, y1, x2, y2)
    return roi, rect

def draw_box(img, rect, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = rect
    out = img.copy()
    cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
    return out

def letterbox_resize(img, out_w, out_h):
    """アスペクト比維持でリサイズし、足りない部分を黒埋め（歪み防止）"""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=img.dtype)

    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((out_h, out_w, 3), dtype=img.dtype)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def show_scaled(win, img, max_w=1200):
    """表示用：横幅が大きすぎるときだけ縮小（縦横比維持）"""
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h*scale)), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(win, img)

# =========================
# 読み込み
# =========================
nail = cv2.imread(NAIL_IMG_PATH)
tip  = cv2.imread(TIP_IMG_PATH)

if nail is None:
    raise FileNotFoundError(f"読めない: {NAIL_IMG_PATH} / cwd={os.getcwd()}")
if tip is None:
    raise FileNotFoundError(f"読めない: {TIP_IMG_PATH} / cwd={os.getcwd()}")

nH, nW = nail.shape[:2]
tH, tW = tip.shape[:2]

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 初期値（あなたの画像傾向に合わせ）
# =========================
init_nx = int(nW * 0.50)
init_ny = int(nH * 0.72)
init_nw = max(40, int(min(nW, nH) * 0.35))
init_nh = max(40, int(min(nW, nH) * 0.25))

init_tx = int(tW * 0.72)
init_ty = int(tH * 0.50)
init_tw = max(60, int(min(tW, tH) * 0.55))
init_th = max(60, int(min(tW, tH) * 0.40))

# =========================
# UI（トラックバー）
# =========================
WIN = "CTRL"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

def add_bar(name, val, mx):
    cv2.createTrackbar(name, WIN, int(val), int(mx), lambda x: None)

add_bar("Nail cx(0-1000)", int(init_nx / nW * 1000), 1000)
add_bar("Nail cy(0-1000)", int(init_ny / nH * 1000), 1000)
add_bar("Nail w(px)", init_nw, nW)
add_bar("Nail h(px)", init_nh, nH)

add_bar("Tip  cx(0-1000)", int(init_tx / tW * 1000), 1000)
add_bar("Tip  cy(0-1000)", int(init_ty / tH * 1000), 1000)
add_bar("Tip  w(px)", init_tw, tW)
add_bar("Tip  h(px)", init_th, tH)

add_bar("SAVE(0/1)", 0, 1)

print("操作:")
print(" - CTRLウィンドウのトラックバーでROIを調整")
print(" - SAVE(0/1)=1 で net入力画像とパラメータ保存")
print(" - q で終了")

save_id = 0

def resize_no_pad_center_crop(img, out_w, out_h):
    """黒パディング無し。アスペクト比を合わせるため中心クロップしてからリサイズ。"""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, 3), dtype=img.dtype)

    target = out_w / out_h
    cur = w / h

    if cur > target:
        # 横が広い → 横を切る
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        cropped = img[:, x0:x0+new_w]
    else:
        # 縦が長い → 縦を切る
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        cropped = img[y0:y0+new_h, :]

    interp = cv2.INTER_AREA if (cropped.shape[0] > out_h or cropped.shape[1] > out_w) else cv2.INTER_LINEAR
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)

# =========================
# メインループ
# =========================
while True:
    nx = int(cv2.getTrackbarPos("Nail cx(0-1000)", WIN) / 1000 * nW)
    ny = int(cv2.getTrackbarPos("Nail cy(0-1000)", WIN) / 1000 * nH)
    nw = max(10, cv2.getTrackbarPos("Nail w(px)", WIN))
    nh = max(10, cv2.getTrackbarPos("Nail h(px)", WIN))

    tx = int(cv2.getTrackbarPos("Tip  cx(0-1000)", WIN) / 1000 * tW)
    ty = int(cv2.getTrackbarPos("Tip  cy(0-1000)", WIN) / 1000 * tH)
    tw = max(10, cv2.getTrackbarPos("Tip  w(px)", WIN))
    th = max(10, cv2.getTrackbarPos("Tip  h(px)", WIN))

    nail_roi, nrect = crop_with_center_wh(nail, nx, ny, nw, nh)
    tip_roi,  trect = crop_with_center_wh(tip,  tx, ty, tw, th)

    nail_box = draw_box(nail, nrect)
    tip_box  = draw_box(tip,  trect)

    # ネット入力用（高さは必ずOUT_Hに揃う）
    nail_in = resize_no_pad_center_crop(nail_roi, OUT_W_LEFT, OUT_H)
    tip_in = resize_no_pad_center_crop(tip_roi, OUT_W_RIGHT, OUT_H)

    if DO_CONCAT:
        net_in = np.concatenate([nail_in, tip_in], axis=1)  # (OUT_H, OUT_W_LEFT+OUT_W_RIGHT, 3)
    else:
        net_in = tip_in

    # ---- 別ウィンドウ表示 ----
    show_scaled("VIEW_NAIL (boxed)", nail_box, max_w=1200)
    show_scaled("VIEW_TIP  (boxed)", tip_box,  max_w=1200)

    show_scaled("NET_NAIL (OUT_H fixed)", nail_in, max_w=600)
    show_scaled("NET_TIP  (OUT_H fixed)", tip_in,  max_w=600)
    show_scaled("NET_CONCAT", net_in, max_w=900)

    # 保存
    if cv2.getTrackbarPos("SAVE(0/1)", WIN) == 1:
        save_id += 1
        img_path = os.path.join(SAVE_DIR, f"net_input_{save_id:03d}.png")
        txt_path = os.path.join(SAVE_DIR, f"roi_params_{save_id:03d}.txt")

        cv2.imwrite(img_path, net_in)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"NAIL_IMG_PATH={NAIL_IMG_PATH}\n")
            f.write(f"TIP_IMG_PATH={TIP_IMG_PATH}\n")
            f.write(f"Nail cx,cy,w,h = {nx},{ny},{nw},{nh}\n")
            f.write(f"Tip  cx,cy,w,h = {tx},{ty},{tw},{th}\n")
            f.write(f"Nail rect = {nrect}\n")
            f.write(f"Tip  rect  = {trect}\n")
            f.write(f"OUT_H={OUT_H}, OUT_W_LEFT={OUT_W_LEFT}, OUT_W_RIGHT={OUT_W_RIGHT}, DO_CONCAT={DO_CONCAT}\n")

        cv2.setTrackbarPos("SAVE(0/1)", WIN, 0)
        print(f"[saved] {img_path} / {txt_path}")

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
