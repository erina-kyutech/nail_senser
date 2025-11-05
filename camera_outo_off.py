# win_cam_lock_exposure_wb.py
import cv2, time

def open_cam(index=0):
    # 優先: DirectShow → ダメならMSMF → 既定
    for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            try:
                name = cap.getBackendName()
            except:
                name = str(api)
            return cap, name
    raise RuntimeError("カメラを開けませんでした")

def set_manual(cap):
    # 露出: DirectShowは 0.25=手動, 0.75=自動 が有名パターン
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手動トグル試行
    # ホワイトバランス
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)          # 自動WB OFF（対応機のみ）
    time.sleep(0.2)

def read_vals(cap):
    vals = {
        "auto_exp": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exp": cap.get(cv2.CAP_PROP_EXPOSURE),            # DSHOWでは-13〜-1あたり（対数っぽい）
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
        "wb_temp": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),  # 2800〜6500Kなど
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "bright": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "contr": cap.get(cv2.CAP_PROP_CONTRAST),
        "sat": cap.get(cv2.CAP_PROP_SATURATION),
    }
    return vals

cap, backend = open_cam(0)
print("Backend:", backend)

# 設定ダイアログを一応叩いてみる（多くはFalse返却で無反応）
ok = cap.set(cv2.CAP_PROP_SETTINGS, 1)
print("CAP_PROP_SETTINGS opened?:", ok)

set_manual(cap)

# 初期値をある程度固定方向に寄せる（失敗しても無視されるだけ）
cap.set(cv2.CAP_PROP_EXPOSURE, -6)             # 例: -6（数値は環境依存）
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)     # 例: 4500K
cap.set(cv2.CAP_PROP_GAIN, 0)

cv2.namedWindow("cam", cv2.WINDOW_AUTOSIZE)  # resizeWindowは呼ばない

print("キー操作: a=自動露出トグル, z/x=露出-+, s=自動WBトグル, c/v=WB-+, q=終了")

auto_exp_manual = True
while True:
    ret, frame = cap.read()
    if not ret: break

    vals = read_vals(cap)
    txt = f"[{backend}] AE:{vals['auto_exp']:.2f} EXP:{vals['exp']:.2f}  AWB:{vals['auto_wb']:.0f} WB:{vals['wb_temp']:.0f}  GAIN:{vals['gain']:.1f}"
    cv2.putText(frame, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    cv2.imshow("cam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        # 自動露出トグル
        auto_exp_manual = not auto_exp_manual
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if auto_exp_manual else 0.75)
    elif key == ord('z'):
        cap.set(cv2.CAP_PROP_EXPOSURE, vals['exp'] - 1)
    elif key == ord('x'):
        cap.set(cv2.CAP_PROP_EXPOSURE, vals['exp'] + 1)
    elif key == ord('s'):
        # 自動WBトグル
        new = 1 if vals['auto_wb'] == 0 else 0
        cap.set(cv2.CAP_PROP_AUTO_WB, new)
    elif key == ord('c'):
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, vals['wb_temp'] - 100)
    elif key == ord('v'):
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, vals['wb_temp'] + 100)

cap.release()
cv2.destroyAllWindows()
