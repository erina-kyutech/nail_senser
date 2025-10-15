import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"カメラ {i} は使用可能です")
        cap.release()
    else:
        print(f"カメラ {i} は使えません")
