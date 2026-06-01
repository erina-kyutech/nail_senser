import cv2
# カメラのインデックス番号を確認する
for index in range(5):  # 0から4までのインデックスを試す
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"カメラが見つかりました: インデックス {index}")
        cap.release()
    else:
        print(f"カメラが見つかりませんでした: インデックス {index}")