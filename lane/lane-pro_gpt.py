import cv2
import numpy as np

# 動画読み込み
cap = cv2.VideoCapture('highway.mov')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # --- ROI設定：画像の下半分だけを使う ---
    roi = frame[height // 2:, :]  # 高さの半分から下すべて、横は全体

    # グレースケール + ぼかし
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Cannyエッジ検出
    edges = cv2.Canny(blur, 50, 150)

    # Hough変換で直線を検出
    #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            #minLineLength=50, maxLineGap=50)
# HoughLinesPで直線を検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

        # 傾きを計算（水平線を除外）
            if x2 - x1 == 0:  # 垂直線（無限大の傾き）
                slope = float('inf')
            else:
                slope = (y2 - y1) / (x2 - x1)

        # 傾きの絶対値が小さい場合（≒水平線）はスキップ
            if abs(slope) < 0.5:
                continue  # 横線に近いのでスキップ

        # 傾きがある線のみ描画
            cv2.line(frame, (x1, y1 + height // 2), (x2, y2 + height // 2), (0, 0, 255), 2)

    # 表示
    cv2.imshow('Lane Detection (ROI)', frame)

    if cv2.waitKey(30) == 27:  # ESCで終了
        break

cap.release()
cv2.destroyAllWindows()
