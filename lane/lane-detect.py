import cv2
#import math
#import numpy as np

Hist_height = 30
ROI_number = 80
ROI_height = 10#2  横に取る範囲を拡大した
ROI_width = 640 // ROI_number
ROI_y0 = 360 - ROI_height
ROI_y1 = 360

cv2.namedWindow('src')
cap = cv2.VideoCapture('highway.mov') # Open movie
fr = cap.get(cv2.CAP_PROP_FPS) # フレームレート

while True:
    (ret, img_src) = cap.read()	# retは画像を取得成功フラグ
    if ret:		# フレームが得られていれば表示する
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        ROI_x = 0
        edges=cv2.Canny(img_gray,50,150)
        for _ in range(ROI_number):
            ROI = img_gray[ROI_y0:ROI_y1, ROI_x:ROI_x + ROI_width]
            bright = ROI.mean()
            h = int(bright * Hist_height / 255)
            cv2.rectangle(img_src,(ROI_x,ROI_y1-h-1,ROI_width,h),(100,255,100),-1)
            cv2.rectangle(img_src,(ROI_x,ROI_y1-h-1,ROI_width,h),(0,0,0),)
            ROI_x += ROI_width
            cv2.imshow('src',img_src)
        cv2.imshow('src', img_src)
        k = cv2.waitKey(30)	# キー入力を30msec待つ
        if k == 27:			# ESCキーで終了
            break
    else:	# フレームが得られていなければ終了する
        break

cap.release()
cv2.destroyAllWindows()

