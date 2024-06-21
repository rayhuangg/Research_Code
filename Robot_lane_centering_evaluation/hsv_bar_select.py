import cv2
import numpy as np

def nothing(x):
    pass

# 讀取圖片
img = cv2.imread('IMG_7571.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 創建一個名為 'HSV' 的視窗
cv2.namedWindow('HSV')

# 創建滑動條
cv2.createTrackbar('HLower','HSV',85,180,nothing)
cv2.createTrackbar('SLower','HSV',0,255,nothing)
cv2.createTrackbar('VLower','HSV',109,255,nothing)
cv2.createTrackbar('HUpper','HSV',180,180,nothing)
cv2.createTrackbar('SUpper','HSV',190,255,nothing)
cv2.createTrackbar('VUpper','HSV',255,255,nothing)

while(1):
    # 獲取滑動條的值
    hLower = cv2.getTrackbarPos('HLower','HSV')
    sLower = cv2.getTrackbarPos('SLower','HSV')
    vLower = cv2.getTrackbarPos('VLower','HSV')
    hUpper = cv2.getTrackbarPos('HUpper','HSV')
    sUpper = cv2.getTrackbarPos('SUpper','HSV')
    vUpper = cv2.getTrackbarPos('VUpper','HSV')

    # 定義HSV的範圍
    lower_hsv = np.array([hLower, sLower, vLower])
    upper_hsv = np.array([hUpper, sUpper, vUpper])

    # 創建遮罩
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 使用遮罩來提取顏色
    res = cv2.bitwise_and(img, img, mask=mask)

    # 顯示圖片
    cv2.imshow('HSV', res)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()