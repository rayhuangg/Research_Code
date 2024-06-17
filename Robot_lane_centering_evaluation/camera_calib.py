import cv2
import numpy as np
import glob
import os

# 設定棋盤格尺寸
chessboard_size = (6, 4)

# 準備棋盤格點
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 儲存所有影像的棋盤格點（世界座標系下的3D點）和影像上的棋盤格角點（影像座標系下的2D點）
objpoints = []  # 3D點
imgpoints = []  # 2D點

# 讀取所有校正影像
input_dir = 'calibration_images/exp_lab2'
images = glob.glob(os.path.join(input_dir, '*.jpg'))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找到棋盤格角點
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到，則添加點
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 繪製並顯示角點
        img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
# 校正相機
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 顯示相機校正結果
print("Camera matrix:")
print(camera_matrix)
print("Distortion coefficients:")
print(dist_coeffs)
# print("Rotation vectors:")
# print(rvecs)
# print("Translation vectors:")
# print(tvecs)

# 儲存校正結果
np.savez(os.path.join(input_dir, 'camera_calibration.npz'), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print(f"Camera calibration complete. Calibration results saved to {os.path.join(input_dir, 'camera_calibration.npz')}")
