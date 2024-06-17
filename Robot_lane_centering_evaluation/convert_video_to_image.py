import cv2
import os

# 設定變數
video_path = "Data/exp_lab2.mp4"  # 影片檔案路徑
output_dir = "Data/imgs/exp_lab2"  # 輸出圖像的目錄
interval_seconds = 3  # 每幾秒截取一張照片

# output_dir = os.path.join(output_dir, exp_name)  # 合併輸出目錄和實驗名稱

# 確保輸出目錄存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 讀取影片
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 影片的幀率
frame_interval = int(frame_rate * interval_seconds)  # 計算幀間隔

frame_index = 0
saved_image_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % frame_interval == 0:
        output_path = os.path.join(output_dir, f"frame_{saved_image_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
        saved_image_count += 1

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

print(f"Total {saved_image_count} images saved in '{output_dir}' directory.")
