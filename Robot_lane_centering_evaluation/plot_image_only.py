from matplotlib import pyplot as plt
import cv2

image_path = 'Data/exp_lab2-0002.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()