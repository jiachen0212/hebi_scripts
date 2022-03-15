# coding=utf-8
import cv2
import numpy as np

img_path = r'C:\Users\15974\Desktop\20220302\1\20220302_144753417_NG.png'
img = cv2.imread(img_path)
h, w = img.shape[0], img.shape[1]
mask_img = np.zeros((h, w, 3), dtype=int)

#  x1, y1, x2, y2, x3, y3, x4, y4
points = [4854.76, 1175.26, 4884.9, 3194.85, 553.727, 3252.7, 531.197, 1233.78]
ps = [int(points[ind]) for ind in [7,5,6,0]]

point_size = 1
point_color = (0, 0, 255) # BGR
thickness = 4 # 可以为 0 、4、8
mask_img[ps[0]:ps[1]:, ps[2]:ps[3], :] += img[ps[0]:ps[1]:, ps[2]:ps[3], :]
cv2.imwrite('./2.jpg', mask_img)

