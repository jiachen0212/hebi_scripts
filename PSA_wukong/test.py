import cv2
import numpy as np


i = '/Users/chenjia/Desktop/1/20220209_164708435_Z__OK.png'
tmp = cv2.imdecode(np.fromfile(i, dtype=np.uint8), -1)
h, w = tmp.shape[0], tmp.shape[1]
print(w, h)
img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
cv2.imshow('1', img)
cv2.waitKey(1000)