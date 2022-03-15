# coding=utf-8
import os
import cv2

new_path = '/Users/chenjia/Desktop/yayin_evaluate1'
dir_ = '/Users/chenjia/Desktop/yayin_evaluate'
ims = [os.path.join(dir_, a) for a in os.listdir(dir_)]
for im in ims:
    img = cv2.imread(im)
    cv2.imwrite(os.path.join(new_path, os.path.basename(im).split('.')[0]+'.jpg'), img)