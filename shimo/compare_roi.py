# coding=utf-8
import os
import cv2
from PIL import Image
import numpy as np

base_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi_反面/gw1/yayin'
roi_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi_反面/gw1/yayin1'
if not os.path.exists(roi_dir):
    os.makedirs(roi_dir)
imgs = [os.path.join(base_dir, a) for a in os.listdir(base_dir) if a.split('.')[-1] in ['bmp', 'png']]

roi = [0, 918, 7594, 7594]

for path_image in imgs:
    img = Image.open(path_image)
    tmp = np.asarray(img)
    h, w = tmp.shape[0], tmp.shape[1]
    tmp_img = tmp[roi[0]:roi[2], roi[1]:roi[3]]
    # 8192 8192
    # 6676 7474

    total_img = np.zeros((h, w*2), np.uint8)
    total_img[:, :h] = tmp
    total_img[roi[0]:roi[2], w+roi[1]:w+roi[3]] = tmp_img

    cv2.imwrite(os.path.join(roi_dir, os.path.basename(path_image).split('.')[0]+'.jpg'), total_img)
























# # coding=utf-8
# import os
# import cv2
# from PIL import Image
# import numpy as np



# base_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project'
# alls = open(os.path.join(base_dir, 'train.txt'), 'r').readlines()
# tests = open(os.path.join(base_dir, 'test.txt'), 'r').readlines()
# alls += tests
# roi_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1_roi'
# if not os.path.exists(roi_dir):
#     os.makedirs(roi_dir)

# imgs = ['/data{}'.format(a.split('||')[0]) for a in alls if len(a) > 1]

# # imgs = ['/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi_反面/model_gw1/test.png']

# roi = [0, 918, 7594, 7594]

# for path_image in imgs:
#     img = Image.open(path_image)
#     tmp = np.asarray(img)
#     h, w = tmp.shape[0], tmp.shape[1]
#     print(h, w)
#     tmp_img = tmp[roi[0]:roi[2], roi[1]:roi[3]]
#     # 8192 8192
#     # 6676 7474

#     total_img = np.zeros((h, w*2), np.uint8)
#     total_img[:, :h] = tmp
#     total_img[roi[0]:roi[2], w+roi[1]:w+roi[3]] = tmp_img

#     cv2.imwrite(os.path.join(roi_dir, os.path.basename(path_image).split('.')[0]+'.jpg'), total_img)