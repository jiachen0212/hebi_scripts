# coding=utf-8

import os
import shutil
from debug import help
import random
import cv2

def roi_imgs(out_dir, base_dir, roi):
    ims = [os.path.join(base_dir, a) for a in os.listdir(base_dir) if a.split('.')[-1] in ['png', 'bmp']]
    for im in ims:
        basename = os.path.basename(im)
        img = cv2.imread(im)

        cuted_img = img[roi[1]: roi[3], roi[0]: roi[2], :]
        # cv2.imshow('2', cuted_img)
        # cv2.waitKey(2000)
        h, w = cuted_img.shape[0], cuted_img.shape[1]

        cuted_img_1 = cuted_img[:h//2, :, :]
        cuted_img_2 = cuted_img[h//2:, :, :]
        cv2.imwrite(os.path.join(out_dir, basename.split('.')[0]+'_0.jpg'), cuted_img_1)
        cv2.imwrite(os.path.join(out_dir, basename.split('.')[0]+'_1.jpg'), cuted_img_2)

flag = 1

def_list = ['keli', 'bianxing', 'qipao']
tr_file = './keliqipaobianxing_train.txt'
te_file = './keliqipaobianxing_test.txt'
train_all = []
test_all = []

# roi 横,纵 起终点.
# roi = (670,0,7594, 7269)
# keli qipao bianxing
roi = (1000, 0, 7410, 7580) 


# only test roi
if flag == 0:
    base_dir = '/Users/chenjia/Desktop/qipao'
    out_dir = '/Users/chenjia/Desktop/qipao/test_roi'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    roi_imgs(out_dir, base_dir, roi)


if flag == 1:
    base_dir = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ'
    debug_2bins = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins'
    if not os.path.exists(debug_2bins):
        os.makedirs(debug_2bins)
    split_target = (1, 2)
    for defect in def_list:
        tmp = []
        out_dir = os.path.join(debug_2bins, defect)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        path1 = os.path.join(base_dir, defect)
        help(path1, roi, split_target, out_dir)

        ims = [a for a in os.listdir(out_dir) if os.path.splitext(a)[-1] in ['.bmp' , '.png']]
        for im in ims:
            pre, ext = os.path.splitext(im)
            js_path = os.path.join(out_dir, pre+'.json')
            if os.path.exists(js_path):
                line = '{}||{}\n'.format(os.path.join(out_dir, im)[5:], js_path[5:])
                print(line)
                tmp.append(line)
        random.shuffle(tmp)
        train_all.extend(tmp[:int(len(tmp)*0.7)])
        test_all.extend(tmp[int(len(tmp)*0.7):])

    # 写入txt
    tr_txt = open(tr_file, 'w')
    te_txt = open(te_file, 'w')
    for tr in train_all:
        tr_txt.write(tr)

    for te in test_all:
        te_txt.write(te)


# train.txt和之前的train txt合并
if flag == 6:
    all_file = open('./all.txt', 'w')

    pre_data_file = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/codes_for_cut_2bins/roi_debug/all_data_all_Q.txt'
    pre_data = open(pre_data_file, 'r').readlines()
    pre_data = [a for a in pre_data if len(a) > 1]

    new_train = open(tr_file, 'r').readlines()
    new_train = [a for a in new_train if len(a) > 1]
    new_test = open(te_file, 'r').readlines()
    new_test = [a for a in new_test if len(a) > 1]

    alls = pre_data + new_train + new_test

    for line in alls:
        all_file.write(line)



