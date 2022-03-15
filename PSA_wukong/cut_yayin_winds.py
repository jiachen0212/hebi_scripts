# coding=utf-8
import cv2
import os
import json
from crop_bianxing import fun
from shutil import copyfile
import math
import numpy as np
from PIL import Image


def json_label_check(js_path, label_list):

    try:
        data = json.load(open(js_path, 'r'))
    except:
        return False

    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            if cls_['label'] in label_list:
                return True
        return False
    else:
        return False



def generate_yayin_paths(data_dir, yayindir):
    yayins = []
    cls123 = os.listdir(data_dir)
    for cls_index, cls_cur in enumerate(cls123):
        cls_cur_path = os.path.join(data_dir, cls_cur)
        sub_defects = os.listdir(cls_cur_path)
        for sub_defect in sub_defects:
            path1 = os.path.join(cls_cur_path, sub_defect)
            gong_weis = os.listdir(path1)
            for gw in gong_weis:
                if gw == '3':
                    # img_path: /data/home/jiachen/data/seg_data/hebi/20220214/imgs/PSA/1cls/yashang/3
                    img_path = os.path.join(path1, gw)
                    json_path = '{}/jsons{}'.format(img_path.split('/imgs')[0], img_path.split('/imgs')[1])
                    imgs = os.listdir(img_path)
                    full_im_paths = [os.path.join(img_path, a) for a in imgs]
                    full_js_paths = [os.path.join(json_path, a[:-4]+'.json') for a in imgs]
                    for ind, js in enumerate(full_js_paths):
                        if json_label_check(js, ['yayin', 'heixian']):
                            yayins.append(full_im_paths[ind])
                            copyfile(full_im_paths[ind], os.path.join(yayindir, os.path.basename(full_im_paths[ind])))


def overlap_cut_img(yayin_2bins_dir, stride_h, stride_w, bin_size, sub_bins, roi):

    ims = [os.path.join(yayin_2bins_dir, name) for name in os.listdir(yayin_2bins_dir)]
    for im in ims:
        img = cv2.imread(im)
        basename = os.path.basename(im)

        # 剔除左右边的黑色冗余
        h, w = img.shape[0], img.shape[1]
        img = img[roi[0]:h - roi[1], roi[2]: w - roi[3], :]   # img = img[:h-600, 300: w-200, :]
        h, w = img.shape[0], img.shape[1]
        # cv2.imwrite(r'C:\Users\15974\Desktop\1\2.png', img)

        # 设置滑窗stride
        # stride_h, stride_w = 1400, 600
        # bin_size = 2048

        bins = [math.ceil((h-bin_size)/stride_h)+1, math.ceil((w-bin_size)/stride_w)+1]
        padding_w, padding_h = (bins[1]-1)*stride_w+bin_size-w, (bins[0]-1)*stride_h+bin_size-h
        print('cut bins: {}, padding_w_h: {}'.format(bins, [padding_w, padding_h]))
        H, W = padding_w+w, padding_h+h  # 3848 9248
        new_img = np.zeros((W, H, 3), dtype=int)
        new_img[:h, :w, :] += img

        x = 0
        for i in range(bins[0]):
            y = 0
            for j in range(bins[1]):
                bin_img = new_img[x:x+bin_size, y:y+bin_size, :]
                y += stride_w
                cv2.imwrite(os.path.join(sub_bins, '{}_{}_{}.png'.format(basename[:-4], i, j)), bin_img)
            x += stride_h


if __name__ == "__main__":

    # yayindir = '/data/home/jiachen/data/seg_data/hebi/14_23_yayin'
    # yayindir_cut = '/data/home/jiachen/data/seg_data/hebi/14_23_yayin_2bins'

    # 左右切分物料.
    #1. 切0214的压印数据.
    datas = ['0214']
    for data in datas:
        # js_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/jsons/PSA'.format(data)
        data_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs/PSA'.format(data)
        # generate_yayin_paths(data_dir, yayindir)
        roi = (0, 0, 8192, 8192)
        split_target = (2, 1)
        # fun(yayindir, yayindir_cut, roi, split_target)


    #2. 单独切0223的压印数据.
    data_dir = '/data/home/jiachen/data/seg_data/hebi/20220223/imgs/PSA/2cls/yayin/3'
    roi = (0, 0, 8192, 8192)
    split_target = (2, 1)
    # fun(data_dir, yayindir_cut, roi, split_target)


    # 针对每一个小物料, overlap切割小块
    yayin_2bins_dir = r'C:\Users\15974\Desktop\14_23_yayin\14_23_yayin_2bins'
    sub_bins_dir = r'C:\Users\15974\Desktop\0301_0214_0223_0226_subs_yayin'

    stride_h, stride_w, bin_size = 1800, 1000, 3072
    rois = [800, 600, 500, 200]
    overlap_cut_img(yayin_2bins_dir, stride_h, stride_w, bin_size, sub_bins_dir, rois)










