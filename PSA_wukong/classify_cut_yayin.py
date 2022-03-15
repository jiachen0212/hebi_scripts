# coding=utf-8
import cv2
import os
import json
from crop_bianxing import fun
from shutil import copyfile
import math
import numpy as np
from PIL import Image
import random


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
        try:
            h, w = img.shape[0], img.shape[1]
        except:
            continue

        img = img[roi[0]:h - roi[1], roi[2]: w - roi[3], :]   # img = img[:h-600, 300: w-200, :]
        h, w = img.shape[0], img.shape[1]
        # cv2.imshow('1', img)
        # cv2.waitKey(1000)

        bins = [math.ceil((h-bin_size)/stride_h)+1, math.ceil((w-bin_size)/stride_w)+1]
        padding_w, padding_h = (bins[1]-1)*stride_w+bin_size-w, (bins[0]-1)*stride_h+bin_size-h
        print('cut bins: {}, padding_w_h: {}'.format(bins, [padding_w, padding_h]))
        H, W = padding_w+w, padding_h+h
        new_img = np.zeros((W, H, 3), dtype=np.uint8)
        new_img[:h, :w, :] += img

        x = 0
        for i in range(bins[0]):
            y = 0
            for j in range(bins[1]):
                bin_img = new_img[x:x+bin_size, y:y+bin_size, :]
                y += stride_w
                cv2.imwrite(os.path.join(sub_bins, '{}_{}_{}.png'.format(basename[:-4], i, j)), bin_img)
            x += stride_h



def split_train_test(js_dir, out_dir, yes_file, no_file, train_t, test_t):

    train_txt = open(train_t, 'w')
    test_txt = open(test_t, 'w')

    yeses = open(os.path.join(js_dir, yes_file), 'r').readlines()
    nos = open(os.path.join(js_dir, no_file), 'r').readlines()
    yeses = [a[:-1] for a in yeses if len(a) > 0]
    nos = [a[:-1] for a in nos if len(a) > 0]
    print("yes_yayin: {}, no_yayin: {}".format(len(yeses), len(nos)))
    random.shuffle(yeses)
    random.shuffle(nos)

    train_yes = int(len(yeses)*0.7)
    train_no = int(len(nos)*0.7)
    trains, tests = [], []
    for d in yeses[:train_yes]:
        img_path = os.path.join(out_dir, d)
        # print(img_path)
        if os.path.exists(img_path):
            line = '{}:[0,1]\n'.format(img_path)
            print(line)
            train_txt.write(line)
    for d in nos[:train_no]:
        img_path = os.path.join(out_dir, d)
        if os.path.exists(img_path):
            line = '{}:[1,0]\n'.format(img_path)
            train_txt.write(line)

    for d in yeses[train_yes:]:
        img_path = os.path.join(out_dir, d)
        if os.path.exists(img_path):
            line = '{}:[0,1]\n'.format(img_path)
            test_txt.write(line)

    for d in nos[train_no:]:
        img_path = os.path.join(out_dir, d)
        if os.path.exists(img_path):
            line = '{}:[1,0]\n'.format(img_path)
            test_txt.write(line)


if __name__ == "__main__":

    # 1.左右物料切割
    data_dir = '/data/home/jiachen/data/seg_data/hebi/mask_yayin'
    yayindir_cut = '/data/home/jiachen/data/seg_data/hebi/mask_yayin_2bins'
    roi = (0, 0, 8192, 8192)
    split_target = (2, 1)
    fun(data_dir, yayindir_cut, roi, split_target)

    # 左右切分物料.
    #1. 切0214的压印数据.
    # datas = ['0223']
    # for data in datas:
    #     yayindir = '/data/home/jiachen/data/seg_data/hebi/full_yayin'
    #     data_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs/PSA'.format(data)
        # generate_yayin_paths(data_dir, yayindir)
    #     roi = (0, 0, 8192, 8192)
    #     split_target = (2, 1)
    #     fun(yayindir, yayindir_cut, roi, split_target)


    #2. 单独切指定文件夹的压印数据.
    # data_dir = '/data/home/jiachen/data/seg_data/hebi/20220223/imgs/PSA/2cls/yayin/3'
    # roi = (0, 0, 8192, 8192)
    # split_target = (2, 1)
    # fun(data_dir, yayindir_cut, roi, split_target)


    # 针对每一个小物料, overlap切割成小块
    # sub_bins_dir = '/data/home/jiachen/data/seg_data/hebi/yayin/imgs_3072'
    # if not os.path.exists(sub_bins_dir):
    #     os.makedirs(sub_bins_dir)

    # 2. 滑窗切割分类的2048子图
    stride_h, stride_w, bin_size = 1400, 600, 2048
    rois = [0, 600, 300, 200]
    sub_bins_dir = '/data/home/jiachen/data/seg_data/hebi/mask_2048_yayin'
    overlap_cut_img(yayindir_cut, stride_h, stride_w, bin_size, sub_bins_dir, rois)

    # 3072尺寸切割
    # stride_h, stride_w, bin_size = 1800, 1000, 3072
    # overlap_cut_img(yayindir_cut, stride_h, stride_w, bin_size, sub_bins_dir, rois)


    # yes_yayin, no_yayin的标注回来了, 生成train.txt test.txt
    # data = '0302'
    # js_dir = '/data/home/jiachen/data/seg_data/hebi/yayin/txts'
    # yes_file = '{}_yes_yayin.txt'.format(data)
    # no_file = '{}_no_yayin.txt'.format(data)
    # train_txt = './{}_yayin_train.txt'.format(data)
    # test_txt = './{}_yayin_test.txt'.format(data)
    # split_train_test(js_dir, sub_bins_dir, yes_file, no_file, train_txt, test_txt)













