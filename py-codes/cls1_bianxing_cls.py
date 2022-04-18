# coding=utf-8
'''
for cls1: bianxing cls pre_data_process

1. 需要在集群上把所有的bianxing数据都copy出来,
2. 然后 3x6切割.
3. 最后接 junguang的 yes, no 图像名.

'''


import os
from shutil import copyfile
import json
# 导入切图脚本
from crop_bianxing import fun
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


def copy_bianxing(js_dir, data_dir, clses, bianxing_dir, flag=None):
    for cls_ in clses:
        path1 = os.path.join(data_dir, cls_)
        for sub_defect in os.listdir(path1):
            path2 = os.path.join(path1, sub_defect)
            for dir_ in os.listdir(path2):
                img_dir = os.path.join(path2, dir_)
                ims = os.listdir(img_dir)
                js_dir = '{}/jsons{}'.format(img_dir.split(flag)[0], img_dir.split(flag)[1])
                im_paths = [os.path.join(img_dir, a) for a in ims]
                js_paths = [os.path.join(js_dir, '{}.json'.format(a[:-4])) for a in ims]
                for ind, js_path in enumerate(js_paths):
                    if json_label_check(js_path, ['bianxing']):
                        if ('/3/' in js_path) or ('/4/' in js_path):
                            print(js_path)
                        copyfile(im_paths[ind], os.path.join(bianxing_dir, os.path.basename(im_paths[ind])))


def split_train_test(js_dir, out_dir, yes_file, no_file, train_t, test_t):

    train_txt = open(train_t, 'w')
    test_txt = open(test_t, 'w')

    yeses = open(os.path.join(js_dir, yes_file), 'r').readlines()
    nos = open(os.path.join(js_dir, no_file), 'r').readlines()
    yeses = [a[:-1] for a in yeses if len(a) > 0]
    nos = [a[:-1] for a in nos if len(a) > 0]
    random.shuffle(yeses)
    random.shuffle(nos)

    train_yes = int(len(yeses)*0.7)
    train_no = int(len(nos)*0.7)
    trains, tests = [], []
    for d in yeses[:train_yes]:
        img_path = os.path.join(out_dir, d)
        if os.path.exists(img_path):
            line = '{}:[0,1]\n'.format(img_path)
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


def merge_data(txts, train_, test_):
    for ind, txt in enumerate(txts):
        lines = open(txt, 'r').readlines()
        lines = [a for a in lines if len(a) > 0]
        if ind in [0, 2]:
            for line in lines:
                train_.write(line[5:])
        else:
            for line in lines:
                test_.write(line[5:])


if __name__ == "__main__":

    #1. copy 出变形图像
    data = '0223'

    js_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/jsons/PSA'.format(data)
    data_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs/PSA'.format(data)
    bianxing_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs/{}bianxing'.format(data, data)
    out_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs/{}bianxing_split'.format(data, data)

    if not os.path.exists(bianxing_dir):
        os.makedirs(bianxing_dir)
    clses = ['1cls', '2cls', '3cls']
    # copy_bianxing(js_dir, data_dir, clses, bianxing_dir, flag='/imgs')

    #2. 切割图像
    # roi = (0, 200, 5472, 3648)
    # split_target = (6, 3)
    # fun(bianxing_dir, out_dir, roi, split_target)

    # del bianxing_dir
    ims = os.listdir(bianxing_dir)
    for im in ims:
        path = os.path.join(bianxing_dir, im)
        os.remove(path)
    os.rmdir(bianxing_dir)

    # 3.导入yes_bianxing, no_bianxing标注. 分别在yes, no 下做train/test拆分, 写入train.txt, test.txt

    for data in ['0214', '0223']:
        # 生成train, test数据
        js_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/jsons/PSA'.format(data)
        out_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs/{}bianxing_split'.format(data, data)
        yes_file = '{}_yes_bianxing.txt'.format(data)
        no_file = '{}_no_bianxing.txt'.format(data)
        train_txt = './{}cls1_bianxing_train.txt'.format(data)
        test_txt = './{}cls1_bianxing_test.txt'.format(data)
        split_train_test(js_dir, out_dir, yes_file, no_file, train_txt, test_txt)


    # merge 0214 and 0223 data
    all_train = open('./bx_train.txt', 'w')
    all_test = open('./bx_test.txt', 'w')
    txts = []
    for data in ['0214', '0223']:
        txts.append('./{}cls1_bianxing_train.txt'.format(data))
        txts.append('./{}cls1_bianxing_test.txt'.format(data))
    merge_data(txts, all_train, all_test)





