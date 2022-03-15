# coding=utf-8
import os
from shutil import copyfile
import json


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


def copy_bianxing(js_dir, data_dir, clses, bianxing_dir):
    for cls_ in clses:
        path1 = os.path.join(data_dir, cls_)
        for sub_defect in os.listdir(path1):
            path2 = os.path.join(path1, sub_defect)
            for dir_ in os.listdir(path2):
                img_dir = os.path.join(path2, dir_)
                js_dir = '{}/jsons{}'.format(img_dir.split('/data')[0], img_dir.split('/data')[1])
                im_paths = [os.path.join(img_dir, a) for a in os.listdir(img_dir)]
                js_paths = [os.path.join(js_dir, '{}.json'.format(a[:-4])) for a in os.listdir(img_dir)]
                for ind, js_path in enumerate(js_paths):
                    if json_label_check(js_path, ['bianxing']):
                        # print(os.path.basename(im_paths[ind]))
                        # print(js_path)
                        if ('/3/' in js_path) or ('/4/' in js_path):
                            print(js_path)
                        copyfile(im_paths[ind], os.path.join(bianxing_dir, os.path.basename(im_paths[ind])))


if __name__ == "__main__":

    js_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/20220223/jsons/PSA'
    data_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/20220223/data/PSA'
    bianxing_dir = '/Users/chenjia/Desktop/0223bianxing'
    if not os.path.exists(bianxing_dir):
        os.makedirs(bianxing_dir)
    clses = ['2cls'] # ['1cls', '2cls', '3cls']
    copy_bianxing(js_dir, data_dir, clses, bianxing_dir)
