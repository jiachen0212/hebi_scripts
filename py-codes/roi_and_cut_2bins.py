# coding=-utf-8
import cv2
import os
import json
from shutil import copyfile
from cut_js_img_help import help
import random



def till_get_im_and_js(dir_path, pre_files):
    # 遍历所有文件夹套文件夹下的所有files
    files = [os.path.join(dir_path, a) for a in os.listdir(dir_path)]
    for file in files:
        if os.path.isdir(file):
            # 进入某文件夹
            # os.chdir(file_path)
            till_get_im_and_js(file, pre_files)
        else:
            pre_files.append(file)


train, test = [], []

data_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1'
info = dict()
info["data_dir"] = data_dir
info["categories"] = dict()


defects = ['aokeng', 'tudian', 'yayin']
defect_list = [[] for i in range(3)]
lines = [[] for i in range(3)]

for ind, defect in enumerate(defects):
    defeat_dir = os.path.join(data_dir, defect)
    till_get_im_and_js(defeat_dir, defect_list[ind])

for ind, defect_js_im in enumerate(defect_list):
    defect_ims = [a for a in defect_js_im if os.path.splitext(a)[1] in ['.bmp', '.png']]
    for im in defect_ims:
        js_path = os.path.join(os.path.dirname(im), os.path.splitext(os.path.basename(im))[0]+'.json')
        if os.path.exists(js_path):
            line = "{}||{}\n".format(im[5:], js_path[5:])
            lines[ind].append(line)


for ind, defect_lines in enumerate(lines):
    random.shuffle(defect_lines)
    info["categories"][defects[ind]] = len(defect_lines)
    tr_lens = int(len(defect_lines)*0.7)
    for tr in defect_lines[:tr_lens]:
        train.append(tr)
    for te in defect_lines[tr_lens:]:
        test.append(te)

with open('./info.json', 'w') as f:
    json.dump(info, f, indent=4)



'''
分别对train_list, test_list做2bins切割.

'''


def get_train_test_txts(train_2bins, train_txt):
    trs = os.listdir(train_2bins)
    trs_ims = [a for a in trs if a.split('.')[-1] in ['png', 'bmp']]
    for im in trs_ims:
        im_path = os.path.join(train_2bins, im)
        js_path = os.path.join(train_2bins, im.split('.')[0]+'.json')
        if not os.path.exists(js_path):
            continue
        try:
            img = cv2.imread(im_path)
            h = img.shape[0]
        except:
            continue
        line = '{}||{}\n'.format(im_path[5:], js_path[5:])
        print(line)
        train_txt.write(line)



# 落盘切割后的2bins
train_2bins = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1_2bins/train'
test_2bins = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1_2bins/test'
if not os.path.exists(train_2bins):
    os.makedirs(train_2bins)
if not os.path.exists(test_2bins):
    os.makedirs(test_2bins)

split_target = (1, 2)
roi = [0, 0, 8192, 8192]  # [0, 918, 7594, 7594]

help(train, train_2bins, split_target, roi=roi)
help(test, test_2bins, split_target, roi=roi)

# 对single_wuliao: train_2bins and test_2bins 生成train.txt test.txt
tra_txt = open('./train.txt', 'w')
tes_txt = open('./test.txt', 'w')
get_train_test_txts(train_2bins, tra_txt)
get_train_test_txts(test_2bins, tes_txt)




