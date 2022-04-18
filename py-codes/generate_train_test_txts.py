# coding=utf-8
import os
import json
import cv2
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



train = open('./train.txt', 'w')
test = open('./test.txt', 'w')

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
        train.write(tr)
    for te in defect_lines[tr_lens:]:
        test.write(te)

with open('./info.json', 'w') as f:
    json.dump(info, f, indent=4)



