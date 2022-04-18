# coding=utf-8
import os
import json
import shutil
from roate_im_js import roate_im_jss

def json_label_check(js_path, label_list):
    try:
        data = json.load(open(js_path, 'r'))
    except:
        return 0

    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            if cls_['label'] in label_list:
                return 1
        return 0
    else:
        return 0



# txt_path = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/0410/0410_all_train.txt'
# posun_dir = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/gw3_posun'
# lines = open(txt_path).readlines()
# lines = [a for a in lines if len(a) > 1]
# for line in lines:
#     im_path, js_path = line[:-1].split(':')[0], line[:-1].split(':')[1]
#     baseneme = os.path.basename(im_path)
#     if json_label_check(js_path, ['Pet_posun']):
#         shutil.copyfile(im_path, os.path.join(posun_dir, 'JPEG', baseneme))
#         shutil.copyfile(js_path, os.path.join(posun_dir, 'JSON', baseneme.split('.')[0]+'.json'))

# 19对json和img
dirpath = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/gw3_posun'
savepath = os.path.join(dirpath, 'roate')
roate_im_jss(dirpath, savepath)
