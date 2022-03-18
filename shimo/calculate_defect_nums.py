# coding=utf-8
'''
统计json中各个缺陷个数的统计
'''
import os
import json


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


dir0312 = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1/debug_2bins'
full_0312_dir = [os.path.join(dir0312, sub) for sub in ['aokeng', 'tudian', 'yayin']]

dir_byyy = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/byyy/3.17/roi_2bins'

dir_0317kd = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0317kd/roi_2bins'

defect_nums = {"aodian": 0, "tudian": 0, "yayin": 0, "bian_yayin": 0}

data_list = []
data_list.extend(full_0312_dir)
data_list.append(dir_byyy)
data_list.append(dir_0317kd)
assert len(data_list) == 3+1+1

for dir_ in data_list:
    js_paths = [os.path.join(dir_, a) for a in os.listdir(dir_) if '.json' in a]
    for js_path in js_paths:
        for defect in list(defect_nums.keys()):
            defect_nums[defect] += json_label_check(js_path, [defect])
print(defect_nums)
