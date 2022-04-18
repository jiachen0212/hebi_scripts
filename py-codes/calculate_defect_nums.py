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


# dir0312 = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1/debug_2bins'
# full_0312_dir = [os.path.join(dir0312, sub) for sub in ['aokeng', 'tudian', 'yayin']]

# # dir_byyy = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/byyy/3.17/roi_2bins'

# dir_0317kd = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0317kd/roi_2bins'

# dir_0318 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0318/roi_2bins'

# dir_newQ = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins'
# full_dir_newQ = [os.path.join(dir_newQ, sub) for sub in ['bianxing', 'qipao', 'keli']]

defect_nums = {"aodian": 0, "tudian": 0, "yayin": 0, "P-bianxing": 0, "P-qipao": 0, "P-keli": 0, "Pet_posun": 0}

# data_list = []
# data_list.extend(full_0312_dir)
# # data_list.append(dir_byyy)
# data_list.append(dir_0317kd)
# data_list.append(dir_0318)
# data_list.extend(full_dir_newQ)

dir1 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/0328/roi_2bins'
dir2 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins/keli'
dir3 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins/bianxing'
dir4 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins/qipao'
dir6 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0328/roi_2bins'
dir7 = '/data/home/jiachen/data/seg_data/hebi/shimo/toumingmo/0328/roi_2bins'
dir5 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0401keli/roi_2bins'
dir8 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/posun/roi_2bins'
dir9 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0402/roi_2bins'
dir10 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0403/roi_2bins'
dir11 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0408_posun/roi_2bins'
dir12 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/gw3_posun/roate/roi_2bins'
data_list = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8, dir9, dir10, dir11, dir12]

for dir_ in data_list:
    js_paths = [os.path.join(dir_, a) for a in os.listdir(dir_) if '.json' in a]
    for js_path in js_paths:
        for defect in list(defect_nums.keys()):
            defect_nums[defect] += json_label_check(js_path, [defect])
print(defect_nums)
