# coding=utf-8
import os
import json

# dir_ = '/Users/chenjia/Desktop'
# txts = [os.path.join(dir_, a) for a in os.listdir(dir_) if '.txt' in a]

# lines = []
# for txt in txts:
#     print(txt)
#     line = open(txt, 'r').readlines()
#     line = [b for b in line if len(b) > 1]
#     for ll in line:
#         if ll not in lines:
#             lines.append(ll)

# all_ = open('./all_till_0411.txt', 'w')
# for l in lines:
#     all_.write(l)


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


defect_nums = {"aodian": 0, "tudian": 0, "yayin": 0, "P-bianxing": 0, "P-qipao": 0, "P-keli": 0, "Pet_posun": 0}

all_lines = './all_till_0411.txt'
line = open(all_lines, 'r').readlines()
line = [b for b in line if len(b) > 1]
print(len(line))
for l in line:
    js_path = '/data{}'.format(l[:-1].split('||')[1])
    for defect in list(defect_nums.keys()):
        defect_nums[defect] += json_label_check(js_path, [defect])
print(defect_nums)


