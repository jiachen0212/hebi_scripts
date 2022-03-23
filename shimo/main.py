import os
import shutil
from debug import help
import random



# base_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1'
base_dir = '/data/home/jiachen/data/seg_data/hebi/shimo/Q1'
def_list = ['aokeng', 'tudian', 'yayin']

debug_2bins = '/data/home/jiachen/data/seg_data/hebi/shimo/Q1/roi_2bins'

if not os.path.exists(debug_2bins):
    os.makedirs(debug_2bins)

train_all = []
test_all = []
# roi = (918,0,7594, 7594)  # 横,纵 起终点.
roi = (800,0,7594, 7269)

split_target = (1, 2)
for defect in def_list:
    tmp = []
    out_dir = os.path.join(debug_2bins, defect)
    path1 = os.path.join(base_dir, defect)
    help(path1, roi, split_target, out_dir)

    ims = [a for a in os.listdir(out_dir) if os.path.splitext(a)[-1] in ['.bmp' , '.png']]
    for im in ims:
        pre, ext = os.path.splitext(im)
        js_path = os.path.join(out_dir, pre+'.json')
        if os.path.exists(js_path):
            line = '{}||{}\n'.format(os.path.join(out_dir, im)[5:], js_path[5:])
            print(line)
            tmp.append(line)
    random.shuffle(tmp)
    train_all.extend(tmp[:int(len(tmp)*0.7)])
    test_all.extend(tmp[int(len(tmp)*0.7):])

# 写入txt
tr_txt = open('./Q_train.txt', 'w')
te_txt = open('./Q_test.txt', 'w')

for tr in train_all:
    tr_txt.write(tr)

for te in test_all:
    te_txt.write(te)




