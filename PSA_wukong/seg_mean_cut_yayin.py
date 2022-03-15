# coding=utf-8
import os
import json
import random

js_path = '/data/home/jiachen/data/seg_data/hebi/seg_yayin_json'
im_path = '/data/home/jiachen/data/seg_data/hebi/seg_yayin'

jss = os.listdir(js_path)
random.shuffle(jss)

trains = int(len(jss)*0.7)
test_txt = open('./yayin_seg_test.txt', 'w')
train_txt = open('./yayin_seg_train.txt', 'w')
for js in jss[:trains]:
    line = '{}||{}\n'.format(os.path.join(im_path, js[:-4]+'png'), os.path.join(js_path, js))
    train_txt.write(line)

for js in jss[trains:]:
    line = '{}||{}\n'.format(os.path.join(im_path, js[:-4]+'png'), os.path.join(js_path, js))
    test_txt.write(line)

