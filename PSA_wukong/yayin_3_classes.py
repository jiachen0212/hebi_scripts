# coding=utf-8
import os
import random

'''
train: no1=no2=yes

test: no1=no2, no1+no2=yes
'''

train_txt = open('./train.txt', 'w')
test_txt = open('./test.txt', 'w')

yayin_data = '/data/home/jiachen/data/seg_data/hebi/yayin/imgs'
no1 = open('/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/classify_cls2/no1_no2_yes3/no1_yayin.txt', 'r').readlines()
no2 = open('/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/classify_cls2/no1_no2_yes3/no2_yayin.txt', 'r').readlines()
yes = open('/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/classify_cls2/no1_no2_yes3/yes_yayin.txt', 'r').readlines()

no1 = [a[:-1] for a in no1 if len(a) > 0]
no2 = [a[:-1] for a in no2 if len(a) > 0]
yes = [a[:-1] for a in yes if len(a) > 0]

random.shuffle(no1)
random.shuffle(no2)
random.shuffle(yes)

tr_len = int(len(yes)*0.7)
yes_train = ['{}:[0,1]\n'.format(os.path.join(yayin_data, im)) for im in yes[:tr_len]]
no1_train = ['{}:[1,0]\n'.format(os.path.join(yayin_data, im))for im in no1[:tr_len]]
no2_train = ['{}:[1,0]\n'.format(os.path.join(yayin_data, im))for im in no2[:tr_len]]
train_ = yes_train + no1_train + no2_train
random.shuffle(train_)
for line in train_:
    train_txt.write(line)

te_len = len(yes) - tr_len
no_test_len = te_len // 2
yes_test = ['{}:[0,1]\n'.format(os.path.join(yayin_data, im)) for im in yes[tr_len:]]
no1_test = ['{}:[1,0]\n'.format(os.path.join(yayin_data, im))for im in no1[tr_len: tr_len+no_test_len]]
no2_test = ['{}:[1,0]\n'.format(os.path.join(yayin_data, im))for im in no2[tr_len: tr_len+no_test_len]]
test_ = yes_test + no1_test + no2_test
random.shuffle(test_)
for line in test_:
    test_txt.write(line)





