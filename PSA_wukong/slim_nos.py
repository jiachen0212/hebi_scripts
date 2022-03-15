#conding=utf-8
import os
import random

'''
统计train, test 中的yes, no. 手动"zuobi"分配train, test 中的类别数据占比
'''


def get_lab(lab):
    lab = lab.strip('[,]')
    if lab[0] == '1':
        return 0
    else:
        return 1


def slim_train_test_nos(base_dir, bx_train, bx_test, no2yes, train_, test_):

    train_txt = open(train_, 'w')
    test_txt = open(test_, 'w')

    trains = open(os.path.join(base_dir, bx_train)).readlines()
    trains = [a[:-1] for a in trains if len(a) > 0]

    tests = open(os.path.join(base_dir, bx_test)).readlines()
    tests = [a[:-1] for a in tests if len(a) > 0]

    for ind, txt in enumerate([trains, tests]):
        ims_yes, ims_nos = [], []
        for line in txt:
            path_, lab = line.split(':')[0], get_lab(line.split(':')[1])
            if lab == 1:
                ims_yes.append(path_)
            else:
                ims_nos.append(path_)
        print("yes: {}, pre_no: {}".format(len(ims_yes), len(ims_nos)))
        slimed_no_lens = min(len(ims_yes)*no2yes[ind], len(ims_nos))
        random.shuffle(ims_nos)
        slim_no_ims = ims_nos[:slimed_no_lens]
        print("yes: {}, new_no: {}".format(len(ims_yes), len(slim_no_ims)))
        trains = [a+':[0,1]\n' for a in ims_yes] + [a+':[1,0]\n' for a in slim_no_ims]
        random.shuffle(trains)
        if ind == 0:
            for line in trains:
                train_txt.write(line)
        else:
            for line in trains:
                test_txt.write(line)



if __name__ == "__main__":

    # bianxing
    base_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/project'
    bx_train = 'bx_train.txt'
    bx_test = 'bx_test.txt'
    txt_out_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/cls1_bx/slim_no'

    # yayin
    # bx_train = '0226_yayin_train.txt'
    # bx_test = '0226_yayin_test.txt'
    # txt_out_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/classify_cls2/overlap/yayin_slim'


    # 控制train, test中, no:yes的比例.
    no2yes = [2, 1]
    slimed_nos_train_txt = os.path.join(txt_out_dir, 'bx_no{}yes_train.txt'.format(no2yes[0]))
    slimed_nos_test_txt = os.path.join(txt_out_dir, 'bx_no{}yes_test.txt'.format(no2yes[1]))

    slim_train_test_nos(base_dir, bx_train, bx_test, no2yes, slimed_nos_train_txt, slimed_nos_test_txt)







