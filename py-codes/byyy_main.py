import os
import shutil
from debug import help
import random
import cv2
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


def roi_imgs(out_dir, base_dir, roi):
    ims = [os.path.join(base_dir, a) for a in os.listdir(base_dir) if a.split('.')[-1] in ['png', 'bmp']]
    for im in ims:
        basename = os.path.basename(im)
        img = cv2.imread(im)

        cuted_img = img[roi[1]: roi[3], roi[0]: roi[2], :]
        # cv2.imshow('2', cuted_img)
        # cv2.waitKey(2000)
        h, w = cuted_img.shape[0], cuted_img.shape[1]

        cuted_img_1 = cuted_img[:h//2, :, :]
        cuted_img_2 = cuted_img[h//2:, :, :]
        cv2.imwrite(os.path.join(out_dir, basename.split('.')[0]+'_0.jpg'), cuted_img_1)
        cv2.imwrite(os.path.join(out_dir, basename.split('.')[0]+'_1.jpg'), cuted_img_2)



# base_dir ='/data/home/jiachen/data/seg_data/hebi/shimo/toumingmo/0328'
# base_dir = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0410yayin'   # 0407_fix' # 0405   # 0401keli'   # posun'
base_dir = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/yayin_gs_youhua'
tr_txt =  './0413.txt'
# te_txt = './0328tmm_test.txt'


flag = 2
out_dir = os.path.join(base_dir, 'roi_2bins')


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


train_all = []
test_all = []
# roi = (670,0,7594, 7269)  # 横,纵 起终点
# roi = (918,0,7594, 7594)  # 0312
# roi = (1100,100,7300, 7400) # 0328 tmm
# roi = (1100,0,7400, 7400)  # 0328night_hm
# roi = (1100, 100, 7500, 7500)  # posun 
# roi = (1100, 0, 7250, 7500)  # 0330下午
# roi = (1200, 100, 7300, 7600)  # 0401keli
# roi = (1000, 100, 7500, 7500)   # 0402早上
roi = (1100, 100, 7400, 7400)  # 0403 and 0405 and 0407night
# roi = (1200, 0, 7400, 7500)    # 0408yayin
# roi = (200, 300, 6600, 7300)  # gw3_roate
split_target = (1, 2)


# only check img_roi
if flag == 0:
    roi_imgs(out_dir, base_dir, roi)


if flag == 1:
    tmp = []
    help(base_dir, roi, split_target, out_dir)

if flag == 2:
    tmp = []
    ims = [a for a in os.listdir(out_dir) if os.path.splitext(a)[-1] in ['.bmp' , '.png']]
    for im in ims:
        pre, ext = os.path.splitext(im)
        js_path = os.path.join(out_dir, pre+'.json')
        if os.path.exists(js_path):
            line = '{}||{}\n'.format(os.path.join(out_dir, im)[5:], js_path[5:])
            print(line)
            tmp.append(line)
    random.shuffle(tmp)
    # train_all.extend(tmp[:int(len(tmp)*0.7)])
    train_all.extend(tmp)
    # test_all.extend(tmp[int(len(tmp)*0.7):])

    # 写入新的txt
    tr_txts = open(tr_txt, 'w')
    # te_txts = open(te_txt, 'w')

    for tr in train_all:
        tr_txts.write(tr)

    # for te in test_all:
        # te_txts.write(te)



if flag == 3:
    # 把新数据的train, test直接加到 all_test_4defects.txt all_train_4defects.txt 后面
    txt_root_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/codes_for_cut_2bins/roi_debug'
    tr_txts = open(os.path.join(txt_root_dir, 'all_train_4defects.txt'), 'a')
    te_txts = open(os.path.join(txt_root_dir, 'all_test_4defects.txt'), 'a')

    train_all = open(tr_txt, 'r').readlines()
    test_all = open(te_txt, 'r').readlines()
    train_all = [a for a in train_all if len(a) > 1]
    test_all = [a for a in test_all if len(a) > 1]

    for tr in train_all:
        tr_txts.write(tr)

    for te in test_all:
        te_txts.write(te)


if flag == 3:
    tr_txt = './till_0320_byyy_train.txt'
    te_txt = './till_0320_byyy_test.txt'
    all_jss = []
    # merge bian_yayin txts
    dir1 = out_dir
    dir2 = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/byyy/3.17/roi_2bins'
    data_list = []
    data_list.append(dir1)
    data_list.append(dir2)
    for dir_ in data_list:
        js_paths = [os.path.join(dir_, a) for a in os.listdir(dir_) if '.json' in a]
        for js_path in js_paths:
            im = os.path.basenmae(js_path).split('.')[0] + '.png'
            line = '{}||{}\n'.format(os.path.join(dir_, im)[5:], js_path[5:])
            print(line)
            all_jss.append(line)


    random.shuffle(all_jss)
    tr_txts = open(tr_txt, 'w')
    te_txts = open(te_txt, 'w')

    # 所有数据加入训练..
    for tr in all_jss:
        tr_txts.write(tr)

    for te in all_jss[:13]:
        te_txts.write(te)





