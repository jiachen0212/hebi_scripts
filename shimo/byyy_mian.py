import os
import shutil
from debug import help
import random
import cv2



def roi_imgs(out_dir, base_dir):
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



base_dir ='/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0318'
out_dir = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0318/roi_2bins'

flag = 1

# base_dir ='/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi_反面/gw1/3.18/1'
# out_dir = '/Users/chenjia/Desktop/test'


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


train_all = []
test_all = []
roi = (670,0,7594, 7269)  # 横,纵 起终点
split_target = (1, 2)



# only check img_roi
if flag == 0:
    roi_imgs(out_dir, base_dir)


if flag == 1:
    tmp = []
    help(base_dir, roi, split_target, out_dir)

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

    # 写入新的txt
    tr_txt = open('./0318_train.txt', 'w')
    te_txt = open('./0318_test.txt', 'w')

    for tr in train_all:
        tr_txt.write(tr)

    for te in test_all:
        te_txt.write(te)


    # 把新数据的train, test直接加到 all_test_4defects.txt all_train_4defects.txt 后面
    txt_root_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/codes_for_cut_2bins/roi_debug'
    tr_txt = open(os.path.join(txt_root_dir, 'all_train_4defects.txt'), 'a')
    te_txt = open(os.path.join(txt_root_dir, 'all_test_4defects.txt'), 'a')

    for tr in train_all:
        tr_txt.write(tr)

    for te in test_all:
        te_txt.write(te)



