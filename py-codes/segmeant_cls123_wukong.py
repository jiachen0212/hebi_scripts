# coding=utf-8
import os
import random
import json
random.seed(666)
from shutil import copyfile
import cv2


def img_json_dict(imgs, jsons, img_json):

    for i, img in enumerate(imgs):
        # print(img, jsons[i])
        img_json[img] = jsons[i]

    return img_json


def merge_help(pre, new, all_, flag=None):

    alls = open(all_, 'w')

    for line in pre:
        if len(line) > 0:
            alls.write(line)
    for line in new:
        if len(line) > 0:
            alls.write(line)

    print("pre{}: {}, new_cls1_test: {}".format(flag, len(pre), len(new)))



def merge_PSA_txts(defects):

    for defect in defects:
        if defect == 'PSA':
            # cls1 mergemodified后的txt, cls3merge未modified的.
            pre_cls1_train = open('./0214_modified1cls_train.txt', 'r').readlines()
            pre_cls1_test = open('./0214_modified1cls_test.txt', 'r').readlines()
            pre_cls3_train = open('./exps/masked_exps/3cls_train.txt', 'r').readlines()
            pre_cls3_test = open('./exps/masked_exps/3cls_test.txt', 'r').readlines()

            all_cls1_train = './14_and_23_cls1_train.txt'
            all_cls1_test = './14_and_23_cls1_test.txt'
            all_cls3_train = './14_and_23_cls3_train.txt'
            all_cls3_test = './14_and_23_cls3_test.txt'

            new_cls1_train = open('./0214_modified1cls_train.txt').readlines()
            new_cls1_test = open('./0214_modified1cls_test.txt').readlines()

            new_cls3_train = open('./0214_modified3cls_train.txt').readlines()
            new_cls3_test = open('./0214_modified3cls_test.txt').readlines()

            merge_help(pre_cls1_train, new_cls1_train, all_cls1_train, flag='cls1_train')
            merge_help(pre_cls1_test, new_cls1_test, all_cls1_test, flag='cls1_test')
            merge_help(pre_cls3_train, new_cls3_train, all_cls3_train, flag='cls3_train')
            merge_help(pre_cls3_test, new_cls3_test, all_cls3_test, flag='cls3_test')


def merge_help1(sub_train, train_, sub_test, test_):
    for line in sub_train:
        if len(line) > 0:
            train_.write(line)
    for line in sub_test:
        if len(line) > 0:
            test_.write(line)


def merge_wukong(clses, data):
    train_ = open('./{}wukong_train.txt'.format(data), 'w')
    test_ = open('./{}wukong_test.txt'.format(data), 'w')
    for sub_cls in clses:
        train = open('./{}{}_{}_train.txt'.format(data, 'wukong', sub_cls), 'r')
        test = open('./{}{}_{}_test.txt'.format(data, 'wukong', sub_cls), 'r')
        trains = train.readlines()
        tests = test.readlines()
        print(len(tests), sub_cls)
        merge_help1(trains, train_, tests, test_)


def json_label_check(js_path, label_list):
    # img 是否损坏check
    # try:
    #     cv2.imread(img_path)
    # except:
    #     return False
    data = json.load(open(js_path, 'r'))
    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            if cls_['label'] in label_list:
                return True
        return False
    else:
        return False


def generate_train_test_txt(js_dir, data_dir, defects, cls_dit, data):

    for defect in defects:
        gong_weis = [str(i) for i in range(1, 5)]
        if defect == 'PSA':
            print(defect)
            img_json = dict()
            Cls = defects[defect]
            all_data = [[] for i in range(3)]
            for cls_index, cls_ in enumerate(Cls):
                Cls_path = os.path.join(data_dir, defect, cls_)
                Json_path = os.path.join(js_dir, defect, cls_)
                try:
                    sub_clses = os.listdir(Cls_path)
                except:
                    continue
                for sub_cls in sub_clses:
                    for dir_ in gong_weis:
                        try:
                            imgs = os.listdir(os.path.join(Cls_path, sub_cls, dir_))
                        except:
                            continue
                        full_imgs = [os.path.join(os.path.join(Cls_path, sub_cls, dir_), a) for a in imgs]
                        full_jsons = [
                            os.path.join(os.path.join(Json_path, sub_cls, dir_), "{}.json".format(a.split('.')[0])) for
                            a in imgs]
                        img_json = img_json_dict(full_imgs, full_jsons, img_json)
                        if dir_ in ['1', '2']:
                            all_data[0].extend(full_imgs)
                        elif dir_ == '3':
                            all_data[1].extend(full_imgs)
                        else:
                            all_data[2].extend(full_imgs)


            for cls_index, each_cls in enumerate(all_data):

                cls_yes = []
                train_txt = open('./{}{}cls_train.txt'.format(data, cls_index+1), 'w')
                test_txt = open('./{}{}cls_test.txt'.format(data, cls_index+1), 'w')
                a, b, c = 0, 0, 0
                for img in each_cls:
                    js = img_json[img]
                    if os.path.exists(js):
                        defect_list = cls_dit[str(cls_index)]
                        # print("cls{}, defect_list: {}".format(cls_index+1, defect_list))
                        if json_label_check(js, defect_list):
                            line = "{}||{}\n".format(img[5:], js[5:])
                            cls_yes.append(line)
                            c += 1
                        b += 1
                    a += 1
                print("cls_{}, all_lens: {}, json_lens:{}, cur_defect_lens: {}".format(cls_index+1, a, b, c))
                random.shuffle(cls_yes)
                train_len = int(len(cls_yes)*0.7)
                for line in cls_yes[:train_len]:
                    train_txt.write(line)
                for line in cls_yes[train_len:]:
                    test_txt.write(line)
                # print("cls_{}, train: {}, test:{}".format(cls_index+1, train_len, len(cls_yes)-train_len))
        elif defect == 'wukong':
            print(defect)
            img_json = dict()
            for sub_cls in defects[defect]:
                train_txt = open('./{}{}_{}_train.txt'.format(data, defect, sub_cls), 'w')
                test_txt = open('./{}{}_{}_test.txt'.format(data, defect, sub_cls), 'w')
                cls_yes = []
                a, b, c = 0, 0, 0
                img_path = os.path.join(data_dir, defect, sub_cls)
                try:
                    json_path = os.path.join(js_dir, defect, sub_cls)
                    imgs = os.listdir(img_path)
                except:
                    continue
                full_imgs = [os.path.join(img_path, a) for a in imgs]
                full_jsons = [os.path.join(json_path, "{}.json".format(a.split('.')[0])) for a in imgs]
                img_json = img_json_dict(full_imgs, full_jsons, img_json)

                for img in full_imgs:
                    js = img_json[img]
                    if os.path.exists(js):
                        if json_label_check(js, [sub_cls]):
                            line = "{}||{}\n".format(img[5:], js[5:])
                            cls_yes.append(line)
                            c += 1
                        b += 1
                    a += 1
                print("{}, all_lens: {}, json_lens:{}, cur_defect_lens: {}".format(sub_cls, a, b, c))
                random.shuffle(cls_yes)
                train_len = int(len(cls_yes)*0.7)
                for line in cls_yes[:train_len]:
                    train_txt.write(line)
                for line in cls_yes[train_len:]:
                    test_txt.write(line)
                print("{}, train: {}, test:{}".format(sub_cls, train_len, len(cls_yes)-train_len))



def merge_wukong_txts(all_wukong_train, all_wukong_test, data_list):
    for ind, data in enumerate(data_list):
        lines = open(data, 'r').readlines()
        for line in lines:
            if ind < 2:
                all_wukong_train.write(line)
            else:
                all_wukong_test.write(line)


def img_and_mask(img_dir, mask_dir):

    masks = os.listdir(mask_dir)
    ims = os.listdir(img_dir)
    for im in ims:
        path = os.path.join(img_dir, im)
        if im not in masks:
            os.remove(path)




if __name__ == "__main__":

    data = '0214'
    js_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/jsons' .format(data)  # r'D:\work\project\DL\hebi\data\20220214\jsons'
    data_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs'.format(data)    # r'D:\work\project\DL\hebi\data\20220214\data'

    # 0226data imgs, masked 对齐
    # masked = '/data/home/jiachen/data/seg_data/hebi/20220226/imgs/masked/3cls/loubai/4'
    # imgs = '/data/home/jiachen/data/seg_data/hebi/20220226/imgs/PSA/3cls/loubai/4'
    # img_and_mask(imgs, masked)

    defects = {'PSA': ['1cls', '2cls', '3cls'], "wukong": ['yijiao', 'jinshusi', 'keli', 'maosi']}
    cls_dit = {"0": ['kengdian', 'tudian', 'yashang', 'juanbian', 'pengshang'], "1": ['yayin', 'heixian'],
    "2": ['lanjiao', 'loubai', 'jiaozhou', 'qipao']}
    # generate_train_test_txt(js_dir, data_dir, defects, cls_dit, data)

    # 合并0214, 0223的PSA数据.
    # merge_PSA_txts(defects)


    # 合并wukong的4中缺陷的train, test.
    # for data in ['0214', '0223']:
    #     merge_wukong(defects['wukong'], data)

    # 合并0214, 0223的wukong数据.
    all_wukong_train = open('./14_and_23_wukong_train.txt', 'w')
    all_wukong_test = open('./14_and_23_wukong_test.txt', 'w')
    data_list = ['0214wukong_train.txt', '0223wukong_train.txt', '0214wukong_test.txt', '0223wukong_test.txt']
    merge_wukong_txts(all_wukong_train, all_wukong_test, data_list)

    # 合并0214, 0226的cls3数据.
    # all_cls3_train = open('./0214_0226_cls3_train.txt', 'w')
    # all_cls3_test = open('./0214_0226_cls3_test.txt', 'w')
    # data_list = ['0214_modified3cls_train.txt', '0226_modified3cls_train.txt', '0214_modified3cls_test.txt', '0226_modified3cls_test.txt']
    # merge_wukong_txts(all_cls3_train, all_cls3_test, data_list)
