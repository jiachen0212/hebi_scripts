# coding=utf-8
import os
import json
import random
random.seed(33)
from shutil import copyfile


def img_json_dict(imgs, jsons, img_json):

    for i, img in enumerate(imgs):
        # print(img, jsons[i])
        img_json[img] = jsons[i]

    return img_json



def test(defects):

    for defect in defects:
        if defect == 'psa':
            Cls = defects[defect]
            # 4个工位的数据一起训
            psa_train = open('./123cls_train.txt', 'w')
            psa_test = open('./123cls_test.txt', 'w')
            for cls_ in Cls:
                train_txt = open('./{}_train.txt'.format(cls_), 'r')
                test_txt = open('./{}_test.txt'.format(cls_), 'r')
                trains = train_txt.readlines()
                tests = test_txt.readlines()
                for line in trains:
                    if len(line) > 0:
                        psa_train.write(line)
                for line in tests:
                    if len(line) > 0:
                        psa_test.write(line)


def json_label_check(js_path, label_list):
    data = json.load(open(js_path, 'r'))
    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            if cls_['label'] in label_list:
                return True
        return False
    else:
        return False


def split_train_test_txt(cls_yes, cls_no, train_txt, test_txt):
    random.shuffle(cls_yes)
    random.shuffle(cls_no)
    train1, train2 = int(len(cls_yes)*0.7), int(len(cls_no)*0.7)
    for line in cls_yes[:train1]:
        train_txt.write(line)
    for line in cls_yes[train1:]:
        test_txt.write(line)

    for line in cls_no[:train2]:
        train_txt.write(line)
    for line in cls_no[train2:]:
        test_txt.write(line)
    print('write train.txt and test.txt done!')


def generate_train_test_txt(js_dir, data_dir, defects):

    for defect in defects:
        gong_weis = [str(i) for i in range(1, 5)]
        if defect == 'PSA':
            print(defect)
            Cls = defects[defect]
            all_data = [[] for i in range(3)]
            all_jss = [[] for i in range(3)]
            for cls_index, cls_ in enumerate(Cls):
                Cls_path = os.path.join(data_dir, defect, cls_)
                Json_path = os.path.join(js_dir, defect, cls_)
                sub_clses = os.listdir(Cls_path)
                for sub_cls in sub_clses:
                    for dir_ in gong_weis:
                        imgs = os.listdir(os.path.join(Cls_path, sub_cls, dir_))
                        full_imgs = [os.path.join(os.path.join(Cls_path, sub_cls, dir_), a) for a in imgs]
                        full_jsons = [os.path.join(os.path.join(Json_path, sub_cls, dir_), "{}.json".format(a.split('.')[0])) for a in imgs]
                        if dir_ in ['1', '2']:
                            all_data[0].extend(full_imgs)
                        elif dir_ == '3':
                            all_data[1].extend(full_imgs)
                            all_jss[1].extend(full_jsons)
                        else:
                            all_data[2].extend(full_imgs)
            img_json2 = dict()
            cls2_imgs = all_data[1]
            cls2_jsons = all_jss[1]
            cls2_yes, cls2_no = [], []
            img_json = img_json_dict(cls2_imgs, cls2_jsons, img_json2)
            a, b = 0, 0
            for img in cls2_imgs:
                js = img_json[img]
                if os.path.exists(js):
                    b += 1
                    # json_file label check
                    if json_label_check(js, ['yayin', 'heixian']):
                        line = "{}:[0,1]\n".format(img)
                        cls2_yes.append(line)
                        a += 1
                    else:
                        line = "{}:[1,0]\n".format(img)
                        cls2_no.append(line)
                        b += 1
                else:
                    b += 1
                    line = "{}:[1,0]\n".format(img)
                    cls2_no.append(line)

            # write train,test.txt
            print("a: {}, b: {}".format(a, b))
            cls2_train = open('./classify_cls2_train_377.txt', 'w')
            cls2_test = open('./classify_cls2_test_377.txt', 'w')
            split_train_test_txt(cls2_yes, cls2_no, cls2_train, cls2_test)

        else:
            defect_index = {"0": "yijiao", "1": "jinshusi", "2":"keli", "3":"maosi"}
            print(defect)
            # wukong缺陷
            all_defect = defects[defect]
            defect_len = len(all_defect)
            # 存下所有子缺陷的img:json dict
            img_json = dict()
            all_data = [[] for i in range(defect_len)]
            all_json = [[] for i in range(defect_len)]
            for index, sub_cls in enumerate(all_defect):
                if sub_cls == "yijiao":
                    path1 = os.path.join(data_dir, defect, sub_cls)
                    json_path = os.path.join(js_dir, defect, sub_cls)
                    # /Users/chenjia/Downloads/Learning/SmartMore/2022/DL/赫比/赫比/2022.02.14/wukong/yijiao
                    imgs = os.listdir(path1)
                    full_imgs = [os.path.join(path1, a) for a in imgs]
                    full_jsons = [os.path.join(json_path, "{}.json".format(a.split('.')[0])) for a in imgs]
                    all_data[index].extend(full_imgs)
                    all_json[index].extend(full_jsons)

            for i in range(4):
                # yijiao test
                if i == 0:
                    a, b = 0, 0
                    train_txt = open('./yijiao_train.txt', 'w')
                    test_txt = open('./yijiao_test.txt', 'w')
                    cls_yes, cls_no = [], []
                    data1 = all_data[i]
                    json1 = all_json[i]
                    img_json1 = img_json_dict(data1, json1, img_json)
                    for img in data1:
                        js = img_json1[img]
                        if os.path.exists(js):
                            if json_label_check(js, [defect_index[str(i)]]):
                                line = "{}:[0,1]\n".format(img)
                                cls_yes.append(line)
                                a += 1
                            else:
                                b += 1
                                line = "{}:[1,0]\n".format(img)
                                cls_no.append(line)
                        else:
                            line = "{}:[1,0]\n".format(img)
                            cls_no.append(line)
                            b += 1
                    split_train_test_txt(cls_yes, cls_no, train_txt, test_txt)



if __name__ == "__main__":

    # jiqun
    js_dir = '/data/home/jiachen/data/seg_data/hebi/20220214/jsons/'   # r'D:\work\project\DL\hebi\data\20220214\jsons'
    data_dir = '/data/home/jiachen/data/seg_data/hebi/20220214/imgs'    # r'D:\work\project\DL\hebi\data\20220214\data'

    # windows
    # js_dir =  r'D:\work\project\DL\hebi\data\20220214\jsons'
    # data_dir = r'D:\work\project\DL\hebi\data\20220214\data'

    # mac
    # data_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/2022.02.14/data'
    # js_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/2022.02.14/jsons'


    defects = {'PSA': ['1cls', '2cls', '3cls'], "wukong": ['yijiao', 'jinshusi', 'keli', 'maosi']}
    generate_train_test_txt(js_dir, data_dir, defects)

    # 直接合并已经生成好的cls123的train test txt
    # test(defects)
