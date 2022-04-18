import cv2
import os
import json
import random


def json_label_check(js_path, img_path, label_list):
    # img 是否损坏check
    try:
        cv2.imread(img_path)
    except:
        return False
    data = json.load(open(js_path, 'r'))
    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            if cls_['label'] in label_list:
                return True
        return False
    else:
        return False



def cv2_imread_check(im_paths, js_paths):
    for ind, path in enumerate(im_paths):
        path = '/data' + path
        js_path = '/data' + js_paths[ind]
        try:
            img = cv2.imread(path)
            h = img.shape[0]
        except:
            os.remove(path)
            os.remove(js_path)
            print('remove: {}'.format(path))




def json_label_check1(js_path, label_list):
    data = json.load(open(js_path, 'r'))
    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            if cls_['label'] in label_list:
                return True
        return False
    else:
        return False

# js_dir = '/data/home/jiachen/data/seg_data/hebi/json_yayin'
# img_dir = '/data/home/jiachen/data/seg_data/hebi/mask_yayin_2bins'
# jss = os.listdir(js_dir)
# jss_path = [os.path.join(js_dir, a) for a in jss]
# for js in jss_path:
#     base_js = os.path.basename(js)
#     im_path = os.path.join(img_dir, base_js[:-4]+'png')
#     if not json_label_check(js, im_path, ['yayin', 'heixian']):
#         try:
#             os.remove(im_path)
#             os.remove(js)
#         except:
#             continue
#         print('not yayin or heixain')

# js_dir = '/data/home/jiachen/data/seg_data/hebi/json_yayin'
# img_dir = '/data/home/jiachen/data/seg_data/hebi/mask_yayin_2bins'
# ims = [a[:-4] for a in os.listdir(img_dir)]
# jss = [a[:-5] for a in os.listdir(js_dir)]
# for im in ims:
#     if im not in jss:
#         os.remove(os.path.join(img_dir, im+'.png'))



# js_dir = '/data/home/jiachen/data/seg_data/hebi/json_yayin'
# jss = os.listdir(js_dir)
# jss_path = [os.path.join(js_dir, a) for a in jss]
# for js in jss_path:
#     base_js = os.path.basename(js)
#     if not json_label_check1(js, ['yayin', 'heixian']):
#         os.remove(js)


js_dir = '/data/home/jiachen/data/seg_data/hebi/json_yayin'
img_dir = '/data/home/jiachen/data/seg_data/hebi/mask_yayin_2bins'
ims = [os.path.join(img_dir, a)[5:] for a in os.listdir(img_dir)]
jss = [os.path.join(js_dir, a[:-3]+'json')[5:] for a in os.listdir(img_dir)]

# cv2.imraed() check
cv2_imread_check(ims, jss)


pair = dict()
for ind, im in enumerate(ims):
    pair[im] = jss[ind]


random.shuffle(ims)

tr_txt = open('/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/single_bin_yayin/single_train.txt', 'w')
te_txt = open('/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/single_bin_yayin/single_test.txt', 'w')
trains = int(len(ims)*0.7)
for tr in ims[:trains]:
    line = '{}||{}\n'.format(tr, pair[tr])
    # print(line)
    tr_txt.write(line)

for te in ims[trains:]:
    line = '{}||{}\n'.format(te, pair[te])
    te_txt.write(line)
