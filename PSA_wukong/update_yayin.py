# coding=-utf-8
import cv2
import os
import json
from shutil import copyfile
from cut_js_img_help import help


def json_label_check(js_path, label_list):

    if not os.path.exists(js_path):
        return False
    data = json.load(open(js_path, 'r'))
    if len(data['shapes']) > 0:
        for cls_ in data['shapes']:
            if cls_['label'] in label_list:
                return True
        return False
    else:
        return False


source_path = '/data/home/jiachen/data/seg_data/hebi/Qyayin/hx_imgs'
json_dir = '/data/home/jiachen/data/seg_data/hebi/Qyayin/hx_jss'
ims = os.listdir(source_path)
for im in ims:
	path_ = os.path.join(source_path, im)
	try:
		img = cv2.imread(path_)
		h = img.shape[0]
	except:
        os.remove(path_)
        continue
	js_path = os.path.join(json_dir, im[:-3]+'json')
	if not json_label_check(js_path, ['yayin', 'heixian']):
        os.remove(js_path)


# txt = open('./yayin.txt', 'r')
# lines = txt.readlines()
# lines = [a[:-1] for a in lines if len(a) > 1]

# source_path = '/data/home/jiachen/data/seg_data/hebi/1'
# json_dir = '/data/home/jiachen/data/seg_data/hebi/2'

# for line in lines:
# 	im_path = '/data{}'.format(line.split('||')[0])
# 	im_name = os.path.basename(im_path)
# 	js_path = '/data{}'.format(line.split('||')[1])
# 	js_name = os.path.basename(js_path)
# 	copyfile(im_path, os.path.join(source_path, im_name))
# 	copyfile(js_path, os.path.join(json_dir, js_name))


# 切割2bins
res_img_path, res_js_path = '/data/home/jiachen/data/seg_data/hebi/Qyayin/2bins_imgs', '/data/home/jiachen/data/seg_data/hebi/Qyayin/2bins_jss'

all_imgs, all_jss = '/data/home/jiachen/data/seg_data/hebi/mask_yayin_2bins', '/data/home/jiachen/data/seg_data/hebi/json_yayin'


if not os.path.exists(res_img_path):
    os.makedirs(res_img_path)
if not os.path.exists(res_js_path):
    os.makedirs(res_js_path)

split_target = (2, 1)
help(source_path, json_dir, res_js_path, res_img_path, split_target)

for im in os.listdir(res_img_path):
    copyfile(os.path.join(res_img_path, im), os.path.join(all_imgs, im))
    os.remove(os.path.join(res_img_path, im))
os.rmdir(res_img_path)

for js in os.listdir(res_js_path):
    copyfile(os.path.join(res_js_path, js), os.path.join(all_jss, js))


# 落盘成txt
jss = os.listdir(res_js_path)
new_txt = open('./0314.txt', 'w')
for js in jss:
    line = "{}||{}\n".format(os.path.join(all_imgs, js[:-4]+'png')[5:], os.path.join(all_jss, js)[5:])
    print(line)
    new_txt.write(line)

for js in os.listdir(res_js_path):
    os.remove(os.path.join(res_js_path, js))
os.rmdir(res_js_path)