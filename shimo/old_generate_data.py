# coding=utf-8
import os
import json
import cv2


# def till_get_im_and_js(file_path, pre_files):
#     if os.path.isdir(file_path):
#         os.path.isdir(file_path)
#         till_get_im_and_js(file_path, pre_files)
#     else:
#         pre_files.append(file_path)



# data_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1'
# defects = ['aokeng', 'tudian', 'yayin']
# defect_list = [[] for i in range(3)]

# for ind, defect in enumerate(defects):
#     defeat_dir = os.path.join(data_dir, defect)
#     files = [os.path.join(defeat_dir, a) for a in os.listdir(defeat_dir)]
#     for file_path in files:
#        till_get_im_and_js(file_path, defect_list[ind])

# for defs in defect_list:
#     print(len(defs))

def help(a):
    if a.split('.')[-1] in ['bmp', 'png', 'json']:
        return True


def help1(a):
    if a.split('.')[-1] in ['bmp', 'png']:
        return True


trains = []
tests = []

data_dir1 = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1/aokeng'
aokengs = [os.path.join(data_dir1, a) for a in os.listdir(data_dir1) if help(a)]
others = ['3.5', '3.6', '3.7']
for other in others:
    path_ = os.path.join(data_dir1, other)
    aokengs.extend([os.path.join(path_, a) for a in os.listdir(path_)])

ak_ims = [a for a in aokengs if help1(a)]
trs = int(len(ak_ims)*0.7)
for tr in ak_ims[:trs]:
    js_path = os.path.join(os.path.dirname(tr), os.path.basename(tr).split('.')[0]+'.json')
    if os.path.exists(js_path):
        line = "{}||{}\n".format(os.path.join(data_dir1, tr)[5:], js_path[5:])
        trains.append(line)

for te in ak_ims[trs:]:
    js_path = os.path.join(os.path.dirname(tr), os.path.basename(tr).split('.')[0]+'.json')
    if os.path.exists(js_path):
        line = "{}||{}\n".format(os.path.join(data_dir1, te)[5:], js_path[5:])
        tests.append(line)



data_dir2 = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1/tudian'
tudians = [os.path.join(data_dir2, a) for a in os.listdir(data_dir2) if help(a)]
others = ['3.5', '3.6']
for other in others:
    path_ = os.path.join(data_dir2, other)
    tudians.extend([os.path.join(path_, a) for a in os.listdir(path_)])

td_ims = [a for a in tudians if help1(a)]
trs = int(len(td_ims)*0.7)
for tr in td_ims[:trs]:
    js_path = os.path.join(os.path.dirname(tr), os.path.basename(tr).split('.')[0]+'.json')
    if os.path.exists(js_path):
        line = "{}||{}\n".format(os.path.join(data_dir2, tr)[5:], js_path[5:])
        trains.append(line)
    else:
        print(os.path.join(data_dir2, tr), js_path)

for te in td_ims[trs:]:
    js_path = os.path.join(os.path.dirname(tr), os.path.basename(tr).split('.')[0]+'.json')
    if os.path.exists(js_path):
        line = "{}||{}\n".format(os.path.join(data_dir2, te)[5:], js_path[5:])
        tests.append(line)



data_dir3 = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1/tudian'
yayins = [os.path.join(data_dir3, a) for a in os.listdir(data_dir3) if help(a)]
others = ['3.5', '3.6']
for other in others:
    path_ = os.path.join(data_dir3, other)
    yayins.extend([os.path.join(path_, a) for a in os.listdir(path_)])

yy_ims = [a for a in yayins if help1(a)]
trs = int(len(yy_ims)*0.7)
for tr in yy_ims[:trs]:
    js_path = os.path.join(os.path.dirname(tr), os.path.basename(tr).split('.')[0]+'.json')
    if os.path.exists(js_path):
        line = "{}||{}\n".format(os.path.join(data_dir3, tr)[5:], js_path[5:])
        trains.append(line)

for te in yy_ims[trs:]:
    js_path = os.path.join(os.path.dirname(tr), os.path.basename(tr).split('.')[0]+'.json')
    if os.path.exists(js_path):
        line = "{}||{}\n".format(os.path.join(data_dir3, te)[5:], js_path[5:])
        tests.append(line)


tr_txt = open('./train.txt', 'w')
te_txt = open('./test.txt', 'w')
for tr in trains:
    tr_txt.write(tr)
for te in tests:
    te_txt.write(te)