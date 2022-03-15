# coding=utf-8
import os
# from locate1234 import locate_fun

txt_dir = './mask'
txts = [os.path.join(txt_dir, a) for a in os.listdir(txt_dir)]
need_mask_paths = []
for txt in txts:
    lines = open(txt, 'r').readlines()
    lines = [a.split('||')[0] for a in lines if len(a) > 0]
    need_mask_paths.extend(lines)

dirs = []
for line in need_mask_paths:
    dirs.append("/data{}".format(line))

for img_path in dirs:
    print(img_path)


