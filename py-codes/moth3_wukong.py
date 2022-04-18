import os
import random

dir1 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao3-8/input_image_data/2022-03-08/6/1/1/OK'
dir2 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao3-8/input_image_data/2022-03-08/6/1/2/NG'
dir3 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao3-8/input_image_data/2022-03-08/6/2/NG'
dir4 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao3-8/input_image_data/2022-03-08/5/1/1/OK'
dir5 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao3-8/input_image_data/2022-03-08/5/1/2/NG'
dir6 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao3-8/input_image_data/2022-03-08/5/1/2/OK'
dir7 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao3-8/input_image_data/2022-03-08/5/2/NG'
dir8 = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/wukong/images_get/yijiao_origin_zg'

dirs = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]

all_data = []
for dir_ in dirs:
    jsons = [a for a in os.listdir(dir_) if '.json' in a]
    for js in jsons:
        line = '{}||{}\n'.format(os.path.join(dir_, js.split('.')[0]+'.png'), os.path.join(dir_, js))
        all_data.append(line)
random.shuffle(all_data)

tra_txt = open('./Q_wk_train.txt', 'w')
tes_txt = open('./Q_wk_test.txt', 'w')
trs = int(len(all_data)*0.7)
for line in all_data[:trs]:
    tra_txt.write(line)

for line in all_data[trs:]:
    tes_txt.write(line)






