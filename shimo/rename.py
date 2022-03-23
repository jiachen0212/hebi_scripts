import os
import shutil


base_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/gw1'
def_list = ['aokeng', 'tudian', 'yayin']
sub_dirs = [['3.5', '3.6', '3.7'], ['3.5', '3.6'], ['3.5', '3.6']]
# test_dir = '/data/home/jiachen/project/seg_project/seg_2022/hebi/shimo_project/debug'

for ind, defect in enumerate(def_list):
    # 把 3.5 3.6 3.7下面的json, imgs rename后copy至path1
    path1 = os.path.join(base_dir, defect)
    for sub in sub_dirs[ind]:
        path2 = os.path.join(path1, sub)
        files = os.listdir(path2)
        ims = [a for a in files if a.split('.')[-1] in ['png', 'bmp']]
        jss = [a for a in files if a.split('.')[-1] in ['json']]
        for im in ims:
            # shutil.copy()
            pre, ext = os.path.splitext(im)
            shutil.copy(os.path.join(path2, im), os.path.join(path1, '{}{}{}'.format(im, sub, ext)))
            shutil.copy(os.path.join(path2, pre+'.json'), os.path.join(path1, '{}{}{}'.format(im, sub, '.json')))











