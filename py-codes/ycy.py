import os

dir1 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/0328/roi_2bins'
dir2 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins/keli'
dir3 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins/bianxing'
dir4 = '/data/home/jiachen/data/seg_data/hebi/shimo/newQ/debug_2bins/qipao'
dir6 = '/data/home/jiachen/data/seg_data/hebi/shimo/aotuyybyyy/0328/roi_2bins'
dir7 = '/data/home/jiachen/data/seg_data/hebi/shimo/toumingmo/0328/roi_2bins'
txt = open('./0322_28_bm.txt', 'w')
dirs = [dir1, dir2, dir3, dir4, dir6, dir7]
for dir_ in dirs:
    jss = os.listdir(dir_)
    jss = [a for a in jss if '.json' in a]
    for js in jss:
        line = '{}||{}\n'.format(os.path.join(dir_, js.split('.')[0]+'.png')[5:], os.path.join(dir_, js)[5:])
        txt.write(line)
