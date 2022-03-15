# coding=utf-8
import json
import os

root_dir = '/../jsons' # '/../imgs'

dirs = [os.path.join(root_dir, '1'), os.path.join(root_dir, '2'), os.path.join(root_dir, '3'), os.path.join(root_dir, '4')]

for ind, dir_ in enumerate(dirs):
    ims = os.listdir(dir_)
    for im in ims:
        os.rename(os.path.join(dir_, im), os.path.join(dir_, '{}_{}.{}'.format(im.split('.')[0], ind+1, im.split('.')[1])))
