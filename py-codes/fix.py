# coding=utf-8
import os
from shutil import copyfile


masked = '/data/home/jiachen/data/seg_data/hebi/mask_yayin'
org = '/data/home/jiachen/data/seg_data/hebi/full_yayin'

alls = os.listdir(org)
masks = os.listdir(masked)
for im in alls:
    if im not in masks:
        copyfile(os.path.join(org, im), os.path.join(masked, im))


