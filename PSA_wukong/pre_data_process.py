# coding=utf-8
import cv2
import os


def imread_check(data_dir):

    ims = [os.path.join(data_dir, a) for a in os.listdir(data_dir)]
    print('pre_ims_lens: {}'.format(len(ims)))
    for im in ims:
        try:
            img = cv2.imread(im)
        except:
            os.remove(im)

    print('check_ed_ims_lens: {}'.format(len(os.listdir(data_dir))))


if __name__ == "__main__":

    data_dir = '/data/home/jiachen/data/seg_data/hebi/20220226/imgs/PSA/3cls/loubai/4'
    # data_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/20220226/imgs/PSA/3cls/loubai/4'
    imread_check(data_dir)