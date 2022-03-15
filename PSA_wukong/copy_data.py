import os
import shutil
import sys


def copy_core(source, target):
    # create the folders if not already exists
    if not os.path.exists(target):
        os.makedirs(target)

    # adding exception handling
    try:
        shutil.copy(source, target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
     print("Unexpected error:", sys.exc_info())


def copy_img_from_datalist(datalist, delimiter, target_path):
    with open(datalist, 'r') as r:
        for line in r.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            if len(line.split(delimiter)) == 2:
                img_path, json_path = line.split(delimiter)
                print("copy {}".format(img_path))
                print("copy {}".format(json_path))
                copy_core(img_path, target_path)
                copy_core(json_path, target_path)
            else:
                img_path = line
                print("copy {}".format(img_path))
                copy_core(img_path, target_path)

if __name__ == "__main__":
    datalist = "/data/home/hengzhiyao/codes/utils/project_resources/hebi/wukong_index.txt"
    delimiter = '||'
    target_path = '/data/home/sharedir/industrial/sbs/PROJECTS/Hebi/conf_zg/hebi_station5'

    copy_img_from_datalist(datalist, delimiter, target_path)