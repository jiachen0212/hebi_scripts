import json
import os

base_dir = '/Users/chenjia/Desktop/loubai'
for dir_ in range(1,5):
    path = os.path.join(base_dir, str(dir_))
    jss = os.listdir(path)
    for js in jss:
        js_path = os.path.join(path, js)
        data = json.load(open(js_path, 'r'))
        if len(data['shapes']) > 0:
            for cls_ in data['shapes']:
                label = cls_['label']
                if label == 'yain':
                    print(js_path)
