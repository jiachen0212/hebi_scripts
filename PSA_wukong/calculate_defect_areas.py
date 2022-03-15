# coding=utf-8
import os
import json




if __name__ == "__main__":

    defect_dict = {"cls1": ['kengdian', 'tudian', 'yashang', 'juanbian', 'pengshang'], "cls3": ['lanjiao', 'loubai', 'jiaozhou', 'qipao']}
    all_defects = ['kengdian', 'tudian', 'yashang', 'juanbian', 'pengshang', 'lanjiao', 'loubai', 'jiaozhou', 'qipao']


    cls3_txt_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/2022.02.14/jsons/cls3_txts/'
    cls1_txt_dir = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/2022.02.14/jsons/cls1_txts/'
    cls13_dir = [cls1_txt_dir, cls3_txt_dir]

    for ind, cls_ in enumerate(cls13_dir):
        tr_te_txts = os.listdir(cls_)
        for txt in tr_te_txts:
            lines = open(os.path.join(cls_, txt), 'r')
            lines = [a[:-1] for a in lines if len(a) > 0]
            for line in lines:
                # json_path = '/data{}'.format(line.split('||')[1])
                json_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/hebi_data/2022.02.14/jsons/PSA/1cls/bianxing/1/20220209_164708435_Z__OK.json'
                js = json.load(open(json_path, 'r'))
                for lab in js["shapes"]:
                    lab_name = lab['label']
                    print(lab)
                    print("points: {}".format(len(lab['points'])))



