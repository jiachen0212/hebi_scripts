import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
import os


def sdk_pre(im_path):
    tmp = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img,(2048, 2048),interpolation=cv2.INTER_LINEAR)[np.newaxis,:,:,:]
    img = np.array(img,dtype=np.float32)
    img -=np.float32( [129.30, 124.07, 112.43])
    img /= np.float32( [68.17, 65.39, 70.42])
    img = np.transpose(img,[0,3,1,2])

    return img


def predict_cls(img_path):

    img = sdk_pre(img_path)
    onnx_inputs = {onnx_session.get_inputs()[0].name: img.astype(np.float32)}
    onnx_predict = onnx_session.run(None, onnx_inputs)
    predict = softmax(onnx_predict[0], 1)[0].tolist()
    res = predict.index(max(predict))

    return res, predict



if __name__ == "__main__":
    onnx_path = '/data/home/jiachen/project/seg_project/seg_2022/hebi/project/exps/classify_cls2/deploy/2000.onnx'

    onnx_session = ort.InferenceSession(onnx_path)

    ok_ok, cls1_cls1 = [], []
    ok_cls1, cls1_ok = [], []


    test_data_txts = open('/data/home/jiachen/project/seg_project/seg_2022/hebi/project/classify_cls2_test.txt', 'r').readlines()
    train_data_txts = open('/data/home/jiachen/project/seg_project/seg_2022/hebi/project/classify_cls2_train.txt', 'r').readlines()
    all_txts = [train_data_txts, test_data_txts]
    flags = ["test", "train"]
    for ind, txt in enumerate(all_txts):
        print("{}: ".format(flags[ind]))
        for line in txt:
            img_path = line.split('[')[0][:-1]
            # im_name = img_path.split('\\')[-1]
            im_name = img_path.split('/')[-1]
            print(line)
            lab = [int(a) for a in (line.split('[')[1][:-2]).split(',')]
            lab = lab.index(1)
            pred, score = predict_cls(img_path)
            # print("img: {}, predict to: {}, gt: {}, score: {}".format(im_name, pred, lab, score))
            print("img: {}, predict to: {}, gt: {}".format(im_name, pred, lab))
            if lab == 0 and pred == 0:
                ok_ok.append(im_name)
            elif lab == 0 and pred == 1:
                ok_cls1.append(im_name)
            elif lab == 1 and pred == 0:
                cls1_ok.append(im_name)
            elif lab == 1 and pred == 1:
                cls1_cls1.append(im_name)
        print("ok_cls1: {}".format(ok_cls1))
        print("cls1_ok: {}".format(cls1_ok))
        print('\n')





