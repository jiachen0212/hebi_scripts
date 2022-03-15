import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
import os


def sdk_pre(im_path, mean_, std_, input_size):
    tmp = cv2.imdecode(np.fromfile(im_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img,(input_size[0], input_size[1]),interpolation=cv2.INTER_LINEAR)[np.newaxis,:,:,:]
    img = np.array(img,dtype=np.float32)
    img -=np.float32(mean_)
    img /= np.float32(std_)
    img = np.transpose(img,[0,3,1,2])

    return img


def predict_cls(img_path, mean_, std_, input_size):

    img = sdk_pre(img_path, mean_, std_, input_size)
    onnx_inputs = {onnx_session.get_inputs()[0].name: img.astype(np.float32)}
    onnx_predict = onnx_session.run(None, onnx_inputs)
    predict = softmax(onnx_predict[0], 1)[0].tolist()
    res = predict.index(max(predict))

    return res, predict


def get_lab(lab):
    lab = lab.strip('[,]')
    if lab[0] == '1':
        return 0
    else:
        return 1

if __name__ == "__main__":

    root_dir = r'D:\work\project\DL\hebi\cls2_yayin'
    root_data_dir = r'C:\Users\15974\Desktop\0214_0223_0226_subs_yayin'
    onnx_path = os.path.join(root_dir, '5000.onnx')
    onnx_session = ort.InferenceSession(onnx_path)

    guosha, lousha = [], []
    mean_ = [129.30, 124.07, 112.43]
    std_ = [68.17, 65.39, 70.42]

    tests = open(os.path.join(root_dir, r'0226_yayin_test.txt'), 'r').readlines()
    tests = [a.split('/')[-1] for a in tests if len(a) > 0]

    yayin_nums = 0
    input_size = [2048, 2048]
    for line in tests:
        img_name, gt = line.split(":")[0], line.split(":")[1][:-1]
        lab = get_lab(gt)
        yayin_nums += lab
        img_path = os.path.join(root_data_dir, img_name)
        pred, score = predict_cls(img_path, mean_, std_, input_size)
        print("img: {}, predict to: {}, gt: {}, score: {}".format(img_name, pred, lab, score))
        if lab == 0 and pred == 1:
            guosha.append(img_name)
        elif lab == 1 and pred == 0:
            lousha.append(img_name)
    print("ok_cls1: {}".format(guosha))
    print("cls1_ok: {}".format(lousha))
    ok_nums = len(tests) - yayin_nums
    print('oks: {}, yayins: {}'.format(ok_nums, yayin_nums))





