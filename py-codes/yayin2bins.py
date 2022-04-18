import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from scipy import spatial
import os
from PIL import Image

def label2colormap(label):
    m = label.astype(np.uint8)
    r, c = m.shape[:2]
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2
    cmap[:, :, 2] = (m & 4) << 5
    return cmap


def sdk_pre(img_t, mean_, std_):
    img_t = img_t[np.newaxis,:,:,:]
    img = np.array(img_t, dtype=np.float32)
    img -= np.float32(mean_)
    img /= np.float32(std_)
    img = np.transpose(img, [0, 3, 1, 2])
    return img


def check_connect_comp(img, label_index):
    mask = np.array(img == label_index, np.uint8)
    num, label = cv2.connectedComponents(mask, 8)
    return mask, num, label


def sdk_post(predict, Confidence=0.5, Threshold=[0, 0, 0, 0], Max_Threshold=[1000, 1000, 1000, 1000]):

    defect_index_cls = dict()
    points = []
    num_class = predict.shape[1]
    map_ = np.argmax(onnx_predict[0], axis=1)
    print(f'pixel_classes: {np.unique(map_)}')
    mask_map = np.max(predict[0, :, :, :], axis=0)
    mask_ = map_[0, :, :]
    temo_predict = np.zeros(mask_.shape)
    score_print = np.zeros(mask_.shape)
    for i in range(num_class):
        if i == 0:
            continue
        else:
            _, num, label = check_connect_comp(mask_, i)
            for j in range(num):
                if j == 0:
                    continue
                else:
                    temp = np.array(label == j, np.uint8)
                    score_temp = temp * mask_map
                    locate = np.where(temp > 0)
                    number_thre = len(locate[0])
                    score_j = np.sum(score_temp) / number_thre
                    if number_thre > Threshold[i] and score_j > Confidence: #  and number_thre < Max_Threshold[i]:
                        contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnt = contours[0]
                        cnt = cnt.reshape(cnt.shape[0], -1)
                        if cnt.shape[0] < 3:
                            continue
                        # print(f'classes: {i}, nums: {number_thre}')

                        candidates = cnt[spatial.ConvexHull(cnt).vertices]
                        dist_mat = spatial.distance_matrix(candidates, candidates)
                        i_, j_ = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
                        defect_index_cls['{}_{}'.format(candidates[i_][0], candidates[i_][1])] = [i, number_thre]

                        temo_predict += temp * i
                        points.append([candidates[i_], candidates[j_]])
                        cv2.putText(score_print, 'confidence: ' + str(score_j)[:6],
                                    (candidates[i_][0], candidates[i_][1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 122, 2)
                        cv2.putText(score_print, 'nums: ' + str(number_thre)[:6],
                                    (candidates[j_][0], candidates[j_][1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 80, 2)

    return temo_predict, points, score_print, defect_index_cls


def crop_bins(img, step_w, step_h, bins=None, pre_cut=None):
    img = Image.fromarray(img)
    croppeds = dict()
    for i in range(bins[0]):
        for j in range(bins[1]):
            left = step_w * i
            upper = step_h * j
            right = step_w * (i+1)
            lower = step_h * (j+1)
            crop_area = (left, upper, right, lower)
            cropped = img.crop(crop_area)
            print(cropped.size)
            croppeds["{}_{}".format(i, j)] = cropped

    return croppeds


if __name__ == "__main__":

    root_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/model_for_工程/0307/cls2-seg-2bins'
    onnx_path = os.path.join(root_path, '10000.onnx')
    onnx_session = ort.InferenceSession(onnx_path)
    size = [1898, 1798]
    mean_ = [52.0965, 52.0965, 52.0965]
    std_ = [52.326, 52.326, 52.326]
    defcet_dict = {'0': 'ok', '1': 'yayin'}
    tmp = {"0": "left", "1": "right"}

    test_name = 'yayin.png'
    test_path = os.path.join(root_path, test_name)
    img = cv2.imread(test_path)
    h, w = img.shape[0], img.shape[1]
    img_left = np.zeros((h, w//2, 3), dtype=np.uint8)
    img_right = np.zeros((h, w//2, 3), dtype=np.uint8)
    img_left[:,:,:] += img[:h, :w//2, :]
    img_right[:,:,:] += img[:h, w//2:, :]


    for ind, img in enumerate([img_left, img_right]):
        img = cv2.resize(img, (w//4, h//2))
        img_ = sdk_pre(img, mean_, std_)
        onnx_inputs     = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
        onnx_predict = onnx_session.run(None, onnx_inputs)
        predict = softmax(onnx_predict[0], 1)
        map_, points, mask_map, defect_index_cls = sdk_post(predict, Max_Threshold=None)
        mask_vis = label2colormap(map_)
        if points:
            # cv2.addWeighted: 融合两张图像: [mask_vis, img]
            img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
            mask_map = np.array(mask_map, dtype=np.uint8)
            mask_map_ = cv2.cvtColor(mask_map, cv2.COLOR_GRAY2BGR)

            save_path = os.path.join(root_path, 'yayin{}.jpg'.format(tmp[str(ind)]))
            img_save = cv2.resize(img_save, (w//2, h), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(save_path, img_save)
            cv2.imshow('res', img_save)
            cv2.waitKey(10000)
