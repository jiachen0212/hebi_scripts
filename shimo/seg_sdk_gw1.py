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


def sdk_post(predict, Confidence=0.5, Threshold=None, Max_Threshold=None):
    '''
    Args:
        Confidence: 缺陷的置信度，低于置信度的缺陷将会被过滤，默认0.8
        Threshold: 少于最少像素的mask将会被过滤，默认[0,0,0]
        Max_Threshold=[1000, 1000, 1000, 1000], 缺陷最大面积过滤.
    '''

    # 缺陷index和缺陷的像素个数
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
                        print(f'classes: {i}, nums: {number_thre}')

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


def get_test_data_path(test_data_txt):

    lines = open(test_data_txt).readlines()
    test_paths = []
    for line in lines:
        test_paths.append(line.split('||')[0])
    # print(test_paths[0])
    return test_paths


if __name__ == "__main__":

    # model for gongwei1
    model_size = [2048, 2048]
    mean_ = [52.0965, 52.0965, 52.0965]
    std_ = [52.326, 52.326, 52.326]

    defect_dict = {'0': 'ok', '1': 'ng'}  # ng代表: aodian tudian yayin这三种缺陷

    root_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi_反面/model_gw1'
    onnx_path = os.path.join(root_path, '2000.onnx')
    img_sava_path = os.path.join(root_path, 'test_result')

    test_paths = [os.path.join(root_path, 'test.png')]

    if not os.path.exists(img_sava_path):
        os.makedirs(img_sava_path)

    onnx_session = ort.InferenceSession(onnx_path)

    Threshold = [0] * len(defect_dict)
    for i in test_paths:
        if i[-3:] in ['png', 'jpg', 'bmp']:
            print(i)
            tmp = cv2.imdecode(np.fromfile(i, dtype=np.uint8), -1)
            h, w = tmp.shape[0], tmp.shape[1]

            half_high = model_size[1] // 2
            img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (model_size[0], model_size[1]), interpolation=cv2.INTER_LINEAR)
            im_name = os.path.basename(i)
            img_ = sdk_pre(img, mean_, std_)
            onnx_inputs = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
            onnx_predict = onnx_session.run(None, onnx_inputs)
            predict = softmax(onnx_predict[0], 1)
            map_, points, mask_map, defect_index_cls = sdk_post(predict, Threshold=Threshold, Max_Threshold=None)

            # visualize
            mask_vis = label2colormap(map_)
            for p in points:
                defect_index_area = defect_index_cls['{}_{}'.format(p[0][0], p[0][1])]

                cv2.putText(mask_vis, '{}'.format(defect_dict[str(defect_index_area[0])]), (p[0][0], p[0][1]), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 0, 255), 2)
                cv2.putText(mask_vis, 'area: {}'.format(defect_index_area[1]), (p[1][0], p[1][1]), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 0, 255), 2)
                img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
                cv2.imwrite(os.path.join(img_sava_path, im_name[:-3]+'jpg'), img_save)

