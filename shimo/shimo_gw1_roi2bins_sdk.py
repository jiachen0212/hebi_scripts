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

                    # Confidence: 置信度参数, 可自行给入值.
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

                        # 开放面积过滤
                        if number_thre > 80:
                            print('number_thre是缺陷像素个数, 可抽象为缺陷的面积信息, 这个值可用来做一些阈值处理.')

                        temo_predict += temp * i
                        points.append([candidates[i_], candidates[j_]])
                        cv2.putText(score_print, 'confidence: ' + str(score_j)[:6],
                                    (candidates[i_][0], candidates[i_][1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 122, 2)
                        cv2.putText(score_print, 'nums: ' + str(number_thre)[:6],
                                    (candidates[j_][0], candidates[j_][1]), cv2.FONT_HERSHEY_PLAIN, 1.0, 80, 2)

    return temo_predict, points, score_print, defect_index_cls



if __name__ == "__main__":

    root_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi_反面/model_gw1/gw1_roi_cut_2bins'
    onnx_path = os.path.join(root_path, '100.onnx')
    onnx_session = ort.InferenceSession(onnx_path)

    # 输入模型的尺寸, 模型的mean和std
    # size = [2048, 2048]
    size = [2531, 54450]    # 和集群的input_size反着来就可以了.
    mean_ = [123.675, 116.28, 103.53]
    std_ = [58.395, 57.12, 57.375]

    defcet_dict = {'0': 'ok', '1': 'ng'}  # ng: aodain, tidian, yayin 三种缺陷
    flag = {"0": "upper", "1": "lower"}

    test_name = 'test.png'
    test_paths = [os.path.join(root_path, test_name)]
    res_dir = os.path.join(root_path, 'res_dir')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # roi边界冗余
    roi = [0, 918, 7594, 7594]

    for test_path in test_paths:

        im_name = os.path.basename(test_path)

        img = Image.open(test_path).convert('RGB')
        tmp = np.asarray(img)

        # 后续会将模型推理后的结果替换至此
        imgcopy = tmp.copy()

        # roi剔除图像边界冗余
        img = tmp[roi[0]:roi[2], roi[1]:roi[3]]
        h, w = img.shape[0], img.shape[1]

        img_upper = img[:h//2, :]  # shape: 3797, 6676 cv2的hw. 和集群的resize是反着的.
        # cv2.imshow('1', img_upper)
        # cv2.waitKey(5000)
        img_lower = img[h//2:, :]

        roi_img_res = np.zeros((h, w, 3), dtype=np.uint8)
        for ind, img in enumerate([img_upper, img_lower]):
            img = cv2.resize(img, (size[0], size[1]))
            img_ = sdk_pre(img, mean_, std_)
            onnx_inputs     = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
            onnx_predict = onnx_session.run(None, onnx_inputs)
            predict = softmax(onnx_predict[0], 1)
            map_, points, mask_map, defect_index_cls = sdk_post(predict, Max_Threshold=None)
            mask_vis = label2colormap(map_)

            if points:
                # 这个可输出upper和lower的缺陷情况
                print("{} has defect".format(flag[str(ind)]))

                img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
                img_save = cv2.resize(img_save, (w, h//2), interpolation=cv2.INTER_LINEAR)
                if ind == 0:
                    roi_img_res[:h//2, :, :] += img_save
                else:
                    roi_img_res[h//2:, :, :] += img_save
            else:
                print("{} has no defect".format(flag[str(ind)]))

        # roi_img_res替换至imgcopy
        imgcopy[roi[0]:roi[2], roi[1]:roi[3], :] = roi_img_res
        cv2.imwrite(os.path.join(res_dir, im_name), imgcopy)
