import cv2
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from scipy import spatial
import os
from PIL import Image
import math

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


def find_farthest_two_points(points, metric="euclidean"):
    """
    快速找出点集中最远距离的两个点：凸包 + 距离计算
    Args:
        points (numpy.ndarray, N x dim): N个d维向量
        metric ("euclidean", optional): 距离度量方式，见scipy.spatial.distance.cdist
    Returns:
        np.ndarray: 两点坐标
    """
    hull = spatial.ConvexHull(points)
    hullpoints = points[hull.vertices]
    hdist = spatial.distance.cdist(hullpoints, hullpoints, metric=metric)
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    return [hullpoints[bestpair[0]], hullpoints[bestpair[1]]]


def sdk_post(predict, Confidence=None, num_thres=None, yayin_limit=None, roi=None, tmp_hw=None):
    defect_index_cls = dict()
    points = []
    boxes = []
    line_points = []
    lines = []
    num_class = predict.shape[1]
    map_ = np.argmax(onnx_predict[0], axis=1)
    # print(f'pixel_classes: {np.unique(map_)}')
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

                    if number_thre > num_thres[i] and score_j > Confidence[i]:
                        contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnt = contours[0]
                        cnt = cnt.reshape(cnt.shape[0], -1)
                        if cnt.shape[0] < 3:
                            continue
                        # print(f'classes: {i}, nums: {number_thre}')

                        # 最小凸包
                        hull = np.squeeze(cv2.convexHull(cnt))
                        # 超过检出range部分的缺陷不显示
                        if np.max(hull, axis=0)[0] <= tmp_hw[0] and np.max(hull, axis=0)[1] <= tmp_hw[1]:

                            # 得到缺陷的最小外接旋转矩形.
                            rect = cv2.minAreaRect(cnt)
                            # 得到旋转矩形的端点
                            box = cv2.boxPoints(rect)
                            box_d = np.int0(box)

                            # 满足面积条件的缺陷, 再计算缺陷的, 最远两个点间的距离. 可对长条形缺陷做筛选.
                            line_point = find_farthest_two_points(np.squeeze(cnt))
                            line_points.append([line_point[0].tolist(), line_point[1].tolist()])
                            tmp = line_point[1] - line_point[0]
                            line = math.hypot(tmp[0], tmp[1])
                            # 有最短检出长度限制
                            if yayin_limit:
                                if line >= yayin_limit:
                                    temo_predict += temp * i
                                    boxes.append(box_d)
                                    lines.append(line)
                            else:
                                temo_predict += temp * i
                                boxes.append(box_d)
                                lines.append(line)

    return temo_predict, boxes


def show_roied_ims(test_paths, roi):

    for test_path in test_paths:
        img = Image.open(test_path).convert('RGB')
        tmp = np.asarray(img)
        img = tmp[roi[0]:roi[2], roi[1]:roi[3]]
        h, w = img.shape[0], img.shape[1]
        img_upper = img[:h//2, :]
        img_lower = img[h//2:, :]
        cv2.imshow('1', img_upper)
        cv2.waitKey(1000)
        cv2.destroyWindow('1')
        cv2.imshow('2', img_lower)
        cv2.waitKey(1000)
        cv2.destroyWindow('2')


if __name__ == "__main__":

    root_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/shimo/model_gw1/shimo_gw1_keli_qipao_bianxing_yayin_aotudian'

    # roi边界冗余 纵横起始点
    roi = [0, 1000, 7580, 7410]

    defcet_dict = {'0': 'ok', '1': 'ng'}  # ng: aodain, tidian, yayin, bianxing, qipao, kelli
    # label_map: &label_map
    # - bg
    # - ng  : aodain, tidian, yayin, bianxing, qipao, kelli
    # 网络输出层: 2层

    WH_ = roi[3] - roi[1], (roi[2] - roi[0]) // 2

    # 输入模型的尺寸
    size = [3000, 1500]

    # 开放给前端的超参, 可针对每个缺陷类别设置不同的值
    # 1.置信度分数
    Confidence = [0.5] * len(defcet_dict)
    # 2.面积过滤阈值: 像素个数小于num_thres的不检出, 这个参数可开放出来给前端调
    num_thres = [200] * len(defcet_dict)
    # 3.检出范围[7200, 7200], 超过此范围缺陷不显示. 这个参数可写死也可提供出来给前端调.
    tmp_hw = [7000, 3500]
    tmp_hw = [tmp_hw[i]/(WH_[i]/size[i]) for i in range(2)] # 等比缩放

    # 模型mean,std值
    mean_ = [123.675, 116.28, 103.53]
    std_ = [58.395, 57.12, 57.375]


    flag = {"0": "upper", "1": "lower"}
    test_dir = os.path.join(root_path, 'tests')
    test_paths = [os.path.join(test_dir, test_name) for test_name in os.listdir(test_dir)]

    # roi_check
    # show_roied_ims(test_paths, roi)

    # 导入模型
    onnx_path = os.path.join(root_path, '2000.onnx')
    onnx_session = ort.InferenceSession(onnx_path)

    res_dir = os.path.join(root_path, 'res_dir')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for test_path in test_paths:

        im_name = os.path.basename(test_path)

        img = Image.open(test_path).convert('RGB')
        tmp = np.asarray(img)


        # 后续会将模型推理后的结果替换至此
        imgcopy = tmp.copy()

        # roi剔除图像边界冗余
        img = tmp[roi[0]:roi[2], roi[1]:roi[3]]
        h, w = img.shape[0], img.shape[1]

        img_upper = img[:h//2, :]
        img_lower = img[h//2:, :]

        roi_img_res = np.zeros((h, w, 3), dtype=np.uint8)
        for ind, img in enumerate([img_upper, img_lower]):
            img = cv2.resize(img, (size[0], size[1]))
            img_ = sdk_pre(img, mean_, std_)
            onnx_inputs     = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
            onnx_predict = onnx_session.run(None, onnx_inputs)
            predict = softmax(onnx_predict[0], 1)
            map_, boxes = sdk_post(predict, Confidence=Confidence, num_thres=num_thres, roi=roi, tmp_hw=tmp_hw)
            mask_vis = label2colormap(map_)

            if boxes:
                # 画出缺陷的最小外接矩形
                for index, box in enumerate(boxes):
                    cv2.drawContours(mask_vis, [box], 0, (0, 255, 0), 2)

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

