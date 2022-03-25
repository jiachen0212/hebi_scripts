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
    找出点集中最远距离的两个点：凸包 + 距离计算
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


def sdk_post(predict, defcets, Confidence=None, num_thres=None, yayin_limit=None, roi=None):

    defects_nums = [0]*len(defcets)

    points = []
    boxes = []
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

                        # 得到缺陷的最小外接旋转矩形.
                        rect = cv2.minAreaRect(cnt)
                        # 得到旋转矩形的端点
                        box = cv2.boxPoints(rect)
                        box_d = np.int0(box)

                        # 满足面积条件的缺陷, 再计算缺陷的, 最远两个点间的距离. 可对长条形缺陷做筛选.
                        line_point = find_farthest_two_points(np.squeeze(cnt))
                        tmp = line_point[1] - line_point[0]
                        line = math.hypot(tmp[0], tmp[1])

                        # 统计各个子缺陷个数
                        if i != 1:
                            defects_nums[i] += 1
                            boxes.append(box_d)
                            temo_predict += temp * i
                        else:
                            # 黑白线另需满足长度阈值
                            if line >= yayin_limit:
                                print()
                                defects_nums[i] += 1
                                boxes.append(box_d)
                                temo_predict += temp * i

    return temo_predict, boxes, defects_nums



if __name__ == "__main__":

    root_path = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/0319工位3缺陷变化/model_psa_gw3_多分类'

    defcets = ['ok', 'heibaixian', 'bianxing', 'pengshang']
    # - ok
    # - heibaixian
    # - bianxing
    # - pengshang
    # 网络输出层数: 4


    # 需要开放给前端的参数
    # 1.压印长度>=20nm的才检出, 否则滤除
    yayin_limit = 20
    # 2. 压印, 碰伤条数阈值, 超过则ng. 和defcet_dict的keys意义对应
    ng_nums = [0, 3, 0, 3]
    # 3.置信度分数[list]: 可针对各个子缺陷设置不同的置信度阈值
    Confidence = [0.5] * len(defcets)
    # 4.面积过滤阈值[list]: 像素个数小于num_thres的不检出, 可针对各个子缺陷设置不同的面积阈值
    num_thres = [200, 210, 230, 50]  # 碰伤面积可能较小, 阈值设置小一些


    flag = {"0": "left", "1": "right"}
    test_dir = os.path.join(root_path, 'tests')
    test_paths = [os.path.join(test_dir, a) for a in os.listdir(test_dir)]
    res_dir = os.path.join(root_path, 'res_dir')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # roi边界冗余 横纵坐标起始点
    roi = [400, 700, 7712, 7378]
    w_, h_ = (roi[2] - roi[0]) // 2 , roi[3] - roi[1]
    # 输入模型的尺寸
    size = [2437, 4452]
    # 缩放比例
    resize_rate = h_ / size[1]
    # 等比缩放压印阈值长度
    yayin_limit /= resize_rate


    # 导入onnx
    onnx_path = os.path.join(root_path, '10000.onnx')
    onnx_session = ort.InferenceSession(onnx_path)
    # 模型的mean和std
    mean_ = [123.675, 116.28, 103.53]
    std_ = [58.395, 57.12, 57.375]

    for test_path in test_paths:

        im_name = os.path.basename(test_path)

        img = Image.open(test_path).convert('RGB')
        tmp = np.asarray(img)

        # 后续会将模型推理后的结果替换至此
        imgcopy = tmp.copy()

        # roi剔除图像边界冗余
        img = tmp[roi[0]:roi[2], roi[1]:roi[3]]
        h, w = img.shape[0], img.shape[1]
        img_left = img[:, :w//2]
        img_right = img[:, w//2:]

        roi_img_res = np.zeros((h, w, 3), dtype=np.uint8)
        for ind, img in enumerate([img_left, img_right]):
            img = cv2.resize(img, (size[0], size[1]))
            img_ = sdk_pre(img, mean_, std_)
            onnx_inputs     = {onnx_session.get_inputs()[0].name: img_.astype(np.float32)}
            onnx_predict = onnx_session.run(None, onnx_inputs)
            predict = softmax(onnx_predict[0], 1)
            map_, boxes, defects_nums = sdk_post(predict, defcets, Confidence=Confidence, num_thres=num_thres, yayin_limit=yayin_limit)
            mask_vis = label2colormap(map_)

            for k in range(1, len(defcets)):
                if defects_nums[k] >= ng_nums[k]:
                    print("{} ng!".format(defcets[k]))

            if boxes:
                # 绘制最小倾斜框
                for box in boxes:
                    cv2.drawContours(mask_vis, [box], 0, (0, 255, 0), 2)
                img_save = cv2.addWeighted(mask_vis, 0.7, img, 0.3, 10)
                img_save = cv2.resize(img_save, (w//2, h), interpolation=cv2.INTER_LINEAR)
                if ind == 0:
                    roi_img_res[:, :w//2, :] += img_save
                else:
                    roi_img_res[:, w//2:, :] += img_save
            else:
                print("{} has no defect".format(flag[str(ind)]))

        # roi_img_res替换至imgcopy
        imgcopy[roi[0]:roi[2], roi[1]:roi[3], :] = roi_img_res
        cv2.imwrite(os.path.join(res_dir, im_name), imgcopy)
