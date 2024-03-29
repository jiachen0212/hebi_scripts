from os import remove
import cv2
import numpy as np

'''
@pred 模型的预测图，图中每个像素代表着对应的类别，其中0为背景，1...n为其他缺陷。
@threshold_LengthCount 模型的长度阈值，list
'''


def removeShortDefect(pred, threshold_LengthCount, rotate_rect=False, filter_width=False):
    result_mask = np.ones(pred.shape)
    for each_label in range(len(threshold_LengthCount)):
        if each_label == 0:
            continue

        mask = np.array(pred == each_label, np.uint8)
        _, labels = cv2.connectedComponents(mask, 8)

        for label in np.unique(labels):
            if label == 0:
                continue
            label_mask = labels == label
            contours, _ = cv2.findContours(label_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            assert len(contours) == 1
            if not rotate_rect:
                x, y, w, h = cv2.boundingRect(contours[0])
                print(h,w)
                target_size = min(w,h) if filter_width else max(w,h)
                if target_size < threshold_LengthCount[each_label]:
                    result_mask[label_mask] = 0
            else:
                center, size, angle = cv2.minAreaRect(contours[0])
                w,h = size
                print(h,w)
                target_size = min(w,h) if filter_width else max(w,h)
                if target_size < threshold_LengthCount[each_label]:
                    result_mask[label_mask] = 0
    pred = pred * result_mask
    return pred


if __name__ == '__main__':
    DebugImg = np.zeros((80, 80), dtype=np.uint8)
    cv2.line(DebugImg, (5, 5), (15, 15), 1, 2)
    # cv2.rectangle(DebugImg, (20, 20), (50, 50), 1, 1)
    cv2.imwrite(r'D:\Work\codes\utils\postprocess\remove_short_defects_before.jpg',
                255 * DebugImg)

    # Demo 1: 按照 正外接矩形 长度 进行筛选
    # 线条坐标轴上长度为13 会被过滤掉
    DebugImgAfterPostPro = removeShortDefect(DebugImg, [0, 14], False, False)
    cv2.imwrite(r'D:\Work\codes\utils\postprocess\demo1.jpg',
                255 * DebugImgAfterPostPro)

    # Demo 2: 按照 正外接矩形 宽度 进行筛选
    # 线条会因为宽度13达标而被保留
    DebugImgAfterPostPro = removeShortDefect(DebugImg, [0, 9], False, True)
    cv2.imwrite(r'D:\Work\codes\utils\postprocess\demo2.jpg',
                255 * DebugImgAfterPostPro)

    # Demo 3.1: 按照 旋转外接矩形 长度 进行筛选
    # 线条会被保留 宽度约为3 长度大约15
    DebugImgAfterPostPro = removeShortDefect(DebugImg, [0, 14], True, False)
    cv2.imwrite(r'D:\Work\codes\utils\postprocess\demo3_1.jpg',
                255 * DebugImgAfterPostPro)

    # Demo 3.2: 按照 旋转外接矩形 长度 进行筛选
    # 线条会被筛选掉  14会被保留
    DebugImgAfterPostPro = removeShortDefect(DebugImg, [0, 19], True, False)
    cv2.imwrite(r'D:\Work\codes\utils\postprocess\demo3_2.jpg',
                255 * DebugImgAfterPostPro)

    # Demo 4: 按照 旋转外接矩形 宽度 进行筛选
    # 线条会被筛选掉  实际宽度为3
    DebugImgAfterPostPro = removeShortDefect(DebugImg, [0, 9], True, True)
    cv2.imwrite(r'D:\Work\codes\utils\postprocess\demo4.jpg',
                255 * DebugImgAfterPostPro)

    # 当然也可以扩展长宽比啥的...
    # 对于赫比，输入尺寸约束[bg, baimo_qipao, shimo_posun, fenqiexian_qipao, keli, lan_canliu]
    # 可以开放上述的 baimo_qipao fenqiexian_qipao