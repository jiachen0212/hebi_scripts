import cv2
import os

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize[1]):
        for x in range(0, image.shape[1], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# 返回滑动窗结果集合，本示例暂时未用到
def get_slice(image, stepSize, windowSize):
    slice_sets = []
    for (x, y, window) in sliding_window(image, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
            continue
        slice = image[y:y + windowSize[1], x:x + windowSize[0]]
        slice_sets.append(slice)
    return slice_sets

if __name__ == '__main__':

    file_name = '/Users/chenjia/Downloads/Learning/SmartMore/2022/DL/hebi/1.png'

    base_name = os.path.basename(file_name)
    image = cv2.imread(file_name)

    w = image.shape[1]
    h = image.shape[0]

    # 先右下 padding 再切割
    (winW, winH) = 2048, 2048
    stepSize = 1200, 600
    cnt = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        slice_img = image[y:y+winH,x:x+winW]
        # cv2.imwrite("{}_{}_{}.png", slice_img)