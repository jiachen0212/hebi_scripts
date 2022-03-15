"""
FILENAME:        locate.py

AUTHORS:         MoYu

START DATE:      2022.02.16

CONTACT:         yu.mo@smartmore.com

Description:
"""
import os.path
import os
import cv2
import json
import numpy as np
# import tensorboard.plugins.projector.metadata

np.set_printoptions(suppress=True)


class LineCalliper(object):
    def __init__(self, **kwargs):
        self.start = kwargs.get("start", (0, 0))
        self.end = kwargs.get("end", (0, 0))
        self.x1, self.y1 = self.start
        self.x2, self.y2 = self.end
        self.num = kwargs.get("num", 2)
        self.unit_step_x, self.unit_step_y = (self.x2 - self.x1) / (self.num - 1), (self.y2 - self.y1) / (self.num - 1)
        self.axis = kwargs.get("axis", True)
        if self.axis:
            self.norm_step_x, self.norm_step_y = self.unit_step_y, -self.unit_step_x
        else:
            self.norm_step_x, self.norm_step_y = -self.unit_step_y, self.unit_step_x
        norm_length = np.sqrt(self.norm_step_x ** 2 + self.norm_step_y ** 2)
        self.norm_step_x = self.norm_step_x / norm_length
        self.norm_step_y = self.norm_step_y / norm_length
        self.axis_half_length = kwargs.get("axis_half_length", 50)
        self.axis_half_width = kwargs.get("axis_half_width", 2)

    def get_line_values(self, image: np.ndarray, idx: int):
        values = []
        cx, cy = self.x1 + self.unit_step_x * idx, self.y1 + self.unit_step_y * idx
        # draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(-self.axis_half_length, self.axis_half_length + 1):
            cpx, cpy = cx + self.norm_step_x * i, cy + self.norm_step_y * i
            # ipx, ipy = int(round(cpx)), int(round(cpy))
            # draw[ipy-1:ipy+2, ipx-1:ipx+2] = (0, 255, 0)
            value = 0
            for j in range(-self.axis_half_width, self.axis_half_width + 1):
                px, py = cpx + self.unit_step_x * j, cpy + self.unit_step_y * j
                rpx, rpy = int(round(px)), int(round(py))
                if rpx < 0 or rpx >= image.shape[1] or rpy < 0 or rpy >= image.shape[0]:
                    continue
                value += image[rpy, rpx]

            value /= (2 * self.axis_half_width + 1)
            values.append(value)

        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return np.array(values)


class CalliperRule(object):
    def __init__(self, **kwargs):
        self.edge_type = kwargs.get("edge_type", "all")
        self.select_type = kwargs.get("select_type", "max")
        self.threshold = kwargs.get("threshold", 10)

    def get_line_edge(self, line_values: np.ndarray):
        diff = line_values[1:] - line_values[:-1]
        if self.edge_type == "all":
            diff = np.abs(diff)
        elif self.edge_type == "black2white":
            pass
        elif self.edge_type == "white2black":
            diff = -diff
        else:
            raise RuntimeError("Unknown edge_type")

        if self.select_type == "first":
            xs = np.where(diff > self.threshold)[0]
            if len(xs) > 0:
                return xs[0]
        elif self.select_type == "last":
            xs = np.where(diff > self.threshold)[0]
            if len(xs) > 0:
                return xs[-1]
        elif self.select_type == "max":
            xs = np.where(diff > self.threshold)[0]
            if len(xs) > 0:
                return np.where(diff == diff.max())[0][0]

        return None


class LineABC(object):
    def __init__(self, **kwargs):
        self.a = kwargs.get("a", None)
        self.b = kwargs.get("b", None)
        self.c = kwargs.get("c", None)


def find_line_by_calliper(image: np.ndarray, calliper: LineCalliper, rule: CalliperRule):
    points = []
    for i in range(0, calliper.num):
        values = calliper.get_line_values(image, i)
        # print(np.max(values[1:] - values[:-1]))
        idx = rule.get_line_edge(values)
        if idx is None:
            continue

        px = calliper.x1 + calliper.unit_step_x * i + calliper.norm_step_x * (idx - calliper.axis_half_length)
        py = calliper.y1 + calliper.unit_step_y * i + calliper.norm_step_y * (idx - calliper.axis_half_length)

        points.append((px, py))

    print("points:", len(points))

    if len(points) < 10:
        return None

    # draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for px, py in points:
    #     ipx, ipy = int(round(px)), int(round(py))
    #     draw[ipy-1:ipy+2, ipx-1:ipx+2] = (0, 255, 0)

    # cv2.imshow("draw", draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    points = np.array(points)
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 1e-2, 1e-2).reshape(-1)
    a = vy
    b = -vx
    c = vx * y - vy * x

    line = LineABC(a=a, b=b, c=c)
    return line


def intersect(line1: LineABC, line2: LineABC):
    a1, b1, c1 = line1.a, line1.b, line1.c
    a2, b2, c2 = line2.a, line2.b, line2.c

    # a1 * x + b1 * y + c1 = 0
    # a2 * x + b2 * y + c2 = 0
    m = np.array([
        [a1, b1],
        [a2, b2]
    ])
    y = np.array([
        [-c1],
        [-c2]
    ])
    x, y = np.linalg.inv(m).dot(y).reshape(-1)
    return x, y


def perspective_transform(p: np.ndarray, m: np.ndarray):
    x, y = p
    u = (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / (m[2, 0] * x + m[2, 1] * y + m[2, 2])
    v = (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / (m[2, 0] * x + m[2, 1] * y + m[2, 2])
    return np.array([u, v])


def train1(image: np.ndarray, conf: dict):
    c1 = LineCalliper(
        start=(1500, 1200),
        end=(3500, 1200),
        num=101,
        axis=True,
        axis_half_length=100,
        axis_half_width=10
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(4850, 1600),
        end=(4850, 2800),
        num=61,
        axis=True,
        axis_half_length=300,
        axis_half_width=10
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(3500, 3200),
        end=(1500, 3200),
        num=101,
        axis=True,
        axis_half_length=100,
        axis_half_width=10
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(530, 2800),
        end=(550, 1600),
        num=61,
        axis=True,
        axis_half_length=300,
        axis_half_width=10
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)

    points = np.array([p1, p2, p3, p4])
    mask_points = []
    for conf in conf["shapes"]:
        mask_ps = conf["points"]
        mask_ps = np.array(mask_ps)
        mask_points.append(mask_ps)

    return points, mask_points


def inference1(image: np.ndarray, train_points: np.ndarray, train_mask_points: list):
    c1 = LineCalliper(
        start=(1700, 1200),
        end=(3300, 1200),
        num=161,
        axis=True,
        axis_half_length=100,
        axis_half_width=5
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=10
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(4850, 1600),
        end=(4850, 2800),
        num=121,
        axis=True,
        axis_half_length=300,
        axis_half_width=5
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=10
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(3300, 3200),
        end=(1700, 3200),
        num=161,
        axis=True,
        axis_half_length=100,
        axis_half_width=5
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=10
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(530, 2800),
        end=(550, 1600),
        num=121,
        axis=True,
        axis_half_length=300,
        axis_half_width=5
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=10
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)
    points = np.array([p1, p2, p3, p4])
    m = cv2.getPerspectiveTransform(train_points, points)

    inference_mask_points = []
    for ps in train_mask_points:
        inference_mask_ps = []
        for p in ps:
            new_point = perspective_transform(p, m)
            inference_mask_ps.append(new_point)
        inference_mask_ps = np.array(inference_mask_ps)
        inference_mask_points.append(inference_mask_ps)

    return points, inference_mask_points


def main1(data_dir, config, ff):
    # 工位1的定位标注
    path = os.path.join(config, "1.png")
    train_image = cv2.imread(path, 0)
    conf_path = os.path.join(config, "1.json")
    with open(conf_path) as f:
        train_conf = json.load(f)

    # 定位+mask工位1的图像
    points, mask_points = train1(image=train_image, conf=train_conf)
    names = os.listdir(data_dir)
    paths = [os.path.join(data_dir, a) for a in names if '.png' in a]
    # paths = glob.glob(r"D:\work\project\DL\hebi\data\20220214\data\PSA\1cls\bianxing\1\*.png")
    for cur_path in paths:
        print(cur_path)
        pre, post = cur_path.split('PSA')[0], cur_path.split('PSA')[1]
        save_path = "{}masked{}".format(pre, post)
        draw = cv2.imread(cur_path)

        try:
            image = cv2.imread(cur_path, 0)
        except:
            print("image fail: {}".format(cur_path))
            continue
        try:
            inference_points, inference_mask_points = inference1(image, points, mask_points)
        except:
            print("mask fail: {}".format(cur_path))
            ff.write(cur_path)
            cv2.imwrite(save_path, image)
            save_pre_dir = os.path.dirname(save_path)
            if not os.path.exists(save_pre_dir):
                os.makedirs(save_pre_dir)
            continue
        # 画出外框绿线
        # cv2.drawContours(draw, [inference_points.astype(np.int32)], 0, [0, 255, 0], 2)
        for ps in inference_mask_points:
            ps = ps.astype(np.int32)
            cv2.drawContours(draw, [ps], 0, [0, 0, 0], -1)

        # 可视化mask结果
        # total_img = np.zeros((image.shape[0] // 8, image.shape[1] // 8, 3), np.uint8)
        # total_img[:, :, :] = cv2.resize(draw, (draw.shape[1] // 8, draw.shape[0] // 8))
        # cv2.imshow("draw1", total_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # save masked_image
        save_pre_dir = os.path.dirname(save_path)
        if not os.path.exists(save_pre_dir):
            os.makedirs(save_pre_dir)
        cv2.imwrite(save_path, draw)


def inference2(image: np.ndarray, train_points: np.ndarray, train_mask_points: list):
    c1 = LineCalliper(
        start=(1000, 650),
        end=(4000, 650),
        num=151,
        axis=True,
        axis_half_length=100,
        axis_half_width=10
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(4700, 1000),
        end=(4700, 2200),
        num=61,
        axis=True,
        axis_half_length=300,
        axis_half_width=10
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(4000, 2650),
        end=(1000, 2650),
        num=151,
        axis=True,
        axis_half_length=100,
        axis_half_width=10
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(300, 2200),
        end=(300, 1000),
        num=61,
        axis=True,
        axis_half_length=300,
        axis_half_width=10
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)
    points = np.array([p1, p2, p3, p4])
    m = cv2.getPerspectiveTransform(train_points, points)

    inference_mask_points = []
    for ps in train_mask_points:
        inference_mask_ps = []
        for p in ps:
            new_point = perspective_transform(p, m)
            inference_mask_ps.append(new_point)
        inference_mask_ps = np.array(inference_mask_ps)
        inference_mask_points.append(inference_mask_ps)

    return points, inference_mask_points


def main2(data_dir, config, ff):
    path = os.path.join(config, "1.png")
    train_image = cv2.imread(path, 0)
    conf_path = os.path.join(config, "1.json")
    with open(conf_path) as f:
        train_conf = json.load(f)

    points, mask_points = train1(image=train_image, conf=train_conf)
    names = os.listdir(data_dir)
    paths = [os.path.join(data_dir, a) for a in names if '.png' in a]
    for cur_path in paths:
        print(cur_path)
        pre, post = cur_path.split('PSA')[0], cur_path.split('PSA')[1]
        save_path = "{}masked{}".format(pre, post)
        draw = cv2.imread(cur_path)
        try:
            image = cv2.imread(cur_path, 0)
        except:
            print("image fail: {}".format(cur_path))
            continue
        try:
            inference_points, inference_mask_points = inference2(image, points, mask_points)
        except:
            print("mask fail: {}".format(cur_path))
            ff.write(cur_path)
            save_pre_dir = os.path.dirname(save_path)
            if not os.path.exists(save_pre_dir):
                os.makedirs(save_pre_dir)
            cv2.imwrite(save_path, image)
            continue

        # cv2.drawContours(draw, [inference_points.astype(np.int32)], 0, [0, 255, 0], 2)
        for ps in inference_mask_points:
            ps = ps.astype(np.int32)
            cv2.drawContours(draw, [ps], 0, [0, 0, 0], -1)
        # save masked_img
        save_pre_dir = os.path.dirname(save_path)
        if not os.path.exists(save_pre_dir):
            os.makedirs(save_pre_dir)
        cv2.imwrite(save_path, draw)


def train3(image: np.ndarray, conf: dict):
    c1 = LineCalliper(
        start=(1500, 1000),
        end=(3000, 1000),
        num=76,
        axis=True,
        axis_half_length=400,
        axis_half_width=2
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(3600, 1600),
        end=(3600, 5600),
        num=401,
        axis=True,
        axis_half_length=300,
        axis_half_width=2
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(3000, 7000),
        end=(1500, 7000),
        num=76,
        axis=True,
        axis_half_length=400,
        axis_half_width=2
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(800, 5800),
        end=(800, 1800),
        num=401,
        axis=True,
        axis_half_length=300,
        axis_half_width=2
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)

    points = np.array([p1, p2, p3, p4])
    mask_points = []
    for conf in conf["shapes"]:
        mask_ps = conf["points"]
        mask_ps = np.array(mask_ps)
        mask_points.append(mask_ps)

    return points, mask_points


def inference31(image: np.ndarray, train_points: np.ndarray, train_mask_points: list):
    c1 = LineCalliper(
        start=(1500, 1000),
        end=(3000, 1000),
        num=76,
        axis=True,
        axis_half_length=400,
        axis_half_width=2
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(3600, 1600),
        end=(3600, 5600),
        num=401,
        axis=True,
        axis_half_length=300,
        axis_half_width=2
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(3000, 7000),
        end=(1500, 7000),
        num=76,
        axis=True,
        axis_half_length=400,
        axis_half_width=2
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(800, 5800),
        end=(800, 1800),
        num=401,
        axis=True,
        axis_half_length=300,
        axis_half_width=2
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=5
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)
    points = np.array([p1, p2, p3, p4])
    m = cv2.getPerspectiveTransform(train_points, points)

    inference_mask_points = []
    for ps in train_mask_points:
        inference_mask_ps = []
        for p in ps:
            new_point = perspective_transform(p, m)
            inference_mask_ps.append(new_point)
        inference_mask_ps = np.array(inference_mask_ps)
        inference_mask_points.append(inference_mask_ps)

    return points, inference_mask_points


def inference32(image: np.ndarray, train_points: np.ndarray, train_mask_points: list):
    c1 = LineCalliper(
        start=(5300, 1000),
        end=(6800, 1000),
        num=76,
        axis=True,
        axis_half_length=400,
        axis_half_width=2
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=15
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(7400, 1600),
        end=(7400, 5600),
        num=401,
        axis=True,
        axis_half_length=300,
        axis_half_width=2
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=15
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(6800, 7000),
        end=(5300, 7000),
        num=76,
        axis=True,
        axis_half_length=400,
        axis_half_width=2
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=15
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(4600, 5800),
        end=(4600, 1800),
        num=401,
        axis=True,
        axis_half_length=300,
        axis_half_width=2
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=15
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)
    points = np.array([p1, p2, p3, p4])
    m = cv2.getPerspectiveTransform(train_points, points)

    inference_mask_points = []
    for ps in train_mask_points:
        inference_mask_ps = []
        for p in ps:
            new_point = perspective_transform(p, m)
            inference_mask_ps.append(new_point)
        inference_mask_ps = np.array(inference_mask_ps)
        inference_mask_points.append(inference_mask_ps)

    return points, inference_mask_points


def main3(data_dir, config, ff):
    path = os.path.join(config, "2.png")
    train_image = cv2.imread(path, 0)
    conf_path = os.path.join(config, "2.json")
    with open(conf_path) as f:
        train_conf = json.load(f)

    points, mask_points = train3(image=train_image, conf=train_conf)
    names = os.listdir(data_dir)
    paths = [os.path.join(data_dir, a) for a in names if '.png' in a]
    for cur_path in paths:
        print(cur_path)
        pre, post = cur_path.split('PSA')[0], cur_path.split('PSA')[1]
        save_path = "{}masked{}".format(pre, post)
        draw = cv2.imread(cur_path)
        try:
            image = cv2.imread(cur_path, 0)
        except:
            print("image fail: {}".format(cur_path))
            continue
        try:
            inference_points1, inference_mask_points1 = inference31(image, points, mask_points)
            inference_points2, inference_mask_points2 = inference32(image, points, mask_points)
        except:
            print("mask fail: {}".format(cur_path))
            ff.write(cur_path)
            save_pre_dir = os.path.dirname(save_path)
            if not os.path.exists(save_pre_dir):
                os.makedirs(save_pre_dir)
            cv2.imwrite(save_path, image)
            continue

        # cv2.drawContours(draw, [inference_points1.astype(np.int32)], 0, [0, 255, 0], 2)
        # cv2.drawContours(draw, [inference_points2.astype(np.int32)], 0, [0, 255, 0], 2)
        for ps in inference_mask_points1:
            ps = ps.astype(np.int32)
            cv2.drawContours(draw, [ps], 0, [0, 0, 0], -1)

        for ps in inference_mask_points2:
            ps = ps.astype(np.int32)
            cv2.drawContours(draw, [ps], 0, [0, 0, 0], -1)

        # save masked_img
        save_pre_dir = os.path.dirname(save_path)
        if not os.path.exists(save_pre_dir):
            os.makedirs(save_pre_dir)
        cv2.imwrite(save_path, draw)


def train4(image: np.ndarray, conf: dict):
    c1 = LineCalliper(
        start=(2000, 800),
        end=(6000, 800),
        num=101,
        axis=True,
        axis_half_length=400,
        axis_half_width=3
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(7000, 1500),
        end=(7000, 3000),
        num=51,
        axis=True,
        axis_half_length=300,
        axis_half_width=3
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(6000, 3600),
        end=(2000, 3600),
        num=101,
        axis=True,
        axis_half_length=400,
        axis_half_width=3
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(1000, 3000),
        end=(1000, 1500),
        num=51,
        axis=True,
        axis_half_length=300,
        axis_half_width=3
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)

    points = np.array([p1, p2, p3, p4])
    mask_points = []
    for conf in conf["shapes"]:
        mask_ps = conf["points"]
        mask_ps = np.array(mask_ps)
        mask_points.append(mask_ps)

    return points, mask_points


def inference41(image: np.ndarray, train_points: np.ndarray, train_mask_points: list):
    c1 = LineCalliper(
        start=(2000, 800),
        end=(6000, 800),
        num=101,
        axis=True,
        axis_half_length=400,
        axis_half_width=3
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(7000, 1500),
        end=(7000, 3000),
        num=51,
        axis=True,
        axis_half_length=300,
        axis_half_width=3
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(6000, 3600),
        end=(2000, 3600),
        num=101,
        axis=True,
        axis_half_length=400,
        axis_half_width=3
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(1000, 3000),
        end=(1000, 1500),
        num=51,
        axis=True,
        axis_half_length=300,
        axis_half_width=3
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)
    points = np.array([p1, p2, p3, p4])
    m = cv2.getPerspectiveTransform(train_points, points)

    inference_mask_points = []
    for ps in train_mask_points:
        inference_mask_ps = []
        for p in ps:
            new_point = perspective_transform(p, m)
            inference_mask_ps.append(new_point)
        inference_mask_ps = np.array(inference_mask_ps)
        inference_mask_points.append(inference_mask_ps)

    return points, inference_mask_points


def inference42(image: np.ndarray, train_points: np.ndarray, train_mask_points: list):
    c1 = LineCalliper(
        start=(2000, 4800),
        end=(6000, 4800),
        num=101,
        axis=True,
        axis_half_length=400,
        axis_half_width=3
    )
    r1 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line1 = find_line_by_calliper(image, c1, r1)

    c2 = LineCalliper(
        start=(7000, 5500),
        end=(7000, 7000),
        num=51,
        axis=True,
        axis_half_length=300,
        axis_half_width=3
    )
    r2 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line2 = find_line_by_calliper(image, c2, r2)

    c3 = LineCalliper(
        start=(6000, 7600),
        end=(2000, 7600),
        num=101,
        axis=True,
        axis_half_length=400,
        axis_half_width=3
    )
    r3 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line3 = find_line_by_calliper(image, c3, r3)

    c4 = LineCalliper(
        start=(1000, 7000),
        end=(1000, 5500),
        num=51,
        axis=True,
        axis_half_length=300,
        axis_half_width=3
    )
    r4 = CalliperRule(
        edge_type="white2black",
        select_type="last",
        threshold=20
    )
    line4 = find_line_by_calliper(image, c4, r4)

    p1 = intersect(line4, line1)
    p2 = intersect(line1, line2)
    p3 = intersect(line2, line3)
    p4 = intersect(line3, line4)
    points = np.array([p1, p2, p3, p4])
    m = cv2.getPerspectiveTransform(train_points, points)

    inference_mask_points = []
    for ps in train_mask_points:
        inference_mask_ps = []
        for p in ps:
            new_point = perspective_transform(p, m)
            inference_mask_ps.append(new_point)
        inference_mask_ps = np.array(inference_mask_ps)
        inference_mask_points.append(inference_mask_ps)

    return points, inference_mask_points


def main4(data_dir, config, ff):
    path = os.path.join(config, "3.png")
    train_image = cv2.imread(path, 0)
    conf_path = os.path.join(config, "3.json")
    with open(conf_path) as f:
        train_conf = json.load(f)

    points, mask_points = train4(image=train_image, conf=train_conf)
    names = os.listdir(data_dir)
    paths = [os.path.join(data_dir, a) for a in names if '.png' in a]
    for cur_path in paths:
        print(cur_path)
        pre, post = cur_path.split('PSA')[0], cur_path.split('PSA')[1]
        save_path = "{}masked{}".format(pre, post)
        draw = cv2.imread(cur_path)
        try:
            image = cv2.imread(cur_path, 0)
        except:
            print("image fail: {}".format(cur_path))
            continue
        try:
            inference_points1, inference_mask_points1 = inference41(image, points, mask_points)
            inference_points2, inference_mask_points2 = inference42(image, points, mask_points)
        except:
            print("mask fail: {}".format(cur_path))
            ff.write(cur_path)
            save_pre_dir = os.path.dirname(save_path)
            if not os.path.exists(save_pre_dir):
                os.makedirs(save_pre_dir)
            cv2.imwrite(save_path, image)
            continue
        # cv2.drawContours(draw, [inference_points1.astype(np.int32)], 0, [0, 255, 0], 2)
        # cv2.drawContours(draw, [inference_points2.astype(np.int32)], 0, [0, 255, 0], 2)
        for ps in inference_mask_points1:
            ps = ps.astype(np.int32)
            cv2.drawContours(draw, [ps], 0, [0, 0, 0], -1)

        for ps in inference_mask_points2:
            ps = ps.astype(np.int32)
            cv2.drawContours(draw, [ps], 0, [0, 0, 0], -1)

        # save masked_img
        save_pre_dir = os.path.dirname(save_path)
        if not os.path.exists(save_pre_dir):
            os.makedirs(save_pre_dir)
        cv2.imwrite(save_path, draw)


if __name__ == '__main__':

    datas = ['0214']
    config_dir = '/data/home/jiachen/data/seg_data/hebi/20220214/imgs/hebi_label'

    for data in datas:
        base_dir = '/data/home/jiachen/data/seg_data/hebi/2022{}/imgs'.format(data)
        config_dir = '/data/home/jiachen/data/seg_data/hebi/20220214/imgs/hebi_label'
        # base_dir = r'D:\work\project\DL\hebi\data\20220214\data'
        # config_dir = r'C:\Users\15974\Desktop\hebi_label'

        defects = {'PSA': ['2cls', '1cls', '3cls']}
        cls_dirs = defects['PSA']
        gongwei_paths = dict()
        for cls in cls_dirs:
            gongwei_paths[cls] = []
        for defect in defects:
            for cls in defects[defect]:
                path1 = os.path.join(base_dir, defect, cls)
                for sub in os.listdir(path1):
                    path2 = os.path.join(path1, sub)
                    for dir_ in os.listdir(path2):
                        path3 = os.path.join(path2, dir_)
                        if os.path.basename(path3) in ['1', '2']:
                            gongwei_paths['1cls'].append(path3)
                        elif os.path.basename(path3) == '3':
                            gongwei_paths['2cls'].append(path3)
                        else:
                            gongwei_paths['3cls'].append(path3)

        ff1 = open('./cls1.txt', 'w')
        ff2 = open('./cls2.txt', 'w')
        ff3 = open('./cls3.txt', 'w')
        ff4 = open('./cls4.txt', 'w')
        for k, v in gongwei_paths.items():
            if k == '1cls':
                config = os.path.join(config_dir, '1')
                for dir_ in v:
                    if os.path.basename(dir_) == '2':
                        main2(dir_, config, ff2)
                    else:
                        main1(dir_, config, ff1)
            elif k == '2cls':
                config = os.path.join(config_dir, '2')
                for dir_ in v:
                    main3(dir_, config, ff3)
            else:
                config = os.path.join(config_dir, '3')
                for dir_ in v:
                    main4(dir_, config, ff4)
