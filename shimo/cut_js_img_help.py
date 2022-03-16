from PIL import Image
import os
import json
import copy
import shutil
import cv2
import numpy as np


Image.MAX_IMAGE_PIXELS = None # 解除最大图像尺寸限制

def crop_left_edge(points,x):
    # 处理左边切割
    n = len(points)
    feedback = []
    for i in range(n):
        s_point = points[i%n]
        e_point = points[(i+1)%n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[0]>=x and e_point[0]>=x:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[0]>=x and e_point[0]<x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x-e_point[0])/float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x,y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[0]<x and e_point[0]>=x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x-e_point[0])/float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x,y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback

def crop_right_edge(points, x):
    # 处理右边切割
    n = len(points)
    feedback = []
    for i in range(n):
        s_point = points[i % n]
        e_point = points[(i + 1) % n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[0] <= x and e_point[0] <= x:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[0] <= x and e_point[0] > x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x - e_point[0]) / float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x, y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[0] > x and e_point[0] <= x:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_y = float(x - e_point[0]) / float(delta_x)
            y = int(delta_y * percent_y + e_point[1])
            cross_point = [x, y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback

def crop_upper_edge(points, y):
    # 处理上边切割
    n = len(points)
    feedback = []

    for i in range(n):
        s_point = points[i % n]
        e_point = points[(i + 1) % n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[1] >= y and e_point[1] >= y:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[1] >= y and e_point[1] < y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[1] < y and e_point[1] >= y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback

def crop_lower_edge(points, y):
    # 处理下边切割
    n = len(points)
    feedback = []

    for i in range(n):
        s_point = points[i % n]
        e_point = points[(i + 1) % n]
        # 1. 两点都在外侧 全部丢弃，不做处理
        # 2. 两点都在内测，保留终止点
        if s_point[1] <= y and e_point[1] <= y:
            feedback.append(e_point)

        # 3. 起始点在内测，终止点在外侧
        if s_point[1] <= y and e_point[1] > y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)

        # 4. 起始点在外侧，终止点在内测
        if s_point[1] > y and e_point[1] <= y:
            delta_x = s_point[0] - e_point[0]
            delta_y = s_point[1] - e_point[1]
            percent_x = float(y - e_point[1]) / float(delta_y)
            x = int(delta_x * percent_x + e_point[0])
            cross_point = [x, y]
            feedback.append(cross_point)
            feedback.append(e_point)
    return feedback

def transfer_one_polygon(polygon, crop_area):
    points = copy.deepcopy(polygon)
    left, upper, right, lower = crop_area

    # 通过Sutherland-Hodgman算法 实现四边形裁剪任意多边形
    points = crop_left_edge(points,left)
    points = crop_right_edge(points,right)
    points = crop_upper_edge(points,upper)
    points = crop_lower_edge(points,lower)

    # 坐标偏移, 将基坐标转移到[0,0]点
    height_limit = lower - upper
    width_limit = right - left
    for index,point in enumerate(points):
        x = point[0]-left
        x = x if x>=0 else 0
        x = x if x<width_limit else width_limit -1
        y = point[1]-upper
        y = y if y>=0 else 0
        y = y if y<height_limit else height_limit - 1
        points[index] = [x, y]
    return points

def crop_area_with_seg_annotation(img,crop_area,cropped_img_name,anno_filename,anno_template):
    left, upper, right, lower = crop_area
    cropped = img.crop(crop_area)
    cropped.save(cropped_img_name)

    # 处理多边形标注
    if anno_template:
        cropped_annotation = copy.deepcopy(anno_template)
        shapes = cropped_annotation.get("shapes", [])
        new_shapes = []
        for shape in shapes:
            if shape.get("shape_type", None) in ["polygon", "linestrip"]:
                new_points = transfer_one_polygon(shape.get("points", []), crop_area)
                if new_points:
                    for index, point in enumerate(new_points):
                        new_points[index] = [point[0], point[1]]
                    shape["points"] = new_points
                    new_shapes.append(shape)
        if new_shapes:
            cropped_annotation["shapes"] = new_shapes
            cropped_annotation["imagePath"] = os.path.split(cropped_img_name)[1]
            cropped_annotation["imageData"] = None
            cropped_annotation["imageHeight"] = int(lower - upper)
            cropped_annotation["imageWidth"] = int(right - left)
            with open(anno_filename, "w", encoding="utf-8") as writer:
                json.dump(cropped_annotation, writer)


def split_image(json_path, path_image, split_target, out_dir=None, roi=None):
    # split_image(json_path,im_path,split_target,out_dir,roi)
    img = Image.open(path_image)
    if roi:
        # [0, 918, 7594, 7594]
        tmp = np.asarray(img)
        tmp_img = tmp[roi[0]:roi[2], roi[1]:roi[3]]
        img = Image.fromarray(tmp_img)
        w, h = img.size

    target_w, target_h = split_target
    step_w = w/target_w
    step_h = h/target_h # 3338, 7474

    img_name = os.path.basename(path_image)
    pre_name, ext = os.path.splitext(img_name)

    anno_template = None
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as reader:
            anno_template = json.load(reader)

    for i in range(target_w):
        for j in range(target_h):
            # 处理输出图像路径&名称
            if not out_dir:
                pre, ext = os.path.splitext(path_image)
            else:
                image_name_ext = os.path.split(path_image)[1]
                image_name = os.path.splitext(image_name_ext)[0]
                pre = os.path.join(out_dir,image_name)
            cropped_img_name = "{}_{}_{}_{}".format(pre,i,j,ext)
            cropped_json_name = '{}/{}_{}_{}_.json'.format(out_dir, os.path.basename(json_path).split('.')[0], i, j)
            # 处理子图裁剪
            left = step_w * i
            upper = step_h * j
            right = step_w * (i+1)
            lower = step_h * (j+1)

            if right > w:
                right = w
            if lower > h:
                lower = h

            crop_area = (left, upper, right, lower) # (left, upper, right, lower)
            crop_area_with_seg_annotation(img,crop_area,cropped_img_name,cropped_json_name,anno_template)


def split_img_dir(im_js_lines, split_target,out_dir=None,roi=None):
    for line in im_js_lines:
        im_path, json_path = '/data{}'.format(line[:-1].split("||")[0]), '/data{}'.format(line[:-1].split("||")[1])
        im_path, json_path = '{}'.format(line[:-1].split("||")[0]), '{}'.format(line[:-1].split("||")[1])
        split_image(json_path,im_path,split_target,out_dir,roi)


# train, res_im_js_path, split_target, roi=roi
def help(im_js_lines, out_dir, split_target, roi=None):
    split_img_dir(im_js_lines, split_target, out_dir=out_dir, roi=roi)


if __name__ == "__main__":
    line = ['/Users/chenjia/Desktop/111/8.bmp||/Users/chenjia/Desktop/111/8.json']
    split_target = (1, 2)
    roi = [0, 918, 7594, 7594]
    out_dir = '/Users/chenjia/Desktop/222'
    split_img_dir(line, split_target, out_dir=out_dir, roi=roi)

