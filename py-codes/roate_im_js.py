# coding:utf-8

import cv2
import os
import numpy as np
import math
import copy
import json

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]

    rangle = np.deg2rad(angle) #angle in radians
    nw = (abs(np.sin(rangle)*h)+abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h)+abs(np.sin(rangle)*w))*scale

    rot_mat = cv2.getRotationMatrix2D((nw*0.5,nh*0.5), angle, scale)# rotate with center
    rot_move = np.dot(rot_mat,np.array([(nw-w)*0.5,(nh-h)*0.5,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src,rot_mat,(int(math.ceil(nw)),int(math.ceil(nh))),flags = cv2.INTER_LANCZOS4)

def _rotated_location(x1,y1,center_x,center_y,new_center_x,new_center_y,angle):
    x1 = float(x1) - center_x
    y1 = -(float(y1) - center_y)

    rangle = np.deg2rad(angle)
    rotated_x1 = np.cos(rangle)*x1 - np.sin(rangle)*y1
    rotated_y1 = np.cos(rangle)*y1 + np.sin(rangle)*x1

    final_x1 = rotated_x1 + new_center_x
    final_y1 = -rotated_y1 + new_center_y

    return final_x1, final_y1

def enhancement_using_rotation(ImagePath,AnnotationsPath,ImageSavePath,AnnotationsSavePath):

    #rotation
    for angle in rotation_angle:

        for imgfile in os.listdir(ImagePath):
            if not os.path.isfile(os.path.join(AnnotationsPath,imgfile[:-4]+'.json')):
                continue
            print(imgfile)
            img = cv2.imread(os.path.join(ImagePath,imgfile))

            h = img.shape[0]
            w = img.shape[1]

            new_data = {}

            Annotations = os.path.join(AnnotationsPath, imgfile[:-4]+'.json')

            new_name = 'r' + str(angle) + '_' + imgfile
            saveAnnotations = os.path.join(AnnotationsSavePath,new_name[:-4]+'.json')

            with open(Annotations, 'r') as f:
                data = json.load(f)

            imagePath = data['imagePath']
            new_data["imageData"] = None
            new_data['imagePath'] = 'r'+ str(angle) + '_' + imagePath
            new_data['version'] = data['version']
            new_data['flags'] = data['flags']
            new_data['imageHeight'] = h
            new_data['imageWidth'] = w

            new_shapes = []

            rotate_img = rotate_about_center(img,angle)
            cv2.imwrite(os.path.join(ImageSavePath,new_name),rotate_img)

            center_x = img.shape[1]/2
            center_y = img.shape[0]/2
            new_center_x = rotate_img.shape[1]/2
            new_center_y = rotate_img.shape[0]/2

            for shape in data['shapes']:
                new_info = {}

                new_info['label'] = shape['label']
                new_info['group_id'] = shape['group_id']
                new_info['shape_type'] = shape['shape_type']
                new_info['flags'] = shape['flags']

                points = shape['points']
                new_points = []

                for point in points:

                    temp_list = []
                    p_x = point[0]
                    p_y = point[1]

                    rotate_x1, rotate_y1 =  _rotated_location(p_x,p_y,center_x,center_y,new_center_x,new_center_y,angle)

                    temp_list.append(rotate_x1)
                    temp_list.append(rotate_y1)

                    if len(temp_list) > 0:
                        new_points.append(temp_list)

                #pts = np.array(new_points, np.int32)
                #pts = pts.reshape((-1,1,2))
                #new_rotate_img = cv2.polylines(rotate_img,[pts],True,(0,255,255))
                #cv2.imwrite(os.path.join(ImageSavePath,new_name),new_rotate_img)

                new_info['points'] = new_points
                new_shapes.append(new_info)

            new_data['shapes'] = new_shapes

            with open(saveAnnotations,"w") as savef:
                json.dump(new_data,savef, indent=4)


def main(dirpath, savepath):
    # savepath = './test-rotate270/'
    ImageSavePath = savepath
    AnnotationsSavePath = savepath

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(ImageSavePath):
        os.makedirs(ImageSavePath)
    if not os.path.exists(AnnotationsSavePath):
        os.makedirs(AnnotationsSavePath)

    #augument trainning data by rotating
    print('rotating...')
    ImagePath = dirpath + '/JPEG/'
    AnnotationsPath = dirpath + '/JSON/'
    enhancement_using_rotation(ImagePath,AnnotationsPath,ImageSavePath,AnnotationsSavePath)


# if __name__ == '__main__':
def roate_im_jss(dirpath, savepath):

    global annotation_file, rotation_angle
    annotation_file = 'json' #'txt',xml'
    rotation_angle=[90]
    # dirpath = '/Users/chenjia/Desktop/gw3'
    main(dirpath, savepath)
