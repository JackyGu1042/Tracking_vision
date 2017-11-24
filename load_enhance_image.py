import os
import cv2
import math
import glob
import json
import random
import numpy as np
import tensorflow as tf
from random import shuffle
from time import gmtime, strftime


from load_process import video
from load_process import frame
from load_process import BoundingBox
from load_process import annotation
from load_process import cropPadImage
from load_process import preprocess
from load_process import load_annotation_file


def data_shift(mission, img_path, json_path, lamda_shift,lamda_scale,min_scale,max_scale):
    json_file = open(json_path)
    str = json_file.read()
    data = json.loads(str,encoding="GB2312")

    x_value_curr = [data[u'plate_0'][u'PosPt0X'],data[u'plate_0'][u'PosPt1X'],data[u'plate_0'][u'PosPt2X'],data[u'plate_0'][u'PosPt3X']]
    y_value_curr = [data[u'plate_0'][u'PosPt0Y'],data[u'plate_0'][u'PosPt1Y'],data[u'plate_0'][u'PosPt2Y'],data[u'plate_0'][u'PosPt3Y']]

    x1 = (min(x_value_curr))/10#x bottom left
    y1 = (min(y_value_curr))/10#y bottom left
    x2 = (max(x_value_curr))/10#x top right
    y2 = (max(y_value_curr))/10#y top right

    print num_img, ' bbox:',x1,y1,x2,y2
    bbox = BoundingBox(x1,y1,x2,y2)

    image = cv2.imread(img_path)

    bbox_curr_gt = bbox
    bbox_curr_shift = BoundingBox(0, 0, 0, 0)
    bbox_curr_shift = bbox_curr_gt.shift(image, lamda_scale, lamda_shift, min_scale, max_scale, True, bbox_curr_shift)
    rand_search_region, rand_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, image)

    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
    bbox_gt_recentered.scale(rand_search_region)
    bbox_gt_scaled = bbox_gt_recentered

    bbox_curr_y = [bbox_gt_scaled.x1, bbox_gt_scaled.y1, bbox_gt_scaled.x2, bbox_gt_scaled.y2]

    cv2.imwrite('/home/huaxin/data_3T/license_dataset/license_track/Dataset_0/' + mission + '/image_unprocess_'+format(num_img, '05d')+'.jpg', rand_search_region)
    # train_y_bbox_curr.append(bbox_curr_y)
    return bbox_curr_y

def data_noshift(mission, img_path, json_path, lamda_shift,lamda_scale,min_scale,max_scale):
    json_file = open(json_path)
    str = json_file.read()
    data = json.loads(str,encoding="GB2312")

    x_value_curr = [data[u'plate_0'][u'PosPt0X'],data[u'plate_0'][u'PosPt1X'],data[u'plate_0'][u'PosPt2X'],data[u'plate_0'][u'PosPt3X']]
    y_value_curr = [data[u'plate_0'][u'PosPt0Y'],data[u'plate_0'][u'PosPt1Y'],data[u'plate_0'][u'PosPt2Y'],data[u'plate_0'][u'PosPt3Y']]

    x1 = (min(x_value_curr))/10#x bottom left
    y1 = (min(y_value_curr))/10#y bottom left
    x2 = (max(x_value_curr))/10#x top right
    y2 = (max(y_value_curr))/10#y top right

    print num_img, ' bbox:',x1,y1,x2,y2
    bbox = BoundingBox(x1,y1,x2,y2)

    image = cv2.imread(img_path)

    bbox_curr_gt = bbox
    rand_search_region, rand_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_gt, image)

    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
    bbox_gt_recentered.scale(rand_search_region)
    bbox_gt_scaled = bbox_gt_recentered

    bbox_curr_y = [bbox_gt_scaled.x1, bbox_gt_scaled.y1, bbox_gt_scaled.x2, bbox_gt_scaled.y2]

    cv2.imwrite('/home/huaxin/data_3T/license_dataset/license_track/Dataset_0/' + mission + '/image_unprocess_'+format(num_img, '05d')+'.jpg', rand_search_region)
    # train_y_bbox_curr.append(bbox_curr_y)
    return bbox_curr_y

width_resize = 64
height_resize = 64
channel_resize = 3

license_folder = './reasonable'

if not os.path.isdir(license_folder):
    print('{} is not a valid directory'.format(license_folder))

image_files = sorted(glob.glob(os.path.join(license_folder, '*.jpg')))
annotations_files = sorted(glob.glob(os.path.join(license_folder, '*.json')))

train_y_bbox_curr = []

#################################################################################
begin_index = 0
end_index = 5000

lamda_shift = 2
lamda_scale = 2
min_scale = 0.2
max_scale = 0.2

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_shift('train',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)

#################################################################################
begin_index = 5000
end_index = 10000

lamda_shift = 5
lamda_scale = 15
min_scale = 0.01
max_scale = 10

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_shift('train',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)

#################################################################################
begin_index = 10000
end_index = 15000

lamda_shift = 5
lamda_scale = 15
min_scale = 0.4
max_scale = 0.4

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_shift('train',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)
#################################################################
begin_index = 15000
end_index = 20000

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_noshift('train',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)

np.savetxt('/home/huaxin/data_3T/license_dataset/license_track/Dataset_0/train/bbox_y_gt', train_y_bbox_curr, delimiter=' ', newline="\n", fmt='%f')

#################################################################################
#################################################################################
begin_index = 20000
end_index = 23000

lamda_shift = 2
lamda_scale = 2
min_scale = 0.2
max_scale = 0.2

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_shift('evaluation',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)

#################################################################################
begin_index = 23000
end_index = 25000

lamda_shift = 5
lamda_scale = 15
min_scale = 0.01
max_scale = 10

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_shift('evaluation',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)

#################################################################################
begin_index = 25000
end_index = 27000

lamda_shift = 5
lamda_scale = 15
min_scale = 0.4
max_scale = 0.4

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_shift('evaluation',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)
#################################################################
begin_index = 27000
end_index = 29000

for num_img in range(begin_index,end_index):
    img_path = image_files[num_img]
    json_path = annotations_files[num_img]
    train_y_bbox = data_noshift('evaluation',img_path, json_path, lamda_shift, lamda_scale, min_scale, max_scale)
    train_y_bbox_curr.append(train_y_bbox)

np.savetxt('/home/huaxin/data_3T/license_dataset/license_track/Dataset_0/evaluation/bbox_y_gt', train_y_bbox_curr, delimiter=' ', newline="\n", fmt='%f')

#################################################################################
