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

lamda_shift = 1
lamda_scale = 1
min_scale = 0.2
max_scale = 3

width_resize = 64
height_resize = 64
channel_resize = 3

license_folder = './reasonable'

if not os.path.isdir(license_folder):
    print('{} is not a valid directory'.format(license_folder))

image_files = sorted(glob.glob(os.path.join(license_folder, '*.jpg')))
print image_files[0]

annotations_files = sorted(glob.glob(os.path.join(license_folder, '*.json')))
print annotations_files[0]

batch_x_curr = []
batch_y_curr = []

for num_img in range(0,10000):
    lamda_shift = 2
    lamda_scale = 2
    min_scale = 0.2
    max_scale = 0.2

    img_path = image_files[num_img]
    image = cv2.imread(img_path)

    json_path = annotations_files[num_img]
    json_file = open(json_path)
    str = json_file.read()
    data = json.loads(str,encoding="GB2312")

    x_value_curr = [data[u'plate_0'][u'PosPt0X'],data[u'plate_0'][u'PosPt1X'],data[u'plate_0'][u'PosPt2X'],data[u'plate_0'][u'PosPt3X']]
    y_value_curr = [data[u'plate_0'][u'PosPt0Y'],data[u'plate_0'][u'PosPt1Y'],data[u'plate_0'][u'PosPt2Y'],data[u'plate_0'][u'PosPt3Y']]

    x1 = (min(x_value_curr))/10#x bottom left
    y1 = (min(y_value_curr))/10#y bottom left
    x2 = (max(x_value_curr))/10#x top right
    y2 = (max(y_value_curr))/10#y top right

    print num_img,'bbox:',x1,y1,x2,y2
    bbox = BoundingBox(x1,y1,x2,y2)

    bbox_curr_gt = bbox
    bbox_curr_shift = BoundingBox(0, 0, 0, 0)
    bbox_curr_shift = bbox_curr_gt.shift(image, lamda_scale, lamda_shift, min_scale, max_scale, True, bbox_curr_shift)
    rand_search_region, rand_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, image)

    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
    bbox_gt_recentered.scale(rand_search_region)
    bbox_gt_scaled = bbox_gt_recentered

    # rand_search_x = preprocess(rand_search_region)
    bbox_curr_y = [bbox_gt_scaled.x1, bbox_gt_scaled.y1, bbox_gt_scaled.x2, bbox_gt_scaled.y2]

    # batch_x_curr.append(rand_search_x)
    batch_y_curr.append([num_img] + bbox_curr_y)

    # bbox_gt_scaled.unscale(rand_search_region)
    # img_unscale_curr = cv2.rectangle(rand_search_region,(int(bbox_gt_scaled.x1),int(bbox_gt_scaled.y1)),(int(bbox_gt_scaled.x2),int(bbox_gt_scaled.y2)),(0,255,0),1)
    # rand_search_region = cv2.resize(rand_search_region, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)
    rand_search_x = preprocess(rand_search_region)
    cv2.imwrite('/home/huaxin/data_3T/license_dataset/license_track/dataset_0/image_unprocess_'+format(num_img, '03d')+'.jpg', rand_search_x)

for num_img in range(10000,15000):
    lamda_shift = 5
    lamda_scale = 15
    min_scale = 0.01
    max_scale = 10

    img_path = image_files[num_img]
    image = cv2.imread(img_path)

    json_path = annotations_files[num_img]
    json_file = open(json_path)
    str = json_file.read()
    data = json.loads(str,encoding="GB2312")

    x_value_curr = [data[u'plate_0'][u'PosPt0X'],data[u'plate_0'][u'PosPt1X'],data[u'plate_0'][u'PosPt2X'],data[u'plate_0'][u'PosPt3X']]
    y_value_curr = [data[u'plate_0'][u'PosPt0Y'],data[u'plate_0'][u'PosPt1Y'],data[u'plate_0'][u'PosPt2Y'],data[u'plate_0'][u'PosPt3Y']]

    x1 = (min(x_value_curr))/10#x bottom left
    y1 = (min(y_value_curr))/10#y bottom left
    x2 = (max(x_value_curr))/10#x top right
    y2 = (max(y_value_curr))/10#y top right

    print num_img,'bbox:',x1,y1,x2,y2
    bbox = BoundingBox(x1,y1,x2,y2)

    bbox_curr_gt = bbox
    bbox_curr_shift = BoundingBox(0, 0, 0, 0)
    bbox_curr_shift = bbox_curr_gt.shift(image, lamda_scale, lamda_shift, min_scale, max_scale, True, bbox_curr_shift)
    rand_search_region, rand_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, image)

    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
    bbox_gt_recentered.scale(rand_search_region)
    bbox_gt_scaled = bbox_gt_recentered

    # rand_search_x = preprocess(rand_search_region)
    bbox_curr_y = [bbox_gt_scaled.x1, bbox_gt_scaled.y1, bbox_gt_scaled.x2, bbox_gt_scaled.y2]

    # batch_x_curr.append(rand_search_x)
    batch_y_curr.append([num_img] + bbox_curr_y)

    # bbox_gt_scaled.unscale(rand_search_region)
    # img_unscale_curr = cv2.rectangle(rand_search_region,(int(bbox_gt_scaled.x1),int(bbox_gt_scaled.y1)),(int(bbox_gt_scaled.x2),int(bbox_gt_scaled.y2)),(0,255,0),1)
    # rand_search_region = cv2.resize(rand_search_region, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)
    rand_search_x = preprocess(rand_search_region)
    cv2.imwrite('/home/huaxin/data_3T/license_dataset/license_track/dataset_0/image_unprocess_'+format(num_img, '03d')+'.jpg', rand_search_x)

for num_img in range(15000,25000):
    lamda_shift = 5
    lamda_scale = 15
    min_scale = 0.4
    max_scale = 0.4

    img_path = image_files[num_img]
    image = cv2.imread(img_path)

    json_path = annotations_files[num_img]
    json_file = open(json_path)
    str = json_file.read()
    data = json.loads(str,encoding="GB2312")

    x_value_curr = [data[u'plate_0'][u'PosPt0X'],data[u'plate_0'][u'PosPt1X'],data[u'plate_0'][u'PosPt2X'],data[u'plate_0'][u'PosPt3X']]
    y_value_curr = [data[u'plate_0'][u'PosPt0Y'],data[u'plate_0'][u'PosPt1Y'],data[u'plate_0'][u'PosPt2Y'],data[u'plate_0'][u'PosPt3Y']]

    x1 = (min(x_value_curr))/10#x bottom left
    y1 = (min(y_value_curr))/10#y bottom left
    x2 = (max(x_value_curr))/10#x top right
    y2 = (max(y_value_curr))/10#y top right

    print num_img,'bbox:',x1,y1,x2,y2
    bbox = BoundingBox(x1,y1,x2,y2)

    bbox_curr_gt = bbox
    bbox_curr_shift = BoundingBox(0, 0, 0, 0)
    bbox_curr_shift = bbox_curr_gt.shift(image, lamda_scale, lamda_shift, min_scale, max_scale, True, bbox_curr_shift)
    rand_search_region, rand_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, image)

    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
    bbox_gt_recentered.scale(rand_search_region)
    bbox_gt_scaled = bbox_gt_recentered

    # rand_search_x = preprocess(rand_search_region)
    bbox_curr_y = [bbox_gt_scaled.x1, bbox_gt_scaled.y1, bbox_gt_scaled.x2, bbox_gt_scaled.y2]

    # batch_x_curr.append(rand_search_x)
    batch_y_curr.append([num_img] + bbox_curr_y)

    # bbox_gt_scaled.unscale(rand_search_region)
    # img_unscale_curr = cv2.rectangle(rand_search_region,(int(bbox_gt_scaled.x1),int(bbox_gt_scaled.y1)),(int(bbox_gt_scaled.x2),int(bbox_gt_scaled.y2)),(0,255,0),1)
    # rand_search_region = cv2.resize(rand_search_region, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)

    rand_search_x = preprocess(rand_search_region)
    cv2.imwrite('/home/huaxin/data_3T/license_dataset/license_track/dataset_0/image_unprocess_'+format(num_img, '03d')+'.jpg', rand_search_x)

for num_img in range(25000,29000):
    img_path = image_files[num_img]
    image = cv2.imread(img_path)

    json_path = annotations_files[num_img]
    json_file = open(json_path)
    str = json_file.read()
    data = json.loads(str,encoding="GB2312")

    x_value_curr = [data[u'plate_0'][u'PosPt0X'],data[u'plate_0'][u'PosPt1X'],data[u'plate_0'][u'PosPt2X'],data[u'plate_0'][u'PosPt3X']]
    y_value_curr = [data[u'plate_0'][u'PosPt0Y'],data[u'plate_0'][u'PosPt1Y'],data[u'plate_0'][u'PosPt2Y'],data[u'plate_0'][u'PosPt3Y']]

    x1 = (min(x_value_curr))/10#x bottom left
    y1 = (min(y_value_curr))/10#y bottom left
    x2 = (max(x_value_curr))/10#x top right
    y2 = (max(y_value_curr))/10#y top right

    print num_img,'bbox:',x1,y1,x2,y2
    bbox = BoundingBox(x1,y1,x2,y2)

    bbox_curr_gt = bbox
    curr_search_region, curr_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_gt, image)

    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(curr_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
    bbox_gt_recentered.scale(curr_search_region)
    bbox_gt_scaled = bbox_gt_recentered

    # curr_search_x = preprocess(curr_search_region)
    bbox_curr_y = [bbox_gt_scaled.x1, bbox_gt_scaled.y1, bbox_gt_scaled.x2, bbox_gt_scaled.y2]

    # batch_x_curr.append(curr_search_x)
    batch_y_curr.append([num_img] + bbox_curr_y)

    # bbox_gt_scaled.unscale(curr_search_region)
    # img_unscale_curr = cv2.rectangle(curr_search_region,(int(bbox_gt_scaled.x1),int(bbox_gt_scaled.y1)),(int(bbox_gt_scaled.x2),int(bbox_gt_scaled.y2)),(0,255,0),1)
    # curr_search_region = cv2.resize(curr_search_region, (width_resize, height_resize), interpolation=cv2.INTER_CUBIC)
    curr_search_x = preprocess(curr_search_region)
    cv2.imwrite('/home/huaxin/data_3T/license_dataset/license_track/dataset_0/image_unprocess_'+format(num_img, '03d')+'.jpg', curr_search_x)

np.savetxt('/home/huaxin/data_3T/license_dataset/license_track/dataset_0/bbox_y_gt', batch_y_curr, delimiter=' ', newline="\n", fmt='%1.8f')









    # bbox_gt_scaled.uncenter(image, rand_search_location, edge_spacing_x, edge_spacing_y)
    # img_unscale_curr = cv2.rectangle(image,(int(bbox_gt_scaled.x1),int(bbox_gt_scaled.y1)),(int(bbox_gt_scaled.x2),int(bbox_gt_scaled.y2)),(0,255,0),1)
    # cv2.imwrite('./output/image_output/image_uncenter_'+format(num_img, '03d')+'.jpg', img_unscale_curr)

    # img_out_curr = cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)
    # img_out_curr = img_out_curr# + np.array([104, 117, 123])
    # cv2.imwrite('./output/image_output/image_rect.jpg', img_out_curr)
    # cv2.imwrite('./output/image_output/image_target_'+format(num_img, '03d')+'.jpg', target_pad)
    # cv2.imwrite('./output/image_output/image_shif_'+format(num_img, '03d')+'.jpg', rand_search_region)

# cv2.imshow('show', image)
#
# cv2.waitKey(0)


# import os, sys
# path = license_folder
# for filename in os.listdir(path):
#     if filename.endswith('.json'):
#         print filename
#         newname = filename.replace('\xe6\xb5\x99', 'Z')
#         print newname
#         # print path
#         os.rename(license_folder+'/'+filename, license_folder+'/'+newname)
