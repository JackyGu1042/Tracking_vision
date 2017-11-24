import os
import cv2
import math
import glob
import json
import random
import skvideo.io
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

model_name = 'model_vision_2017-11-221028_re7.ckpt'

# Parameters
lamda_shift = 2
lamda_scale = 2
min_scale = 0.2
max_scale = 0.2

batch_gt_prev_initial = BoundingBox(276,177,363,232) #drunk

n_output = 4

#Training resize image size
width_resize = 64
height_resize = 64
channel_resize = 3

#weight and bias initial parameter
mu = 0
sigma = 0.01

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32], mean = mu, stddev = sigma), name='weight'),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64], mean = mu, stddev = sigma), name='weight'),
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128], mean = mu, stddev = sigma), name='weight'),

    'wd1': tf.Variable(tf.random_normal([8*8*128, 1024*4], mean = mu, stddev = sigma), name='weight'),
    'wd2': tf.Variable(tf.random_normal([1024*4, 1024*4], mean = mu, stddev = sigma), name='weight'),
    'out': tf.Variable(tf.random_normal([1024*4, n_output], mean = mu, stddev = sigma), name='weight')}

biases = {
    'bc1': tf.Variable(tf.random_normal([32], mean = mu, stddev = sigma), name='bias'),
    'bc2': tf.Variable(tf.random_normal([64], mean = mu, stddev = sigma), name='bias'),
    'bc3': tf.Variable(tf.random_normal([128], mean = mu, stddev = sigma), name='bias'),


    'bd1': tf.Variable(tf.random_normal([1024*4], mean = mu, stddev = sigma), name='bias'),
    'bd2': tf.Variable(tf.random_normal([1024*4], mean = mu, stddev = sigma), name='bias'),
    'out': tf.Variable(tf.random_normal([n_output], mean = mu, stddev = sigma), name='bias')}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def conv_net(x_cur, weights, biases, dropout):
    with tf.name_scope('conv1'):
        # Layer 1 - 227*227*1 to 114*114*32
        conv1 = conv2d(x_cur, weights['wc1'], biases['bc1'])
        conv1 = maxpool2d(conv1, k=2)
        print conv1.get_shape()

    with tf.name_scope('conv2'):
        # Layer 2 - 114*114*32 to 58*58*64
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = maxpool2d(conv2, k=2)
        print conv2.get_shape()
    with tf.name_scope('conv3'):
        # Layer 2 - 58*58*64 to 30*30*128
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = maxpool2d(conv3, k=2)
        print conv3.get_shape()

    with tf.name_scope('conv_reshape'):
        # Fully connected layer - 15*15*256 to 1024*4
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        print fc1.get_shape()
#######################################################################

    with tf.name_scope('fc1'):
        # Fully connected layer - 1024*4 to 1024*4
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)
        print fc1.get_shape()
    with tf.name_scope('fc2'):
        # Fully connected layer - 1024*4 to 1024*4
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, dropout)
        print fc2.get_shape()

    with tf.name_scope('outputs'):
        # Output Layer - regression - 1024 to 4
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
        print out.get_shape()
    return out

# define tf Graph input
with tf.name_scope('inputs'):
    x_cur = tf.placeholder(tf.float32, [None, width_resize, height_resize, 3], name='x_cur')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

with tf.name_scope('truth'):
    y = tf.placeholder(tf.float32, [None, n_output], name='y_input')

with tf.name_scope('network'):
    with tf.name_scope('logits'):
        # Model
        logits = conv_net(x_cur, weights, biases, keep_prob)
        tf.summary.histogram('logits', logits)

    # Define loss and optimizer
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.square(logits-y))
        cost_L1 = tf.reduce_mean(tf.square(logits-y)/2)
        #Send to TensorBoard to display
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('cost_L1', cost_L1)

    #Accuracy
    with tf.name_scope('predict'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #Send to TensorBoard to display
        tf.summary.scalar('Accuracy', accuracy)

    with tf.name_scope('initial'):
        # Initializing the variables
        init = tf.initialize_all_variables()

#Create a saver object which will save all the variables
saver = tf.train.Saver()

#######################################################################

license_folder = './reasonable'

if not os.path.isdir(license_folder):
    print('{} is not a valid directory'.format(license_folder))

image_files = sorted(glob.glob(os.path.join(license_folder, '*.jpg')))
annotations_files = sorted(glob.glob(os.path.join(license_folder, '*.json')))

train_x_path_curr = []
train_y_bbox_curr = []

for num_img in range(0,len(image_files)):
    img_path = image_files[num_img]

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

    print 'bbox:',x1,y1,x2,y2
    bbox = BoundingBox(x1,y1,x2,y2)

    train_x_path_curr.append(img_path)
    train_y_bbox_curr.append(bbox)

print 'Training data size',len(train_x_path_curr)
#######################################################################

network_out = []
network_out_width_crop = []
network_out_height_crop = []
network_out_xstart_crop = []
network_out_ystart_crop = []
vide_output = []

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, "./tmp/"+model_name)
    print("\nModel restored.")

    err_all = []
    for num_img in range(0,10000):
        if num_img == 0:
            bbox = train_y_bbox_curr[num_img]

        image_index = num_img

        img_path = train_x_path_curr[image_index]
        image = cv2.imread(img_path)

        # if num_img == 0:
        #     cv2.imshow("first", image)
        #     cv2.waitKey(0)

        bbox = train_y_bbox_curr[num_img]
        bbox_curr_gt = bbox
        bbox_curr_shift = BoundingBox(0, 0, 0, 0)
        bbox_curr_shift = bbox_curr_gt.shift(image, lamda_scale, lamda_shift, min_scale, max_scale, True, bbox_curr_shift)
        rand_search_region, rand_search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, image)

        bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
        bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)
        bbox_gt_recentered.scale(rand_search_region)
        bbox_gt_scaled = bbox_gt_recentered

        rand_search_x = preprocess(rand_search_region)

        bbox_estimate = sess.run(logits, feed_dict={
                    x_cur: [rand_search_x],
                    y: [[1,1,1,1]],
                    keep_prob: 1})

        bbox_estimate = BoundingBox(bbox_estimate[0, 0], bbox_estimate[0, 1], bbox_estimate[0, 2], bbox_estimate[0, 3])

        err_x1 = abs((bbox_gt_scaled.x1-bbox_estimate.x1)/bbox_curr_gt.x1)
        err_y1 = abs((bbox_gt_scaled.y1-bbox_estimate.y1)/bbox_curr_gt.y1)
        err_x2 = abs((bbox_gt_scaled.x2-bbox_estimate.x2)/bbox_curr_gt.x2)
        err_y2 = abs((bbox_gt_scaled.y2-bbox_estimate.y2)/bbox_curr_gt.y2)
        err = [err_x1,err_y1,err_x2,err_y2]
        err_max = np.max(err)
        print bbox_gt_scaled.x1,bbox_estimate.x1
        print num_img, 'position err_max', err_max
        err_all.append(err_max)



    eval_rate = np.sum(err_all)/len(err_all)
    print  'position eval_rate', eval_rate
    print "\nmodel_name",model_name
