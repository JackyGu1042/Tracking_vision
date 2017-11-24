import os
import cv2
import math
import glob
import random
import skvideo.io
import numpy as np
import tensorflow as tf
from random import shuffle
from time import gmtime, strftime

import pygame
import pygame.camera
from pygame.locals import *

from load_process import video
from load_process import frame
from load_process import BoundingBox
from load_process import annotation
from load_process import cropPadImage
from load_process import preprocess
from load_process import load_annotation_file

model_name = 'model_vision_2017-11-221028_re8.ckpt'

# Parameters
batch_gt_prev_initial = BoundingBox(208,167,414,215) #car

n_output = 4

#Training resize image size
width_resize = 64
height_resize = 64
channel_resize = 3

#weight and bias initial parameter
mu = 0
sigma = 0.01

lamda_shift = 30
lamda_scale = 30
min_scale = 0.4
max_scale = 0.4

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

pygame.init()
pygame.camera.init()

cam = pygame.camera.Camera("/dev/video0")#,(640,480))
cam.start()

#######################################################################

network_out = []
network_out_width_crop = []
network_out_height_crop = []
network_out_xstart_crop = []
network_out_ystart_crop = []
vide_output = []

bbox_curr_prior_tight = batch_gt_prev_initial
num_img = 0
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, "./tmp/"+model_name)
    print("\nModel restored.")

    while(True):
        if num_img ==0:
            bbox_curr_prior_tight = batch_gt_prev_initial
        # Capture frame-by-frame
        frame = cam.get_image()
        frame_numpy =  pygame.surfarray.array3d(frame)
        frame_numpy =  np.fliplr(frame_numpy)
        frame_numpy_90 = np.rot90(frame_numpy,1)

        b,g,r = cv2.split(frame_numpy_90)           # get b, g, r
        frame_input = cv2.merge([r,g,b])     # switch it to r, g, b
        image_curr_input = frame_input

        # cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_prior_tight, image_curr_input)

        bbox_curr_gt = bbox_curr_prior_tight
        # bbox_curr_shift = BoundingBox(0, 0, 0, 0)
        # bbox_curr_shift = bbox_curr_gt.shift(image_curr_input, lamda_scale, lamda_shift, min_scale, max_scale, True, bbox_curr_shift)
        cur_search_region, search_location , edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_gt, image_curr_input)
        curr_search_x = preprocess(cur_search_region)

        bbox_estimate = sess.run(logits, feed_dict={
                    x_cur: [curr_search_x],
                    y: [[1,1,1,1]],
                    keep_prob: 1})

        bbox_estimate = BoundingBox(bbox_estimate[0, 0], bbox_estimate[0, 1], bbox_estimate[0, 2], bbox_estimate[0, 3])

        # Inplace correction of bounding box
        print 'height', cur_search_region.shape[0], 'width', cur_search_region.shape[1]
        bbox_estimate.unscale(cur_search_region)

        print 'search_location.x1', search_location.x1, 'search_location.y1', search_location.y1
        print 'edge_spacing_x',edge_spacing_x,' edge_spacing_y',edge_spacing_y
        bbox_estimate.uncenter(image_curr_input, search_location, edge_spacing_x, edge_spacing_y)

        bbox_curr_prior_tight = bbox_estimate

        bbox_output = [bbox_estimate.x1, bbox_estimate.y1, bbox_estimate.x2, bbox_estimate.y2]
        bbox_output = [int(i) for i in bbox_output]
        print 'bbox_output int', bbox_output

        img_out_curr = cv2.rectangle(image_curr_input,(bbox_output[0],bbox_output[1]),(bbox_output[2],bbox_output[3]),(0,255,0),2)
        # cv2.imwrite('./output/image_output/image_unscale_cur'+format(num_img, '03d')+'.jpg', img_out_curr)
        num_img=num_img+1

        # Display the resulting frame
        cv2.imshow('frame',img_out_curr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # bbox_output_unscale = [bbox_estimate.x1, bbox_estimate.y1, bbox_estimate.x2, bbox_estimate.y2]
        # bbox_output_unscale = [int(i) for i in bbox_output_unscale]
        # img_out_curr_unscale = cv2.rectangle(cur_search_region,(bbox_output_unscale[0],bbox_output_unscale[1]),(bbox_output_unscale[2],bbox_output_unscale[3]),(0,255,0),1)
        # img_out_curr_unscale = img_out_curr_unscale
        # img_out_prev_unscale = target_pad
        # vis = np.concatenate((img_out_curr_unscale, img_out_prev_unscale), axis=1)
        # cv2.imwrite('./output/image_output/image_unscale_pre_cur'+format(num_img, '03d')+'.jpg', vis)
