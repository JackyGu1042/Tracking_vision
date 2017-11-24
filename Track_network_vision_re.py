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
# from numpy import newaxis #for add one more dimension

model_name = 'model_vision_2017-11-221028_re7'

# Parameters
learning_rate = 0.0001
# kNumBatches = 1000

epochs = 10
batch_size = 128

# Network Parameters
n_output = 4  # Cx,Cy,W,H
dropout = 0.5  # Dropout, probability to keep units

#Training resize image size
width_resize = 64
height_resize = 64
channel_resize = 3

#weight and bias initial parameter
mu = 0
sigma = 0.01

#large movement
# lamda_shift = 1
# lamda_scale = 1
# min_scale = 0.2
# max_scale = 6

#stable
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

    with tf.name_scope('train'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        #     .minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(cost)
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

#################################################################################

license_folder = '/home/huaxin/data_3T/license_dataset/license_track/Dataset_0/train/'

if not os.path.isdir(license_folder):
    print('{} is not a valid directory'.format(license_folder))

image_files = sorted(glob.glob(os.path.join(license_folder, '*.jpg')))

text_file = open(license_folder+"bbox_y_gt", "r")
groundtruth_all = text_file.readlines()
text_file.close()

print 'len(groundtruth_frame):', len(groundtruth_all)

train_x_path_curr = []
train_y_bbox_curr = []

for num_img in range(0,len(image_files)):
    img_path = image_files[num_img]
    image = cv2.imread(img_path)

    rand_search_x = preprocess(image)
    # print image
    train_x_path_curr.append(rand_search_x)

    #############################################################################

    y_temp_cur = groundtruth_all[num_img]
    y_temp_cur = y_temp_cur.split()

    bbox_curr_y = [float(i) for i in y_temp_cur]
    # print '\nbox_curr_y',bbox_curr_y

    print num_img, ' bbox:',bbox_curr_y[0],bbox_curr_y[1],bbox_curr_y[2],bbox_curr_y[3]
    train_y_bbox_curr.append(bbox_curr_y)

print 'Training data size',len(train_x_path_curr)

#################################################################################

# Launch the graph
with tf.Session() as sess:
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    merged = tf.summary.merge_all() # tensorflow >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

    sess.run(init)

    # Restore variables from disk.
    saver.restore(sess, "./tmp/"+model_name+".ckpt")
    print("\nModel restored.")

    for epoch in range(epochs):
        #################################################################
        x_trainset_cur_shuf = []
        y_trainset_cur_shuf = []

        index_shuf = range(len(train_x_path_curr))
        shuffle(index_shuf)
        for i in index_shuf:
            x_trainset_cur_shuf.append(train_x_path_curr[i])
            y_trainset_cur_shuf.append(train_y_bbox_curr[i])

        #################################################################
        for batch in range(len(train_x_path_curr)//batch_size):

            batch_x_cur = x_trainset_cur_shuf[batch_size*batch:batch_size*(batch+1)]
            batch_y_cur = y_trainset_cur_shuf[batch_size*batch:batch_size*(batch+1)]

            logits_out = sess.run(logits, feed_dict={
                x_cur: batch_x_cur,
                y: batch_y_cur,
                keep_prob: dropout})

            sess.run(optimizer, feed_dict={
                x_cur: batch_x_cur,
                y: batch_y_cur,
                keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={
                x_cur: batch_x_cur,
                y: batch_y_cur,
                keep_prob: 1.})

            # Calculate batch loss and accuracy
            loss_L1 = sess.run(cost_L1, feed_dict={
                x_cur: batch_x_cur,
                y: batch_y_cur,
                keep_prob: 1.})

            rs = sess.run(merged, feed_dict={
                x_cur: batch_x_cur,
                y: batch_y_cur,
                keep_prob: dropout})
            writer.add_summary(rs, epoch)

            print('\nEpoch {:>2}, Batch {:>3} -'
                  'loss_L1: {:>10.4f}'.format(
                epoch + 1,
                batch + 1,
                loss_L1))

            print 'logits_out[0]',logits_out[0]
            print "batch_y[0]", batch_y_cur[0]
            print 'learning_rate:', learning_rate , ' dropout: ', dropout


            if (epoch+1) % 50 == 0 and batch == 1:
                time = strftime("%Y-%m-%d%H%M", gmtime())
                save_path = saver.save(sess, "./tmp/"+model_name+"_re8.ckpt") #
                print("Model saved in file: %s" % save_path)

    # Save the variables to disk.
    time = strftime("%Y-%m-%d%H%M", gmtime())
    save_path = saver.save(sess, "./tmp/"+model_name+"_re8.ckpt") #
    print("Model saved in file: %s" % save_path)
