from __future__ import division, print_function, absolute_import

import os, tflearn, datetime
import numpy as np
import tensorflow as tf

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d as conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from DataLoader import *

# Dataset Parameters
batch_size = 32
load_size = 313
fine_size = 299
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate_initial = 0.045
learning_rate_decay = 0.964
decay_steps = int(100000/batch_size)
momentum = 0.9
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 50
step_save = 10000
path_save = 'google_net_bn'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def google_net(x):
    #Layer 1
    conv1_7_7 = tflearn.layers.conv.conv_2d(x, nb_filter=64, filter_size=7, strides=2, activation='relu', name='conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2, name='pool1_3_3_s2')
    pool1_3_3 = local_response_normalization(pool1_3_3)

    #Layer 2
    conv2_3_3_reduce = tflearn.layers.conv.conv_2d(pool1_3_3, nb_filter=64, filter_size=1, activation='relu', name='conv2_3_3_reduce')
    conv2_3_3 = tflearn.layers.conv.conv_2d(conv2_3_3_reduce, nb_filter=192, filter_size=3, activation='relu', name='conv2_3_3')
    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3,strides=2, name='pool2_3_3_s2')

    #Inception 3a
    inception_3a_1_1 = tflearn.layers.conv.conv_2d(pool2_3_3, nb_filter=64, filter_size = 1, activation='relu', name='inception_3a_1_1')
    inception_3a_3_3_reduce = tflearn.layers.conv.conv_2d(pool2_3_3, nb_filter=96, filter_size=1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = tflearn.layers.conv.conv_2d(inception_3a_3_3_reduce, nb_filter=128, filter_size=3, activation='relu', name='inception_3a_3_3')
    inception_3a_5_5_reduce = tflearn.layers.conv.conv_2d(pool2_3_3, nb_filter=16, filter_size=1, activation='relu', name='inception_3a_5_5_reduce')
    inception_3a_5_5 = tflearn.layers.conv.conv_2d(inception_3a_5_5_reduce, nb_filter=32,filter_size = 5, activation='relu', name='inception_3a_5_5')
    inception_3a_pool_3_3 = max_pool_2d(pool2_3_3, kernel_size = 3, strides=1, name='inception_3a_pool_3_3')
    inception_3a_pool_1_1 = tflearn.layers.conv.conv_2d(inception_3a_pool_3_3, nb_filter=32,filter_size = 1, strides=1, name='inception_3a_pool_1_1')
    inception_3a_concat = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    #Inception 3b
    inception_3b_1_1 = tflearn.layers.conv.conv_2d(inception_3a_concat, nb_filter=128, filter_size=1, activation='relu', name='inception_3b_1_1')
    inception_3b_3_3_reduce = tflearn.layers.conv.conv_2d(inception_3a_concat, nb_filter=128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = tflearn.layers.conv.conv_2d(inception_3b_3_3_reduce, nb_filter=192, filter_size=3, activation='relu', name='inception_3b_3_3')
    inception_3b_5_5_reduce = tflearn.layers.conv.conv_2d(inception_3a_concat, nb_filter=32, filter_size=1, activation='relu', name='inception_3b_5_5_reduce')
    inception_3b_5_5 = tflearn.layers.conv.conv_2d(inception_3b_5_5_reduce, nb_filter=96, filter_size=5, activation='relu', name='inception_3b_5_5')
    inception_3b_pool_3_3 = max_pool_2d(inception_3a_concat, kernel_size=3, strides=1, name='inception_3b_pool')
    inception_3b_pool_1_1 = tflearn.layers.conv.conv_2d(inception_3b_pool_3_3, nb_filter=64, filter_size = 1, strides=1, name='inception_3b_pool_1_1')
    inception_3b_concat = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat', axis=3)
    
    #Inception 4a
    inception_4a_pool = max_pool_2d(inception_3b_concat, kernel_size = 3, strides=2, name='inception_4a_pool_3_3')
    inception_4a_1_1 = tflearn.layers.conv.conv_2d(inception_4a_pool, nb_filter=192, filter_size = 1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = tflearn.layers.conv.conv_2d(inception_4a_pool, nb_filter=96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = tflearn.layers.conv.conv_2d(inception_4a_3_3_reduce, nb_filter=208, filter_size=3, activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = tflearn.layers.conv.conv_2d(inception_4a_pool, nb_filter=16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = tflearn.layers.conv.conv_2d(inception_4a_5_5_reduce, nb_filter=48,filter_size = 5, activation='relu', name='inception_4a_5_5')
    inception_4a_pool_3_3 = max_pool_2d(inception_4a_pool, kernel_size = 3, strides=1, name='inception_4a_pool_3_3')
    inception_4a_pool_1_1 = tflearn.layers.conv.conv_2d(inception_4a_pool_3_3, nb_filter=64,filter_size = 1, strides=1, name='inception_4a_pool_1_1')
    inception_4a_concat = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3)

    #Inception 4b
    inception_4b_1_1 = tflearn.layers.conv.conv_2d(inception_4a_concat, nb_filter=160, filter_size=1, activation='relu', name='inception_4b_1_1')
    inception_4b_3_3_reduce = tflearn.layers.conv.conv_2d(inception_4a_concat, nb_filter=112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
    inception_4b_3_3 = tflearn.layers.conv.conv_2d(inception_4b_3_3_reduce, nb_filter=224, filter_size=3, activation='relu', name='inception_4b_3_3')
    inception_4b_5_5_reduce = tflearn.layers.conv.conv_2d(inception_4a_concat, nb_filter=24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
    inception_4b_5_5 = tflearn.layers.conv.conv_2d(inception_4b_5_5_reduce, nb_filter=64, filter_size=5, activation='relu', name='inception_4b_5_5')
    inception_4b_pool_3_3 = max_pool_2d(inception_4a_concat, kernel_size=3, strides=1, name='inception_4b_pool')
    inception_4b_pool_1_1 = tflearn.layers.conv.conv_2d(inception_4b_pool_3_3, nb_filter=64, filter_size = 1, strides=1, name='inception_4b_pool_1_1')
    inception_4b_concat = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3)

    #Average Pool
    avg_pool_4b = avg_pool_2d(inception_4a_concat, kernel_size=5, strides=3)

    #Convolution
    conv_4b = tflearn.layers.conv.conv_2d(avg_pool_4b, nb_filter=128, filter_size=1, activation='relu')

    #FC
    fc_4b = fully_connected(conv_4b, n_units=1024, activation='relu')

    dropout_4b = tflearn.layers.core.dropout(fc_4b, 0.3)

    #Softmax
    softmax0 = tflearn.layers.core.activation(dropout_4b, activation='softmax')

    #Inception 4c
    inception_4c_1_1 = tflearn.layers.conv.conv_2d(inception_4b_concat, nb_filter=128, filter_size=1, activation='relu', name='inception_4c_1_1')
    inception_4c_3_3_reduce = tflearn.layers.conv.conv_2d(inception_4b_concat, nb_filter=128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
    inception_4c_3_3 = tflearn.layers.conv.conv_2d(inception_4c_3_3_reduce, nb_filter=256, filter_size=3, activation='relu', name='inception_4c_3_3')
    inception_4c_5_5_reduce = tflearn.layers.conv.conv_2d(inception_4b_concat, nb_filter=24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
    inception_4c_5_5 = tflearn.layers.conv.conv_2d(inception_4c_5_5_reduce, nb_filter=64, filter_size=5, activation='relu', name='inception_4c_5_5')
    inception_4c_pool_3_3 = max_pool_2d(inception_4b_concat, kernel_size=3, strides=1, name='inception_4c_pool')
    inception_4c_pool_1_1 = tflearn.layers.conv.conv_2d(inception_4c_pool_3_3, nb_filter=64, filter_size = 1, strides=1, name='inception_4c_pool_1_1')
    inception_4c_concat = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3)

    #Inception 4d
    inception_4d_1_1 = tflearn.layers.conv.conv_2d(inception_4c_concat, nb_filter=112, filter_size=1, activation='relu', name='inception_4d_1_1')
    inception_4d_3_3_reduce = tflearn.layers.conv.conv_2d(inception_4c_concat, nb_filter=144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
    inception_4d_3_3 = tflearn.layers.conv.conv_2d(inception_4d_3_3_reduce, nb_filter=288, filter_size=3, activation='relu', name='inception_4d_3_3')
    inception_4d_5_5_reduce = tflearn.layers.conv.conv_2d(inception_4c_concat, nb_filter=32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
    inception_4d_5_5 = tflearn.layers.conv.conv_2d(inception_4c_5_5_reduce, nb_filter=64, filter_size=5, activation='relu', name='inception_4d_5_5')
    inception_4d_pool_3_3 = max_pool_2d(inception_4c_concat, kernel_size=3, strides=1, name='inception_4d_pool')
    inception_4d_pool_1_1 = tflearn.layers.conv.conv_2d(inception_4d_pool_3_3, nb_filter=64, filter_size = 1, strides=1, name='inception_4d_pool_1_1')
    inception_4d_concat = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3)

    #Inception 4e
    inception_4e_1_1 = tflearn.layers.conv.conv_2d(inception_4d_concat, nb_filter=256, filter_size=1, activation='relu', name='inception_4e_1_1')
    inception_4e_3_3_reduce = tflearn.layers.conv.conv_2d(inception_4d_concat, nb_filter=160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
    inception_4e_3_3 = tflearn.layers.conv.conv_2d(inception_4e_3_3_reduce, nb_filter=320, filter_size=3, activation='relu', name='inception_4e_3_3')
    inception_4e_5_5_reduce = tflearn.layers.conv.conv_2d(inception_4d_concat, nb_filter=32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
    inception_4e_5_5 = tflearn.layers.conv.conv_2d(inception_4e_5_5_reduce, nb_filter=128, filter_size=5, activation='relu', name='inception_4e_5_5')
    inception_4e_pool_3_3 = max_pool_2d(inception_4c_concat, kernel_size=3, strides=1, name='inception_4e_pool')
    inception_4e_pool_1_1 = tflearn.layers.conv.conv_2d(inception_4d_pool_3_3, nb_filter=128, filter_size = 1, strides=1, name='inception_4e_pool_1_1')
    inception_4e_concat = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1], mode='concat', axis=3)

    #Average Pool
    avg_pool_4e = avg_pool_2d(inception_4d_concat, kernel_size=5, strides=3)

    #Convolution
    conv_4e = tflearn.layers.conv.conv_2d(avg_pool_4e, nb_filter=128, filter_size=1, activation='relu')

    #FC
    fc_4e = fully_connected(conv_4e, n_units=1024, activation='relu')

    dropout_4e = tflearn.layers.core.dropout(fc_4e, 0.3)

    #Softmax
    softmax1 = tflearn.layers.core.activation(dropout_4e, activation='softmax')

    #Inception 5a
    inception_5a_pool = max_pool_2d(inception_4e_concat, kernel_size = 3, strides=2, name='inception_5a_pool_3_3')
    inception_5a_1_1 = tflearn.layers.conv.conv_2d(inception_5a_pool, nb_filter=256, filter_size = 1, activation='relu', name='inception_5a_1_1')
    inception_5a_3_3_reduce = tflearn.layers.conv.conv_2d(inception_5a_pool, nb_filter=160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3 = tflearn.layers.conv.conv_2d(inception_5a_3_3_reduce, nb_filter=320, filter_size=3, activation='relu', name='inception_5a_3_3')
    inception_5a_5_5_reduce = tflearn.layers.conv.conv_2d(inception_5a_pool, nb_filter=32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    inception_5a_5_5 = tflearn.layers.conv.conv_2d(inception_5a_5_5_reduce, nb_filter=128,filter_size = 5, activation='relu', name='inception_5a_5_5')
    inception_5a_pool_3_3 = max_pool_2d(inception_5a_pool, kernel_size = 3, strides=1, name='inception_5a_pool_3_3')
    inception_5a_pool_1_1 = tflearn.layers.conv.conv_2d(inception_5a_pool_3_3, nb_filter=128,filter_size = 1, strides=1, name='inception_5a_pool_1_1')
    inception_5a_concat = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], mode='concat', axis=3)

    #Inception 5b
    inception_5b_1_1 = tflearn.layers.conv.conv_2d(inception_5a_concat, nb_filter=384, filter_size=1, activation='relu', name='inception_5b_1_1')
    inception_5b_3_3_reduce = tflearn.layers.conv.conv_2d(inception_5a_concat, nb_filter=192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
    inception_5b_3_3 = tflearn.layers.conv.conv_2d(inception_5b_3_3_reduce, nb_filter=384, filter_size=3, activation='relu', name='inception_5b_3_3')
    inception_5b_5_5_reduce = tflearn.layers.conv.conv_2d(inception_5a_concat, nb_filter=48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
    inception_5b_5_5 = tflearn.layers.conv.conv_2d(inception_5b_5_5_reduce, nb_filter=128, filter_size=5, activation='relu', name='inception_5b_5_5')
    inception_5b_pool_3_3 = max_pool_2d(inception_5a_concat, kernel_size=3, strides=1, name='inception_5b_pool')
    inception_5b_pool_1_1 = tflearn.layers.conv.conv_2d(inception_5b_pool_3_3, nb_filter=128, filter_size = 1, strides=1, name='inception_5b_pool_1_1')
    inception_5b_concat = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], mode='concat', axis=3)

    #Average Pool
    avg_pool_5b = avg_pool_2d(inception_5b_concat, kernel_size=7, strides=1)

    #Convolution
    conv_5b = tflearn.layers.conv.conv_2d(avg_pool_5b, nb_filter=128, filter_size=1, activation='relu')

    fc_5b = fully_connected(conv_5b, n_units=1024, activation='relu')

    dropout_5b = tflearn.layers.core.dropout(fc_5b, 0.3)

    #Softmax
    softmax2 = tflearn.layers.core.activation(dropout_5b, activation='softmax')
    return softmax2

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': 'data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': 'data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = google_net(x)

# Define learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate_initial, global_step, decay_steps, learning_rate_decay, staircase=True)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

# Launch the graph
with tf.Session(config=config) as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
    
    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))
        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
        
    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
