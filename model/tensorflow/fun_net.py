import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 128
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
path_save = 'fun_net_bn'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def fun_net(x, keep_dropout, train_phase):
    weights = {

        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=np.sqrt(2./(3*3*3)))),
        'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=np.sqrt(2./(3*3*32)))),

        'wsc3d': tf.Variable(tf.random_normal([3, 3, 64, 1], stddev=np.sqrt(2./(3*3*64)))),
        'wsc3p': tf.Variable(tf.random_normal([1, 1, 64, 128], stddev=np.sqrt(2./(1*1*64)))),

        'wsc4d': tf.Variable(tf.random_normal([3, 3, 128, 1], stddev=np.sqrt(2./(3*3*128)))),
        'wsc4p': tf.Variable(tf.random_normal([1, 1, 128, 128], stddev=np.sqrt(2./(1*1*128)))),

        'wc4a': tf.Variable(tf.random_normal([1, 1, 64, 128], stddev=np.sqrt(2./(1*1*64)))),

        'wsc5d': tf.Variable(tf.random_normal([3, 3, 128, 1], stddev=np.sqrt(2./(3*3*128)))),
        'wsc5p': tf.Variable(tf.random_normal([1, 1, 128, 256], stddev=np.sqrt(2./(1*1*128)))),

        'wsc6d': tf.Variable(tf.random_normal([3, 3, 256, 1], stddev=np.sqrt(2./(3*3*256)))),
        'wsc6p': tf.Variable(tf.random_normal([1, 1, 256, 256], stddev=np.sqrt(2./(1*1*256)))),

        'wc6a': tf.Variable(tf.random_normal([1, 1, 128, 256], stddev=np.sqrt(2./(1*1*128)))),

        'wsc7d': tf.Variable(tf.random_normal([3, 3, 256, 1], stddev=np.sqrt(2./(3*3*256)))),
        'wsc7p': tf.Variable(tf.random_normal([1, 1, 256, 728], stddev=np.sqrt(2./(1*1*256)))),

        'wsc8d': tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728)))),
        'wsc8p': tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728)))),

        'wc8a': tf.Variable(tf.random_normal([1, 1, 256, 728], stddev=np.sqrt(2./(1*1*256)))),

        'wsc9d': tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728)))),
        'wsc9p': tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728)))),

        'wsc10d': tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728)))),
        'wsc10p': tf.Variable(tf.random_normal([1, 1, 728, 1024], stddev=np.sqrt(2./(1*1*728)))),

        'wc10a': tf.Variable(tf.random_normal([1, 1, 728, 1024], stddev=np.sqrt(2./(1*1*728)))),

        'wsc11d': tf.Variable(tf.random_normal([3, 3, 1024, 1], stddev=np.sqrt(2./(3*3*1024)))),
        'wsc11p': tf.Variable(tf.random_normal([1, 1, 1024, 1536], stddev=np.sqrt(2./(1*1*1024)))),

        'wsc12d': tf.Variable(tf.random_normal([3, 3, 1536, 1], stddev=np.sqrt(2./(3*3*1536)))),
        'wsc12p': tf.Variable(tf.random_normal([1, 1, 1536, 2048], stddev=np.sqrt(2./(1*1*1536)))),

        'wf': tf.Variable(tf.random_normal([2048, 1024], stddev=np.sqrt(2./2048))),
        'wo': tf.Variable(tf.random_normal([1024, 100], stddev=np.sqrt(2./1024)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }


    # Middle flow weights #
    for i in range(8):
        depthwise1 = 'mwsc{0}d'.format(i*3+1)
        depthwise2 = 'mwsc{0}d'.format(i*3+2)
        depthwise3 = 'mwsc{0}d'.format(i*3+3)

        pointwise1 = 'mwsc{0}p'.format(i*3+1)
        pointwise2 = 'mwsc{0}p'.format(i*3+2)
        pointwise3 = 'mwsc{0}p'.format(i*3+3)

        weights[depthwise1] = tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728))))
        weights[depthwise2] = tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728))))
        weights[depthwise3] = tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728))))

        weights[pointwise1] = tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728))))
        weights[pointwise2] = tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728))))
        weights[pointwise3] = tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728))))


    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Entry flow ====================================================================================================================

    #Layer 1: conv32, 3x3, stride 2x2, relu
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)

    #Layer 2: conv64, 3x3, relu
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)

    #Layer 3: Sep Conv 128, 3x3, relu
    conv3 = tf.nn.separable_conv2d(conv2, weights['wsc3d'], weights['wsc3p'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    #Layer 4: Sep Conv 128, 3x3, max pool, 3x3, stride 2x2
    conv4 = tf.nn.separable_conv2d(conv3, weights['wsc4d'], weights['wsc4p'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Layer 4a: conv128, 1x1, stride 2x2
    conv4a = tf.nn.conv2d(conv2, weights['wc4a'], strides=[1, 2, 2, 1], padding='SAME')
    conv4a = batch_norm_layer(conv4a, train_phase, 'bn4a')
    conv4 = tf.add(conv4, conv4a)

    #Layer 5: relu, sep conv 256, 3x3
    conv5 = tf.nn.relu(conv4);
    conv5 = tf.nn.separable_conv2d(conv5, weights['wsc5d'], weights['wsc5p'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')

    #Layer 6: relu, sep conv 256, 3x3, max pool
    conv6 = tf.nn.relu(conv5)
    conv6 = tf.nn.separable_conv2d(conv6, weights['wsc6d'], weights['wsc6p'], strides=[1, 1, 1, 1], padding='SAME')
    conv6 = batch_norm_layer(conv6, train_phase, 'bn6')
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Layer 6a: conv 256, 1x1, stride 2x2
    conv6a = tf.nn.conv2d(conv4a, weights['wc6a'], strides=[1, 2, 2, 1], padding='SAME')
    conv6a = batch_norm_layer(conv6a, train_phase, 'bn6a')
    conv6 = tf.add(conv6, conv6a)

    #Layer 7: relu, sep conv 728, 3x3
    conv7 = tf.nn.relu(conv6);
    conv7 = tf.nn.separable_conv2d(conv7, weights['wsc7d'], weights['wsc7p'], strides=[1, 1, 1, 1], padding='SAME')
    conv7 = batch_norm_layer(conv7, train_phase, 'bn7')

    #Layer 8: relu, sep conv 728, 3x3, max pool
    conv8 = tf.nn.relu(conv7);
    conv8 = tf.nn.separable_conv2d(conv8, weights['wsc8d'], weights['wsc8p'], strides=[1, 1, 1, 1], padding='SAME')
    conv8 = batch_norm_layer(conv8, train_phase, 'bn8')
    conv8 = tf.nn.max_pool(conv8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Layer 8a: conv 728, 1x1, stride 2x2
    conv8a = tf.nn.conv2d(conv6a, weights['wc8a'], strides=[1, 2, 2, 1], padding='SAME')
    entry_flow_out = tf.add(conv8, conv8a)



    # Middle flow --------------------------------------------------------------------------------------------------------------

    middle_flow_val = entry_flow_out

    for i in range(8):

        stored_value = middle_flow_val

        middle_flow_val = tf.nn.relu(middle_flow_val)
        middle_flow_val = tf.nn.separable_conv2d(middle_flow_val, weights['mwsc{0}d'.format(i*3+1)], weights['mwsc{0}p'.format(i*3+1)], strides=[1, 1, 1, 1], padding='SAME')
        middle_flow_val = batch_norm_layer(middle_flow_val, train_phase, 'mbn{0}'.format(i*3+1))

        middle_flow_val = tf.nn.relu(middle_flow_val)
        middle_flow_val = tf.nn.separable_conv2d(middle_flow_val, weights['mwsc{0}d'.format(i*3+2)], weights['mwsc{0}p'.format(i*3+2)], strides=[1, 1, 1, 1], padding='SAME')
        middle_flow_val = batch_norm_layer(middle_flow_val, train_phase, 'mbn{0}'.format(i*3+2))

        middle_flow_val = tf.nn.relu(middle_flow_val)
        middle_flow_val = tf.nn.separable_conv2d(middle_flow_val, weights['mwsc{0}d'.format(i*3+3)], weights['mwsc{0}p'.format(i*3+3)], strides=[1, 1, 1, 1], padding='SAME')
        middle_flow_val = batch_norm_layer(middle_flow_val, train_phase, 'mbn{0}'.format(i*3+3))

        middle_flow_val = tf.add(middle_flow_val, stored_value)




    # Exit flow -------------------------------------------------------------------------------------------------------------------

    conv9 = tf.nn.relu(middle_flow_val)
    conv9 = tf.nn.separable_conv2d(conv9, weights['wsc9d'], weights['wsc9p'], strides=[1, 1, 1, 1], padding='SAME')
    conv9 = batch_norm_layer(conv9, train_phase, 'bn9')

    conv10 = tf.nn.relu(conv9)
    conv10 = tf.nn.separable_conv2d(conv10, weights['wsc10d'], weights['wsc10p'], strides=[1, 1, 1, 1], padding='SAME')
    conv10 = batch_norm_layer(conv10, train_phase, 'bn10')
    conv10 = tf.nn.max_pool(conv10, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv10a = tf.nn.conv2d(middle_flow_val, weights['wc10a'], strides=[1, 2, 2, 1], padding='SAME')
    conv10a = batch_norm_layer(conv10a, train_phase, 'bn10a')
    conv10 = tf.add(conv10, conv10a)

    conv11 = tf.nn.separable_conv2d(conv10, weights['wsc11d'], weights['wsc11p'], strides=[1, 1, 1, 1], padding='SAME')
    conv11 = batch_norm_layer(conv11, train_phase, 'bn11')
    conv11 = tf.nn.relu(conv11)

    conv12 = tf.nn.separable_conv2d(conv11, weights['wsc12d'], weights['wsc12p'], strides=[1, 1, 1, 1], padding='SAME')
    conv12 = batch_norm_layer(conv12, train_phase, 'bn12')
    conv12 = tf.nn.relu(conv12)

    dense = tf.reduce_mean(conv12, [1, 2])
    dense = tf.matmul(dense, weights['wf'])
    dense = batch_norm_layer(dense, train_phase, 'd')
    dense = tf.nn.relu(dense)
    dense = tf.nn.dropout(dense, keep_dropout)

    out = tf.add(tf.matmul(dense, weights['wo']), biases['bo'])

    vars = 0
    for v in tf.global_variables():
        vars += np.prod(v.get_shape().as_list())

    print vars 

    return out

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
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
logits = fun_net(x, keep_dropout, train_phase)

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

# Launch the graph
with tf.Session() as sess:
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
