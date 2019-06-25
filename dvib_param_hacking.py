#%%
## libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys

#%%
## Goal: figure out how to change the parameters and measure the resulting performance of the network (i.e. without retraining). 

# Initialising graph:
graph = tf.get_default_graph()

#%%
# paths
modelPath = './DATA/mnistvib-12000'
graphfile = './DATA/mnistvib-12000.meta'
newSavePath = './DATA/modified'

#%%
# Importing data set:
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./DATA/mnistdata')
print('MNIST data imported')


#%%
# quick inspect of specific tensors in checkpoint file
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as ptif 
ptif(file_name = modelPath, all_tensors=False, tensor_name='decoder*')
ptif(file_name = modelPath, all_tensors=False, tensor_name='Mean_1')

#%%
# Testing accuracy
with tf.Session() as sess:
    loader = tf.train.import_meta_graph(graphfile)
    loader.restore(sess, modelPath) # restores the graph
    
    # print(tf.global_variables())

    ## get specific tensors we want to modify
    wc = graph.get_tensor_by_name('decoder/fully_connected/weights:0')

    ## modify the tensors:
    # sess.run( tf.assign( wc, tf.multiply(wc, 0) ) ) # Setting the values of the variable 'wc' in the model to zero.
    # sess.run( tf.assign( logits, tf.multiply(logits, 0) ) ) # 'Tensor' object has no attribute 'assign' (because it's not a variable)

    ## See various variables
    # print("## All operations: ")
    # for op in graph.get_operations():
    #     print(op.name)

    # print("## All variables: ")
    # for v in tf.global_variables():
    #     print(v.name)
    
    # print("## Trainable variables: ")
    # for v in tf.trainable_variables():
    #     print(v.name)

    ## extract specific variables from checkpoint file
    images = graph.get_tensor_by_name('images:0')
    labels = graph.get_tensor_by_name('labels:0')

    accuracy = graph.get_tensor_by_name('Mean_1:0')
    avg_acc = graph.get_tensor_by_name('Mean_3:0')

    IZY_bound = graph.get_tensor_by_name('sub:0')
    IZX_bound = graph.get_tensor_by_name('truediv_1:0')

    class_loss = graph.get_tensor_by_name('truediv:0')
    info_loss = graph.get_tensor_by_name('truediv_1:0')
    total_loss = graph.get_tensor_by_name('add:0')
    beta = 1e-3

    logits = graph.get_tensor_by_name('decoder/fully_connected/BiasAdd:0')
    onehot = graph.get_tensor_by_name('one_hot:0')
    results = tf.argmax(logits, 1)

    ## Testing the accuracy of the model
    feed_dict = {images: mnist_data.test.images, labels: mnist_data.test.labels}
    IZY, IZX, acc, av_acc, wc_out = sess.run([IZY_bound, IZX_bound, accuracy, avg_acc, wc], 
                                        feed_dict=feed_dict)

    ## Checking calculations are being done correctly
    class_l, info_l, total_l = sess.run([class_loss, info_loss, total_loss], feed_dict=feed_dict)
    total_l2 = class_l + beta * info_l
    check_loss = total_l2 - total_l

    ## Output files
    lg, oh, rs = sess.run([logits, onehot, results], feed_dict=feed_dict)

    ## Print outputs:
    print("\nIZY: ", IZY, "\tIZX: ", IZX, "\tAccuracy: ", acc, "\tAverage Accuracy: ", av_acc, \
        "\n\nTotal loss from graph:", total_l, "\tTotal loss from calc:", total_l2, "\tChecking total losses: ", check_loss)
    print("\ndecoder weight tensor: \n", wc_out)
    print("\nlogits: \n", lg, "\nsize of logits matrix: ", lg.shape)
    print("\none_hot_labels:\n", oh)
    print("\nresults:\n", rs)

    ## Save new data
    # save_path = loader.save(sess, newSavePath)
    # print('new Checkpoint saved')

#%%
# more useful data
lgt = lg[:10] # output layer (prob) 
oht = oh[:10] #
rst = rs[:10]

first = lgt[0]
print(first)
print(np.sum(first))

#%%
