# comparison of the initial hidden layers

#%%
## libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import scipy

#%%
# paths
# path for reduced bottleneck model:
modelPath = './DATA/mnistvib-120000'
graphfile = modelPath + '.meta'
savePath = './DATA/modified_reduced'

# path for original bottleneck model:
modelPath2 = './DATA/mnistvib-120000'
graphfile2 = modelPath2 + '.meta'
savePath2 = './DATA/modified'

#%%
# Importing data set:
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./DATA/mnistdata')
print('MNIST data imported')

#%%
# Initialising graph:
# graph = tf.get_default_graph()
g_1 = tf.Graph()
with g_1.as_default():
    sess_1 = tf.Session()

# Retrieving weights from data file
with sess_1 as sess:
    loader = tf.train.import_meta_graph(graphfile)
    loader.restore(sess, modelPath) # restores the graph

    ## get the connections between the fully connected layers
    d_weights  = g_1.get_tensor_by_name('decoder/fully_connected/weights:0')
    e0_weights = g_1.get_tensor_by_name('encoder/fully_connected/weights:0')
    e1_weights = g_1.get_tensor_by_name('encoder/fully_connected_1/weights:0')
    e2_weights = g_1.get_tensor_by_name('encoder/fully_connected_2/weights:0')

    # the test images and true value labels 
    images = g_1.get_tensor_by_name('images:0')
    labels = g_1.get_tensor_by_name('labels:0')

    ## Testing the accuracy of the model
    feed_dict = {images: mnist_data.test.images, labels: mnist_data.test.labels}
    # IZY, IZX, acc, av_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_acc], 
    #                                      feed_dict=feed_dict)


    d, e0, e1, e2 = sess.run([d_weights, e0_weights, e1_weights, e2_weights], feed_dict=feed_dict)

    print("\nDecoder:\n ", d, \
            "\nHidden Layer 1: \n ", e0, \
            "\nHidden Layer 2: \n", e1, \
            "\nBottleneck Layer: \n", e2)
    
sess.close()

#%%
## Gathering weights from other model
g_2 = tf.Graph()
with g_2.as_default():
    sess_2 = tf.Session()

with sess_2 as sess:
    loader = tf.train.import_meta_graph(graphfile2)
    loader.restore(sess, modelPath2)

    d_weights_orig  = g_2.get_tensor_by_name('decoder/fully_connected/weights:0')
    e0_weights_orig = g_2.get_tensor_by_name('encoder/fully_connected/weights:0')
    e1_weights_orig = g_2.get_tensor_by_name('encoder/fully_connected_1/weights:0')
    e2_weights_orig = g_2.get_tensor_by_name('encoder/fully_connected_2/weights:0')

    images = g_2.get_tensor_by_name('images:0')
    labels = g_2.get_tensor_by_name('labels:0')

    # feed_dict2 = {images: mnist_data.test.images, labels: mnist_data.test.labels}
    # IZY, IZX, acc, av_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_acc], 
    #                                      feed_dict=feed_dict)
    
    feed_dict = {images: mnist_data.test.images, labels: mnist_data.test.labels}

    d_o, e0_o, e1_o, e2_o = sess.run([d_weights_orig, e0_weights_orig, 
                                        e1_weights_orig, e2_weights_orig], 
                                        feed_dict=feed_dict)

    print("\nDecoder (original):\n ", d_o, \
            "\nHidden Layer 1 (orig): \n ", e0_o, \
            "\nHidden Layer 2 (orig): \n", e1_o, \
            "\nBottleneck Layer (orig): \n", e2_o)

sess_2.close()

#%%
# comparing the two weights
e0_comparison = np.divide(e0, e0_o) # visible to hidden layer 1
e1_comparison = np.divide(e1, e1_o) # hidden layer 1 to hidden layer 2
# e2_comparison = np.divide(e2, e2_o) # hidden layer 2 to bottleneck layer (K = 256 for orig, K=16 for new)
# d_comparison =  np.divide(d, d_o)   # bottleneck to decoder layer

#%%
print(e0_comparison)
print("e0: ", e0.size, " e0_comparison: ", e0_comparison.size)

#%%
print(e1_comparison)

#%%
## making some histograms of the data
from matplotlib import pyplot as plt 
# plt.hist(e0_comparison) # without ranges
plt.hist(e0_comparison, range=(-2,2)) # with ranges
plt.title("histogram of e0 differences") 
plt.show()

#%%
print("values less than -100000: ", e0_comparison[e0_comparison<-100000])

#%%
plt.hist(e1_comparison, range=(-2,2))
plt.title("histogram of e1 differences") 
plt.show()

#%%
