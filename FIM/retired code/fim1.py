#%%
## libraries
import numpy as np
import tensorflow as tf
import math
import sys
import scipy
import pickle

# parameters -----------
weight_change = 0.01
beta = 3
# ----------------------

# import dataset
import small_mnist_one_image as small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')

# paths
modelPath = './DATA/h100b32-beta'+str(beta)+'-test-15000'
graphfile = modelPath + '.meta'
savePath = './output/'

#%%
## Gathering initial weights and outputs
def calc_original():
    tf.reset_default_graph()

    graph0 = tf.Graph()
    with graph0.as_default():
        sess0 = tf.Session()

    with sess0 as sess:
        loader = tf.train.import_meta_graph(graphfile)
        loader.restore(sess, modelPath) # restores the graph

        ## check all trainable variables in the graph (weights and biases):
        # print("## Trainable variables: ")
        # for v in tf.trainable_variables():
        #     print(v.name)

        # weights
        e1w = graph0.get_tensor_by_name('encoder/fully_connected/weights:0')
        e2w = graph0.get_tensor_by_name('encoder/fully_connected_1/weights:0')
        e3w = graph0.get_tensor_by_name('encoder/fully_connected_2/weights:0')
        dw  = graph0.get_tensor_by_name('decoder/fully_connected/weights:0')

        # biases
        e1b = graph0.get_tensor_by_name('encoder/fully_connected/biases:0')
        e2b = graph0.get_tensor_by_name('encoder/fully_connected_1/biases:0')
        e3b = graph0.get_tensor_by_name('encoder/fully_connected_2/biases:0')
        db  = graph0.get_tensor_by_name('decoder/fully_connected/biases:0')

        # outputs
        logits = graph0.get_tensor_by_name('decoder/fully_connected/BiasAdd:0')
        results = tf.argmax(logits, 1)
        softmax = tf.nn.softmax(logits, 1)

        # preparing the input feed 
        images = graph0.get_tensor_by_name('images:0')
        labels = graph0.get_tensor_by_name('labels:0')
        feed_dict = {images: mnist_data.test.images, labels: mnist_data.test.labels}

        # extracting values from the graph
        e1w_val, e2w_val, e3w_val, dw_val = sess.run([e1w, e2w, e3w, dw], feed_dict=feed_dict)
        e1b_val, e2b_val, e3b_val, db_val = sess.run([e1b, e2b, e3b, db], feed_dict=feed_dict)
        logits, results, softmax = sess.run([logits, results, softmax], feed_dict=feed_dict)

        return e1w_val, e2w_val, e3w_val, dw_val, e1b_val, e2b_val, e3b_val, db_val, logits, softmax
    sess.close()

e1w_val, e2w_val, e3w_val, dw_val, e1b_val, e2b_val, e3b_val, db_val, logits, softmax = calc_original()
out = {"e1w_val": e1w_val, "e2w_val": e2w_val, "e3w_val": e3w_val, "dw_val": dw_val, "e1b_val": e1b_val, "e2b_val": e2b_val, "e3b_val": e3b_val, "db_val": db_val, "logits": logits, "softmax": softmax}

pickle.dump( out, open( "fim1.p", "wb" ) )
print("original logits and softmax have been pickled ")
