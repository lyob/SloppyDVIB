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
l = 6
# ----------------------

# import dataset
import small_mnist_one_image as small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')

# import original
original = pickle.load( open( "fim1.p", "rb" ) )
e1w_val, e2w_val, e3w_val, dw_val = original["e1w_val"], original['e2w_val'], original['e3w_val'], original['dw_val']
e1b_val, e2b_val, e3b_val, db_val = original['e1b_val'], original['e2b_val'], original['e3b_val'], original['db_val']
logits, softmax = original["logits"], original['softmax']

# paths
modelPath = './DATA/h100b32-beta'+str(beta)+'-test-15000'
graphfile = modelPath + '.meta'
savePath = './output/'

#%%
# now we tweak one parameter and compare the output logits
def calc_altered_softmax(parameter):
    if parameter % 1000 == 0:
        print("parameter: "+str(parameter)+"/20001")
    tf.reset_default_graph()

    graph1 = tf.Graph()
    with graph1.as_default():
        sess1 = tf.Session()

    with sess1 as sess:
        loader = tf.train.import_meta_graph(graphfile)
        loader.restore(sess, modelPath) # restores the graph

        # get specific parameter we want to modify
        if parameter in range(0,6400): 
            # e1w, 64 by 100
            signifier = 'w'
            layer_index = parameter - 0
            yrange = 100
            var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected/weights:0')
            modified = np.copy(e1w_val)
            original = np.copy(e1w_val)
        elif parameter in range(6400, 6500):
            # e1b, 100
            signifier = 'b'
            layer_index = parameter - 6400
            var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected/biases:0')
            modified = np.copy(e1b_val)
            original = np.copy(e1b_val)
        elif parameter in range(6500, 16500):
            # e2w, 100 by 100
            signifier = 'w'
            layer_index = parameter - 6500
            yrange = 100
            var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_1/weights:0')
            modified = np.copy(e2w_val)
            original = np.copy(e2w_val)
        elif parameter in range(16500, 16600):
            # e2b, 100
            signifier = 'b'
            layer_index = parameter - 16500
            var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_1/biases:0')
            modified = np.copy(e2b_val)
            original = np.copy(e2b_val)
        elif parameter in range(16600, 19800):
            # e3w, 100 by 32
            signifier = 'w'
            layer_index = parameter - 16600
            yrange = 32
            var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_2/weights:0')
            modified = np.copy(e3w_val)
            original = np.copy(e3w_val)
        elif parameter in range(19800, 19832):
            # e3b, 32
            signifier = 'b'
            layer_index = parameter - 19800
            var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_2/biases:0')
            modified = np.copy(e3b_val)
            original = np.copy(e3b_val)
        elif parameter in range(19832, 19992):
            # dw, 16 by 10
            signifier = 'w'
            layer_index = parameter - 19832
            yrange = 10
            var_pre_mod = graph1.get_tensor_by_name('decoder/fully_connected/weights:0')
            modified = np.copy(dw_val)
            original = np.copy(dw_val)
        elif parameter in range(19992, 20002):
            # db, 10
            signifier = 'b'
            layer_index = parameter - 19992
            var_pre_mod = graph1.get_tensor_by_name('decoder/fully_connected/biases:0')
            modified = np.copy(db_val)
            original = np.copy(db_val)

        # indexing depends on if the layer is a weight or bias
        if signifier == "w":
            xval = int(np.floor(layer_index / yrange))
            yval = layer_index - yrange * xval
            delta_theta = modified[xval][yval] * weight_change
            modified[xval][yval] = modified[xval][yval] * (1-weight_change)
        elif signifier == "b":
            delta_theta = modified[layer_index] * weight_change
            modified[layer_index] = modified[layer_index] * (1-weight_change)

        # modify the tensor:
        assign_op = tf.assign(var_pre_mod, modified) # here we replace the tensor with a modified array
        sess.run(assign_op) # and run this assign operation to insert this new array into the graph

        # outputs
        logits_mod = graph1.get_tensor_by_name('decoder/fully_connected/BiasAdd:0')
        results_mod = tf.argmax(logits, 1)
        softmax_mod = tf.nn.softmax(logits_mod, 1)

        ## Testing the accuracy of the model
        images1 = graph1.get_tensor_by_name('images:0')
        labels1 = graph1.get_tensor_by_name('labels:0')
        feed_dict = {images1: mnist_data.test.images, labels1: mnist_data.test.labels}

        # extracting modded values from the graph
        # var_post_mod = sess.run(var_pre_mod, feed_dict=feed_dict)
        # logits_mod, results_mod, softmax_mod = sess.run([logits_mod, results_mod, softmax_mod], feed_dict=feed_dict)
        softmax_mod = sess.run(softmax_mod, feed_dict=feed_dict)

        ## Print outputs:
        # print("comparing the modded results: \noriginal result: {}\tmodified results: {}".format(results, results_mod))
        # print("the layer post modification: {}".format(var_post_mod))
        # if signifier == 'w':
        #     print("The modified parameter value: \noriginal parameter: {}\naltered parameter: {}".format(original[xval][yval], var_post_mod[xval][yval]))
        # elif signifier == 'b':
        #     print("The modified parameter value: \noriginal parameter: {}\naltered parameter: {}".format(original[layer_index], var_post_mod[layer_index]))

        # print("softmax output (pre mod): {}".format(softmax))
        # print("softmax output (post mod): {}".format(softmax_mod))
        return softmax_mod, delta_theta
    sess.close()

#%%
def calc_score(label, parameter):
    softmax_mod, delta_theta = calc_altered_softmax(parameter)
    if delta_theta != 0:
        score = ( np.log(softmax_mod[0][label]) - np.log(softmax[0][label]) ) / delta_theta
    else:
        score = ( np.log(softmax_mod[0][label]) - np.log(softmax[0][label]) ) / 0.0001
    return score

# build the dictionary of scores
def calc_all_scores():
    print('----------------calculating all theta_i values------------------')
    scores = {}
    for label in range(l,l+1): # for label = 0
        print("------------label: "+str(label)+"/9------------")
        for param in range(20002):
            score = calc_score(label, param)
            scores.update({ 'l'+str(label)+'p'+str(param) : score })
    return scores
# this is fine
scores = calc_all_scores()

pickle.dump( scores, open("scores"+str(l)+".p", "wb") )
print( "the scores for label "+str(l)+" has been pickled" )