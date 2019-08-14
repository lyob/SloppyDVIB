'''
Calculates the elements of the FIM matrix for the h100b32 network
'''
#%%

#%%
## libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import scipy

# parameters -----------
weight_change = 0.01
beta = 3
# ----------------------

# import dataset
import FIM.small_mnist_one_image as small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')

# paths
modelPath = './FIM/DATA/h100b32-beta'+str(beta)+'-test-15000'
graphfile = modelPath + '.meta'
savePath = './FIM/output/'

#%%
## Gathering initial weights and outputs
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

    # print output
    # print("weights for 1st encoding layer: {}".format(e1w_val))
    # print("shape of the weights: {}".format(e1w_val.shape))
    # print("biases for 1st encoding layer: {}".format(e1b_val))
    # print("shape of the biases: {}".format(e1b_val.shape))
    # print("logits: {}".format(logits))
    # print("shape of logits: {}".format(logits.shape))
    # print("onehot: {}".format(onehot))
    # print("final result: {}".format(results))
sess.close()


#%%
# now we tweak one parameter and compare the output logits
def calc_altered_softmax(parameter):
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
# Calculations for FIM matrix elements
def calc_score(label, parameter):
    softmax_mod, delta_theta = calc_altered_softmax(parameter)
    score = ( np.log(softmax_mod[0][label]) - np.log(softmax[0][label]) ) / delta_theta
    return score

#%%
# build the dictionary of scores for all label and all network parameters
def calc_all_scores():
    print('----------------calculating all the scores------------------')
    scores = {}
    for label in range(10):
        # for param in range(20002):
        print("------------\nlabel: "+str(label)+"/9------------")
        for param in range(10):
            score = calc_score(label, param)
            scores.update({ 'l'+str(label)+'p'+str(param) : score })
    return scores
# this is fine
scores = calc_all_scores()
#%%
# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
def calc_score_pairs():
    print('------------------calculating score pairs-------------------')
    score_pairs = []
    for label in range(10):
        print('-------label: '+str(label)+'/9--------')
        # for i in range(20002):
        score_pairs.append({})
        for i in range(10):
            # for j in range(20002):
            for j in range(10):
                if 'i'+str(j)+'j'+str(i) in score_pairs[label]:
                    score_pairs[label]['i'+str(i)+'j'+str(j)] = score_pairs[label]['i'+str(j)+'j'+str(i)]
                else:
                    score_pairs[label].update( {'i'+str(i)+'j'+str(j) : scores['l'+str(label)+'p'+str(i)] * scores['l'+str(label)+'p'+str(j)]} )
    return score_pairs
# we have the if cond because sp(i,j|l) = sp(j,i|l) -- this would remove 10*20002 unnecessary calculations, but it 
# this is fine
score_pairs = calc_score_pairs()

#%%
# calculate FIM by multiplying the matrix for each label with the probability of that label given optimal parameters
def calc_fim():
    print('-----------------------calculating the FIM---------------------------')
    fim = {} # this will be a 20002 * 20002 matrix
    # for i in range(20002):
    for i in range(10):
        # for j in range(20002):
        for j in range(10):
            if 'i'+str(j)+'j'+str(i) in fim:
                fim['i'+str(i)+'j'+str(j)] = fim['i'+str(j)+'j'+str(i)]
            else:
                sum = 0
                for label in range(10):
                    sum += softmax[0][label] * score_pairs[label]['i'+str(i)+'j'+str(j)]
                fim['i'+str(i)+'j'+str(j)] = sum
    return fim
# we can use the same trick we used in score_pairs above: if fim(ji|l) already exists then copy that value to fim(ij|l) 

fim = calc_fim()

#%%
# converting the dictionary into a matrix
def dict_to_matrix(d):
    print('-----------------converting the dictionary FIM into an array-----------------')
    mtx = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            mtx[i][j] = d["i"+str(i)+"j"+str(j)]
    return mtx

mfim = dict_to_matrix(fim)

mfim_file = savePath+'fim.csv'
np.savetxt(mfim_file, mfim, delimiter=',')

#%%
# calculating the eigenvalues and eigenvectors of the matrix
from scipy.linalg import eigh
print('----------------calculating the eigenvectors and eigenvalues of the FIM---------------')
w, v = eigh(mfim)

#%%
# writes output to a file
w_file = savePath+'eigenvalues.csv'
v_file = savePath+'eigenvectors.csv'

np.savetxt(w_file, w, delimiter=',')
np.savetxt(v_file, v, delimiter=',')