# combines all the fim calcs
#
# The file system should have the following structure:
# FIM
# ├───/DATA
# ├───/npy
# ├───/json
# ├───/pickle
# ├───/plots
# ├───fimtotal.py
# └───fim4.py
#%%

## libraries
import math
import numpy as np
import tensorflow as tf
import scipy
import pickle
import multiprocessing as mp
import ray
import psutil
import nujson as ujson
from memory_profiler import profile
from tensorflow.python.framework import dtypes


### global variables ###
# parameters -----------
weight_change = 0.01
beta = 2
size = 100

# paths
modelPath = './DATA/h100b32-beta{}-test-15000'.format(beta)
graphfile = modelPath + '.meta'
npypath = './output-ray/npy/'
jsonpath = './output-ray/json/'
picklepath = './output-ray/pickle/'

# dataset
import small_mnist_one_image as small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')




#----------------------fim1.py------------------------------------------------
## Gathering initial weights and outputs
def calc_original():
    tf.reset_default_graph()

    graph0 = tf.Graph()
    with graph0.as_default():
        sess0 = tf.Session()

    with sess0 as sess:
        loader = tf.train.import_meta_graph(graphfile)
        loader.restore(sess, modelPath) # restores the graph

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
        # results = tf.argmax(logits, 1)
        softmax = tf.nn.softmax(logits, 1)

        # preparing the input feed 
        images = graph0.get_tensor_by_name('images:0')
        labels = graph0.get_tensor_by_name('labels:0')
        feed_dict = {images: mnist_data.test.images, labels: mnist_data.test.labels}

        # extracting values from the graph
        e1w_val, e2w_val, e3w_val, dw_val = sess.run([e1w, e2w, e3w, dw], feed_dict=feed_dict)
        e1b_val, e2b_val, e3b_val, db_val = sess.run([e1b, e2b, e3b, db], feed_dict=feed_dict)
        softmax = sess.run(softmax, feed_dict=feed_dict)

        out = {
                "e1w": e1w_val, "e2w": e2w_val, "e3w": e3w_val, "dw": dw_val, 
                "e1b": e1b_val, "e2b": e2b_val, "e3b": e3b_val, "db": db_val, 
                "softmax": softmax
        }

        pickle.dump( out, open( picklepath + "orig_network-b{}.p".format(beta), "wb" ) )
        print("The original logits and softmax values have been pickled and saved as orig_vals-b{}.json".format(beta))

        return out
    sess.close()

def save_to_json(filename, object):
    with open(jsonpath+filename, 'w') as f:
        ujson.dump(object, f)
    print("The data has been jsonified and saved as {}.".format(filename))

def load_from_json(filename):
    with open(jsonpath+filename, 'r') as f:
        object = ujson.load(f)
    return object





#----------------------fim2.py------------------------------------------------
# now we tweak one parameter and compare the output logits
def calc_altered_softmax(parameter, e1w_val, e2w_val, e3w_val, dw_val, e1b_val, e2b_val, e3b_val, db_val):
    if parameter % 1000 == 0:
        print("parameter: {}/20001".format(parameter))
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
        softmax_mod = tf.nn.softmax(logits_mod, 1)

        ## Testing the accuracy of the model
        images1 = graph1.get_tensor_by_name('images:0')
        labels1 = graph1.get_tensor_by_name('labels:0')
        feed_dict = {images1: mnist_data.test.images, labels1: mnist_data.test.labels}

        softmax_mod = sess.run(softmax_mod, feed_dict=feed_dict)

        return softmax_mod, delta_theta
    sess.close()

def calc_score(label, parameter, out):
    softmax_mod, delta_theta = calc_altered_softmax(parameter, out["e1w"], out["e2w"], out["e3w"], out["dw"], out["e1b"], out["e2b"], out["e3b"], out["db"])
    softmax = out["softmax"]
    if delta_theta != 0:
        score = ( np.log(softmax_mod[0][label]) - np.log(softmax[0][label]) ) / delta_theta
    else:
        score = ( np.log(softmax_mod[0][label]) - np.log(softmax[0][label]) ) / 0.0001
    return score

# build the dictionary of scores
@ray.remote
def calc_all_scores(l, out):
    scores = {}
    for label in range(l,l+1): # for label = 0
        print("------------label: {}/9------------".format(label))
        for param in range(size):
            score = calc_score(label, param, out)
            scores.update({ 'l{}p{}'.format(label,param) : score })
    # output.put(scores)
    return scores


def calc_scores_all_labels(out):
    print("********************** Calculating scores from modified network *****************************")
    # run in parallel using Ray

    # create remote objects
    result_ids = []
    for i in range(10):
        result_ids.append(calc_all_scores.remote(i, out))
    
    # wait for all processes to finish
    result_ids, _ = ray.wait(result_ids, num_returns=10, timeout=None)

    # fetch results (python object) from corresponding remote object 
    scores_all_labels = {}
    for obj in result_ids:
        scores = ray.get(result_ids[obj])
        save_to_json('b{}-scores{}.json'.format(beta, i), scores)
        scores_all_labels = {**scores_all_labels, **scores}

    # save the collection of scores
    save_to_json('scores_all_labels-b{}.json'.format(beta), scores_all_labels)
    
    return scores_all_labels





#----------------------fim3.py------------------------------------------------
# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
# using structured numpy arrays instead of dictionaries to reduce memory usage
def calc_score_pairs(scores, softmax, label):
    print('----- calculating score pairs for label = {} -----'.format(label))
    score_pairs = np.zeros((size, size))
    tracker = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if tracker[j][i] == 1:
                score_pairs[i][j] = score_pairs[j][i]
            else:
                score_pairs[i][j] = scores['l{}p{}'.format(label, i)] * scores['l{}p{}'.format(label, j)] * softmax[0][label]
                tracker[i][j] = 1
    print('The size of the array for label={} is {} bytes'.format(label, score_pairs.nbytes))
    return score_pairs
# since sp(i,j|l) = sp(j,i|l) (ie symmetric square matrix) the if cond removes 10*20002 unnecessary calculations


def calculate_fim(scores, softmax):
    print("********************** Calculating the Fisher Information Matrix *****************************")
    # combining score pairs (weighted by softmax label values) to get the FIM 
    fim = calc_score_pairs(scores, softmax, 0)
    for i in range(1,10):
        mat = calc_score_pairs(scores, softmax, i)
        fim = np.add(fim, mat)
        print(psutil.virtual_memory())
    return fim


def export_fim(matrix):
    np.save(npypath+'fim-b{}.npy'.format(beta), matrix)
    print("The FIM matrix has been exported as fim.npy")





#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    # get computer stats
    num_cores = mp.cpu_count()
    print('This kernel has {} cores.'.format(num_cores))
    print(psutil.virtual_memory())

    # fim1.py
    out = calc_original()

    # fim2.py
    ray.init()
    scores = calc_scores_all_labels(out)
    softmax = out['softmax']

    # fim3.py
    fim = calculate_fim(scores, softmax)
    export_fim(fim)

# After running this script, we should have the following files:
# FIM
# ├───/plots
# ├───/DATA
# ├───/npy
# │   └───fim-b*.npy
# ├───/json
# │   ├───b*-scores*.json
# │   └───scores_all_labels-b*.json
# ├───/pickle
# │   └───orig_network-b*.p
# ├───fimtotal.py
# └───fim4.py


