## libraries
import numpy as np
import tensorflow as tf
import pickle
import multiprocessing as mp
import ray
import psutil
import nujson as ujson
import timing
from memory_profiler import profile


### global variables ###
# parameters -----------
weight_change = 0.01
beta = 12
size = 10

# paths
modelPath = './DATA/h100b32-beta{}-test-15000'.format(beta)
graphfile = modelPath + '.meta'
outpath = './output/beta{}/'.format(beta)
npypath = outpath
jsonpath = outpath
picklepath = outpath

# dataset
import small_mnist_one_image_noskdl as small_mnist
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
@ray.remote(num_cpus=1)
class Scoring:
    def __init__(self, original, label):
        self.e1w = original['e1w']
        self.e1b = original['e1b']
        self.e2w = original['e2w']
        self.e2b = original['e2b']
        self.e3w = original['e3w']
        self.e3b = original['e3b']
        self.dw  = original['dw']
        self.db  = original['db']
        self.softmax = original['softmax']

        self.label = label
        self.parameter = None
        self.score = None
        self.scores = {}


    # Calculate modified 
    def calc_altered_softmax(self):
        if self.parameter % 100 == 0:
            print("parameter: {}/20001".format(self.parameter))

        tf.reset_default_graph()

        graph1 = tf.Graph()
        with graph1.as_default():
            sess1 = tf.Session()

        with sess1 as sess:
            loader = tf.train.import_meta_graph(graphfile)
            loader.restore(sess, modelPath)

            if self.parameter in range(0,6400): 
                # e1w, 64 by 100
                signifier = 'w'
                layer_index = self.parameter - 0
                yrange = 100
                var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected/weights:0')
                modified = np.copy(self.e1w)
                original = np.copy(self.e1w)
            elif self.parameter in range(6400, 6500):
                # e1b, 100
                signifier = 'b'
                layer_index = self.parameter - 6400
                var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected/biases:0')
                modified = np.copy(self.e1b)
                original = np.copy(self.e1b)
            elif self.parameter in range(6500, 16500):
                # e2w, 100 by 100
                signifier = 'w'
                layer_index = self.parameter - 6500
                yrange = 100
                var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_1/weights:0')
                modified = np.copy(self.e2w)
                original = np.copy(self.e2w)
            elif self.parameter in range(16500, 16600):
                # e2b, 100
                signifier = 'b'
                layer_index = self.parameter - 16500
                var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_1/biases:0')
                modified = np.copy(self.e2b)
                original = np.copy(self.e2b)
            elif self.parameter in range(16600, 19800):
                # e3w, 100 by 32
                signifier = 'w'
                layer_index = self.parameter - 16600
                yrange = 32
                var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_2/weights:0')
                modified = np.copy(self.e3w)
                original = np.copy(self.e3w)
            elif self.parameter in range(19800, 19832):
                # e3b, 32
                signifier = 'b'
                layer_index = self.parameter - 19800
                var_pre_mod = graph1.get_tensor_by_name('encoder/fully_connected_2/biases:0')
                modified = np.copy(self.e3b)
                original = np.copy(self.e3b)
            elif self.parameter in range(19832, 19992):
                # dw, 16 by 10
                signifier = 'w'
                layer_index = self.parameter - 19832
                yrange = 10
                var_pre_mod = graph1.get_tensor_by_name('decoder/fully_connected/weights:0')
                modified = np.copy(self.dw)
                original = np.copy(self.dw)
            elif self.parameter in range(19992, 20002):
                # db, 10
                signifier = 'b'
                layer_index = self.parameter - 19992
                var_pre_mod = graph1.get_tensor_by_name('decoder/fully_connected/biases:0')
                modified = np.copy(self.db)
                original = np.copy(self.db)

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

    # Calculate score from modified network
    def calc_score(self):
        softmax_mod, delta_theta = self.calc_altered_softmax()

        if delta_theta != 0:
            self.score = ( np.log(softmax_mod[0][self.label]) - np.log(self.softmax[0][self.label]) ) / delta_theta
        else:
            self.score = ( np.log(softmax_mod[0][self.label]) - np.log(self.softmax[0][self.label]) ) / 0.0001


    # Iterate over all 20002 parameters
    def calc_all_scores(self):
        print("------------label: {}/9------------".format(self.label))

        for param in range(size):
            self.parameter = param
            self.calc_score()
            self.scores.update({ 'l{}p{}'.format(self.label,self.parameter) : self.score })

        return self.scores
# end of class

def calc_scores_all_labels(out):
    print("********************** Calculating scores from modified network *****************************")

    class_ids = [Scoring.remote(out, i) for i in range(10)]
    obj_ids = [c.calc_all_scores.remote() for c in class_ids]
    ready_ids, _ = ray.wait(obj_ids, num_returns=10, timeout=None)

    scores_all_labels = {}
    for r in ready_ids:
        scores = ray.get(r)
        scores_all_labels = {**scores_all_labels, **scores}

    save_to_json('scores_all_labels-b{}.json'.format(beta), scores_all_labels)
    
    return scores_all_labels

#----------------------fim3.py------------------------------------------------
# build the matrix by calculating score(i, label) * score(j, label) for all pairs of parameters i,j
# using structured numpy arrays instead of dictionaries to reduce memory usage

@ray.remote(num_cpus=1)
class Pairing:
    def __init__(self, scores, label, softmax):
        self.score_pairs = np.zeros((size, size))
        self.tracker = np.zeros((size, size))
        self.scores = scores
        self.label = label
        self.softmax = softmax

    def calc_score_pairs(self):
        for i in range(size):
            for j in range(size):
                if self.tracker[j][i] == 1:
                    self.score_pairs[i][j] = self.score_pairs[j][i]
                else:
                    self.score_pairs[i][j] = self.scores['l{}p{}'.format(self.label, i)] \
                                             * self.scores['l{}p{}'.format(self.label, j)] \
                                             * self.softmax[0][self.label]
                    self.tracker[i][j] = 1
        print('The size of the array for label={} is {} bytes'.format(self.label, self.score_pairs.nbytes))
        return self.score_pairs


@ray.remote(num_cpus=1)
def add(x,y):
    return np.add(x,y)

def calculate_fim(scores, softmax):
    print("********************** Calculating the Fisher Information Matrix *****************************")

    # combining score pairs (weighted by softmax label values) to get the FIM 
    class_ids = [Pairing.remote(scores, i, softmax) for i in range(10)]
    obj_ids = [c.calc_score_pairs.remote() for c in class_ids]

    ready_ids, _ = ray.wait(obj_ids, num_returns=10, timeout=None)

    # add the 10 np arrays in log time using a tree structured pattern
    while len(ready_ids) > 1:
        ready_ids = ready_ids[2:] + [add.remote(ready_ids[0], ready_ids[1])]

    fim = ray.get(ready_ids[0])

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
    ray.init(object_store_memory=40 * 1000000000)
    scores = calc_scores_all_labels(out)

    # fim3.py
    fim = calculate_fim(scores, out['softmax'])
    export_fim(fim)

# After running this script, we should have the following files:
# FIM
# ├───/plots
# ├───/DATA
# ├───/output
# │   └───/beta*
# │       ├───fim-b*.npy
# │       ├───scores_all_labels-b*.json
# │       └───orig_network-b*.p
# ├───fimtotal_ray_classes.py
# └───fim4.py


