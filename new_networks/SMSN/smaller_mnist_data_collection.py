'''
The same as mnistvib_blyo_edit_with_var.py but with reduced hidden layer sizes.
The number of hidden layer nodes = 100, size of bottleneck is K = 16.
'''

#%%
# START - SET VARS
n = 4
beta_num = 9
# test or train
dataset = "test"

print('n='+str(n)+', beta=1e-'+str(beta_num)+', dataset='+dataset)

## importing packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
print('All libraries imported')
print('If you want to extract or view data from model, skip the code cell containing the training.')

## Initialising graph
tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


from tensorflow.python.framework import dtypes

# Defining the dataset class
class Dataset(object):

    def __init__(self, images, labels, dtype=dtypes.float32):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
            dtype)

        assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


## Importing 8x8 mnist data from scikit-learn

# required for naming the train and test datasets below
import collections
namedDataset = collections.namedtuple('Dataset', ['train', 'test'])

def read_data_sets():

    # data from the scikit-learn package
    digits = datasets.load_digits() # 8x8 images of digits

    # training and test sets 
    X_digits = digits.data / digits.data.max() # normalises 16-level sensitivity to 0to1 
    y_digits = digits.target

    # In the tensorflow-approved formats
    X_digits = np.array(X_digits, dtype='float32')
    y_digits = np.uint8(y_digits)

    # parameters
    n_samples = len(X_digits)
    # print(n_samples)
    train_size = int(.85 * n_samples) # determines the size of the training set
    buff = math.ceil(n_samples/10) # size of the buffer 

    # using different parts of the training data:
    first = n * buff
    last = n * buff + train_size
    if last > n_samples:
        print("the train set extends over the n_sample size")
        train_images = np.concatenate((X_digits[first:n_samples], X_digits[:last-n_samples]), axis=0)
        train_labels = np.concatenate((y_digits[first:n_samples], y_digits[:last-n_samples]), axis=0)

        test_images = X_digits[last-n_samples:first]
        test_labels = y_digits[last-n_samples:first]
    else:
        print("the train set is within the n_sample size")
        train_images = X_digits[first:last]
        train_labels = y_digits[first:last]

        test_images = np.concatenate((X_digits[last:n_samples], X_digits[:first]), axis=0)
        test_labels = np.concatenate((y_digits[last:n_samples], y_digits[:first]), axis=0)

    # images and labels for training and testing
    # train_images = X_digits[:train_size]
    # print("train images size", train_images.shape)
    # train_labels = y_digits[:train_size]
    # test_images = X_digits[train_size:]
    # print("test images size", test_images.shape)
    # test_labels = y_digits[train_size:]

    print("size of train_images:", train_images.shape)
    print("size of train_labsls:", train_labels.shape)

    print("size of test_images:", test_images.shape)
    print("size of test_labels:", test_labels.shape)

    train = Dataset(train_images, train_labels)
    test = Dataset(test_images, test_labels)

    return namedDataset(train=train, test=test)

def load_8x8_mnist():
    return read_data_sets()
    

# x = np.array([0,1,2,3,4,5,6,7,8,9])
# print("x[0:3]:", x[0:3])
# print("x[3:6]:", x[3:6])
# first = 8
# last = 11
# if last > x.shape[0]:
#     train = np.append(x[first:len(x)], x[:last-len(x)])
#     print(train)


mnist8x8 = load_8x8_mnist()
print('All data imported')


# placeholders for images and labels
images = tf.placeholder(tf.float32, [None, 64], 'images') # input is 8x8 = 64
labels = tf.placeholder(tf.int64, [None], 'labels')
one_hot_labels = tf.one_hot(labels, 10)


## defining the network
layers = tf.contrib.layers
ds = tfp.distributions

# encoder is 2 hidden layers and the bottleneck layer
def encoder(images):
    net = layers.relu(2*images-1, 100) # hidden layers of size 100
    net = layers.relu(net, 100)
    params = layers.linear(net, 32) # 2K = 32
    mu, rho = params[:, :16], params[:, 16:]
    encoding = ds.Normal(mu, tf.nn.softplus(rho - 5.0))
    return encoding


def decoder(encoding_sample):
    net = layers.linear(encoding_sample, 10)
    return net

prior = ds.Normal(0.0, 1.0)

with tf.variable_scope('encoder'):
    encoding = encoder(images)
    
with tf.variable_scope('decoder'):
    logits = decoder(encoding.sample())

with tf.variable_scope('decoder', reuse=True):
    many_logits = decoder(encoding.sample(12))

class_loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=one_hot_labels) / math.log(2)

info_loss = tf.reduce_sum(tf.reduce_mean(
    ds.kl_divergence(encoding, prior), 0)) / math.log(2)

BETA = float("1e-"+str(beta_num))

total_loss = class_loss + BETA * info_loss

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
IZY_bound = math.log(10, 2) - class_loss
IZX_bound = info_loss 


batch_size = 100
steps_per_batch = int(mnist8x8.train.num_examples / batch_size) # 15??? = 1034 (not 600 like normal mnist)


## defining the training process and model/data extraction 
global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                           decay_steps=2*steps_per_batch,
                                           decay_rate=0.97, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate, 0.5)

ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)

vars = tf.model_variables()
ma_update = ma.apply(vars)

## saves the model/data
saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                   global_step,
                                                   update_ops=[ma_update])


tf.global_variables_initializer().run()

# TEST VS TRAIN USED HERE
def evaluate():
    IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                             feed_dict={images: mnist8x8.test.images, labels: mnist8x8.test.labels})
    return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc


#%%
## Training occurs here
## Skip this cell if you want to open pre-existing checkpoint files

# NAME
csvname = "./csv-output/full/full-buff"+str(n)+"-beta"+str(beta_num)+"-"+dataset+".csv"
dataname= "./DATA/mnistvib-8x8/"+dataset+"/mnist-8x8-buff"+str(n)+"-beta"+str(beta_num)+"-"+dataset


print('Training beginning...')
with open(csvname, "a") as f:
    for epoch in range(1000):
        for step in range(steps_per_batch):
            im, ls = mnist8x8.train.next_batch(batch_size)
            sess.run(train_tensor, feed_dict={images: im, labels: ls})
        print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            epoch, *evaluate()), file=f)
        # IZY, IZX, acc, avg-acc, err, avg-err
        sys.stdout.flush()
f.close()
print('Training complete for n='+str(n)+ ', beta='+str(beta_num)+' and the '+dataset+' dataset.')
savepth = saver.save(sess, dataname, global_step) ## data is stored here
print('Checkpoint saved')


#%%
## accuracy data applied with Exponential Moving Average
saver_polyak.restore(sess, savepth)
evaluate()

#%%
## accuracy data without EMA
saver.restore(sess, savepth)
evaluate()

#%%
## data extraction 
modelPath = './DATA/mnist-8x8-15000' # trained model (graph/meta, index, data/weights)
graphfile = './DATA/mnist-8x8-15000.meta' # the model used in training
saver_polyak.restore(sess, modelPath)
evaluate()

#%%
# ## data extraction 
# import csv
# modelPath = './DATA/mnist-8x8-15000' # trained model (graph/meta, index, data/weights)
# graphfile = './DATA/mnist-8x8-15000.meta' # the model used in training
# with open("./csv-output/buff9-beta2-epoch1000-test-LAST.csv", "a") as g:
#     print("", file=g)
#     for i in range(50):
#         saver_polyak.restore(sess, modelPath)
#         print("{},{},{},{},{},{}".format(*evaluate()), file=g) 
#         # IZY, IZX, acc, avg-acc, err, avg-err
    
# g.close()

#%%
## average out the last values
# VARS
n = 0
beta_num = 1
dataset = "train"

for beta_num in range(15):
    beta_num += 1
    modelPath = "./DATA/mnistvib-8x8/"+dataset+"/mnist-8x8-buff"+str(n)+"-beta"+str(beta_num)+"-"+dataset+"-15000"
    graphfile = modelPath+".meta" # the model used in training
    (cizy, cizx, cacc, cavg, cerr, cavgerr) = (0,0,0,0,0,0)
    tot = 50
    for i in range(tot):
        saver_polyak.restore(sess, modelPath)
        (tizy, tizx, tacc, tavg, terr, tavgerr) = evaluate()
        cizy += tizy
        cizx += tizx
        cacc += tacc
        cavg += tavg
        cerr += terr
        cavgerr += tavgerr
    cizy /= tot
    cizx /= tot
    cacc /= tot
    cavg /= tot
    cerr /= tot
    cavgerr /= tot
    with open("./csv-output/last/"+dataset+"/last-buff"+str(n)+"-beta"+str(beta_num)+"-"+dataset+".csv", "a") as h:
        print("{},{},{},{},{},{}".format(cizy, cizx, cacc, cavg, cerr, cavgerr), file=h)
    h.close()
print('All last')


#%%
## Check the operations (nodes), all variables, or trainable variables in the graph (model)
saver2 = tf.train.import_meta_graph(graphfile)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

#%%
## Inspect all variable names:
with tf.Session() as sess2:
    saver.restore(sess, modelPath)

    ## Check all operations (nodes) in the graph:
    # print("## All operations: ")
    # for op in graph.get_operations():
    #     print(op.name)

    ## OR check all variables in the graph:
    # print("## All variables: ")
    # for v in tf.global_variables():
    #     print(v.name)

    ## OR check all trainable variables in the graph (weights and biases are here):
    print("## Trainable variables: ")
    for v in tf.trainable_variables():
        print(v.name)


#%%
## Inspect all tensors and their weight values, along with tensor size:
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(modelPath)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in sorted(var_to_shape_map):
    try:
        termsInOneArray = reader.get_tensor(key)[0].size
        totalNumOfTerms = reader.get_tensor(key).size
        numOfArrays = int(totalNumOfTerms / termsInOneArray)
        print("tensor_name: ", key)
        print(reader.get_tensor(key))
        print("num of terms in one array: ", termsInOneArray)
        print("total num of terms: ", totalNumOfTerms)
        print("num of arrays: ", numOfArrays) ## number of arrays should match 784 - 1024 - 1024 - 256
        print("\n")
    except:
        print("tensor_name: ", key)
        print(reader.get_tensor(key))

#%%
## Another quick way to check tensor size:
print(vars)


#%%
## Save graph to text file:
## pbtxt format is useful for reloading data into the graph
saver2 = tf.train.import_meta_graph(graphfile)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
log_dir = "./log_dir/"
out_file = "train_reduced100.pbtxt"
tf.train.write_graph(input_graph_def, logdir=log_dir, name=out_file, as_text=True)
print('graph saved to text file')

#%%
## Save decoder tensors to file -- if statement can be removed to save all tensors
tensor_values = {}
for key in sorted(var_to_shape_map):
    key_str = str(key)
    if ("decoder/fully_connected/" in key_str):
        tensor_val = reader.get_tensor(key).tolist()
        tensor_values[key_str] = tensor_val
    else:
        break
    ## tensor_values.update( key_str: tensor_val )

import os

tensor_dict = json.dumps(tensor_values)
output_path = os.path.join(log_dir, "decoder_values_reduced100.json")
with open(output_path, 'w') as json_file:
  json_file.write(tensor_dict)
print('decoder values saved to text file')


#%%
