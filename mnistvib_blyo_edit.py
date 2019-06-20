#%%
## importing packages
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
print('All libraries imported')


#%%
## Initialising graph
tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


#%%
## Importing data -- expect errors about soon-to-be-deprecated packages

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./DATA/mnistdata', validation_size=0)
print('All data imported')

images = tf.placeholder(tf.float32, [None, 784], 'images')
labels = tf.placeholder(tf.int64, [None], 'labels')
one_hot_labels = tf.one_hot(labels, 10)


#%%
## defining the network

layers = tf.contrib.layers
# ds = tf.contrib.distributions ## updated to:
ds = tfp.distributions

def encoder(images):
    net = layers.relu(2*images-1, 1024)
    net = layers.relu(net, 1024)
    params = layers.linear(net, 512)
    mu, rho = params[:, :256], params[:, 256:]
#     encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0) ## updated to:
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

BETA = 1e-3

info_loss = tf.reduce_sum(tf.reduce_mean(
    ds.kl_divergence(encoding, prior), 0)) / math.log(2)

total_loss = class_loss + BETA * info_loss

accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(logits, 1), labels), tf.float32))
avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(
    tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
IZY_bound = math.log(10, 2) - class_loss
IZX_bound = info_loss 

#%%
batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)

#%%
## defining the training process and model/data extraction 
# global_step = tf.contrib.framework.get_or_create_global_step() ## updated to:
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

#%%
tf.global_variables_initializer().run()

#%%
def evaluate():
    IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                             feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
    return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc

#%%
## Training occurs here -- skip this cell if you want to open pre-existing checkpoint files
print('Training beginning...')
for epoch in range(200):
    for step in range(steps_per_batch):
        im, ls = mnist_data.train.next_batch(batch_size)
        sess.run(train_tensor, feed_dict={images: im, labels: ls})
    print("{}: IZY={:.2f}\tIZX={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
        epoch, *evaluate()))
    sys.stdout.flush()
print('Training complete')
savepth = saver.save(sess, './DATA/mnistvib', global_step) ## data is stored here
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
######### (SKIP TO HERE IF VIEWING OR EXTRACTING DATA FILES) ##########
## data extraction 
modelPath = './DATA/mnistvib.ckpt-120000' # trained model (graph/meta, index, data/weights)
graphfile = './DATA/mnistvib.ckpt-120000.meta' # the model used in training
saver.restore(sess, modelPath)
evaluate()


#%%
## Check the operations (nodes), all variables, or trainable variables in the graph (model)
saver2 = tf.train.import_meta_graph(graphfile)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
    
## Inspect all variable names:
with tf.Session() as sess:
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
## Save everything (including weights) into text file:
## pbtxt format is not great for manually parsing data but can be useful in reloading data into the graph
log_dir = "./log_dir/"
out_file = "train.pbtxt"
tf.train.write_graph(input_graph_def, logdir=log_dir, name=out_file, as_text=True)


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
output_path = os.path.join(log_dir, "decoder_values.json")
with open(output_path, 'w') as json_file:
  json_file.write(tensor_dict)


#%%
