#%%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
print('All libraries imported')

#%%
tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)

#%%
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('/tmp/mnistdata', validation_size=0)
print('All data imported')

# the above cell returns errors so use this cell instead:
# mnist = tf.keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

#%%
images = tf.placeholder(tf.float32, [None, 784], 'images')
labels = tf.placeholder(tf.int64, [None], 'labels')
one_hot_labels = tf.one_hot(labels, 10)

#%%
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

#%%
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

#%%
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
# global_step = tf.contrib.framework.get_or_create_global_step() ## updated to:
global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                           decay_steps=2*steps_per_batch,
                                           decay_rate=0.97, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate, 0.5)

ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)

# model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) ## the same as the `vars` variable below.
# https://www.tensorflow.org/api_docs/python/tf/GraphKeys

vars = tf.model_variables() # we want to extract the trainable variables (weights and biases)
ma_update = ma.apply(vars)

saver = tf.train.Saver()
saver_polyak = tf.train.Saver(ma.variables_to_restore()) 
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

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
print('Training beginning...')
for epoch in range(200):
    for step in range(steps_per_batch):
        im, ls = mnist_data.train.next_batch(batch_size)
        sess.run(train_tensor, feed_dict={images: im, labels: ls})
    print("{}: IZY={:.2f}\tIZX={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
        epoch, *evaluate()))
    sys.stdout.flush()
print('Training complete')
savepth = saver.save(sess, '/tmp/mnistvib.ckpt', global_step)
print('Checkpoint saved')

#%%
saver_polyak.restore(sess, savepth)
evaluate()

#%%
saver.restore(sess, savepth)
# https://www.tensorflow.org/guide/saved_model
evaluate()


#%%
modelPath = './DATA/mnistvib.ckpt-120000' # trained model (graph/meta, index, data/weights)
graphfile = './DATA/mnistvib.ckpt-120000.meta' # the model used in training
saver.restore(sess, modelPath)
evaluate()


#%%
## Check the operations (nodes), all variables, or trainable variables in the graph; OR even save everything, including the weights into a text file so that you can read them.
saver2 = tf.train.import_meta_graph(graphfile)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
    
with tf.Session() as sess:
    saver.restore(sess, modelPath)

    # Check all operations (nodes) in the graph:
    # print("## All operations: ")
    # for op in graph.get_operations():
    #     print(op.name)

    # OR check all variables in the graph:
    # print("## All variables: ")
    # for v in tf.global_variables():
    #     print(v.name)

    # OR check all trainable variables in the graph:
    print("## Trainable variables: ")
    for v in tf.trainable_variables():
        print(v.name)


#%%
    log_dir = "./log_dir/"
    out_file = "train.json"
    tf.train.write_graph(input_graph_def, logdir=log_dir, name=out_file, as_text=True)




#%%
## Inspect all tensors and their weight values:
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(modelPath)
var_to_shape_map = reader.get_variable_to_shape_map()
    
for key in sorted(var_to_shape_map):
    termsInOneArray = reader.get_tensor(key)[0].size
    totalNumOfTerms = reader.get_tensor(key).size
    numOfArrays = int(totalNumOfTerms / termsInOneArray)
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
    print("num of terms in one array: ", termsInOneArray)
    print("total num of terms: ", totalNumOfTerms)
    print("num of arrays: ", numOfArrays)
    print("\n")


#%%
## log tensors and weight values to file
import json

## turning the output above into a large string:
tensor_values = {}
for key in sorted(var_to_shape_map):
    key_str = str(key)
    if ("decoder/fully_connected/" in key_str):
        tensor_val = reader.get_tensor(key).tolist()
        tensor_values[key_str] = tensor_val
        # print(key, "\n", tensor_val, "\n \n")
    else:
        break
    # print(type(tensor_val))
    # tensor_values.update( key_str: tensor_val )

# print(tensor_values)


#%%
# write tensor values to file
tensor_dict = json.dumps(tensor_values)
with open('decoder_values.json', 'w') as json_file:
  json_file.write(tensor_dict)
    



#%%
