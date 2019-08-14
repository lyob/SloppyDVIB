'''
The same as mnistvib_blyo_edit_with_var.py but with reduced hidden layer sizes.
The number of hidden layer nodes = 100, size of bottleneck is K = 16.
'''

#%%
# START - SET VARS
dataset = "train" # test or train
filename = 'NMSN'

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
# print('If you want to extract or view data from model, skip the code cell containing the training.')

## Importing data
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./DATA/mnistdata', validation_size=0)
print('Full MNIST data imported')

#%%
for beta_num in range(0, 13): # range(0, 13) results in beta = 1e-0 to 1e-12
    print("--------------------------------------------------------\n\
New beta: beta=1e-"+str(beta_num)+', dataset='+dataset)

    ## Initialising graph
    tf.reset_default_graph()

    # Turn on xla optimization
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)

    images = tf.placeholder(tf.float32, [None, 784], 'images')
    labels = tf.placeholder(tf.int64, [None], 'labels')
    one_hot_labels = tf.one_hot(labels, 10)

    ## defining the network
    layers = tf.contrib.layers
    ds = tfp.distributions

    def encoder(images):
        net = layers.relu(2*images-1, 100)
        net = layers.relu(net, 100)
        params = layers.linear(net, 32)
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
    steps_per_batch = int(mnist_data.train.num_examples / batch_size) # = 600

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

    # Test vs Train data is utilised here
    def evaluate():
        IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                                feed_dict={images: mnist_data.train.images, labels: mnist_data.train.labels})
        return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc


    ## Training occurs here
    csvname = "./"+filename+"/csv-output/full/full-NMSN-beta"+str(beta_num)+'-'+dataset+'.csv'
    dataname = './'+filename+'/DATA/'+dataset+'/NMSN-beta'+str(beta_num)+'-'+dataset

    print('Training beginning for beta=1e-'+str(beta_num)+' and the '+dataset+' dataset.')
    with open(csvname, "a") as f:
        for epoch in range(200):
            for step in range(steps_per_batch):
                im, ls = mnist_data.train.next_batch(batch_size)
                sess.run(train_tensor, feed_dict={images: im, labels: ls})
            print(epoch)
            print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                epoch, *evaluate()), file=f)
            sys.stdout.flush()
    f.close()
    print('Training complete for beta=1e-'+str(beta_num)+' and the '+dataset+' dataset.')
    savepth = saver.save(sess, dataname, global_step) ## data is stored here
    print('Checkpoint saved')

    sess.close()


#%%
## accuracy data applied with Exponential Moving Average
saver_polyak.restore(sess, savepth)
evaluate()

## accuracy data without EMA
saver.restore(sess, savepth)
evaluate()


#%%
## average out the last values
# VARS
beta_num = 0
dataset = "test"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
print('All libraries imported')

## Importing data
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./DATA/mnistdata', validation_size=0)
print('Full MNIST data imported')

## Initialising graph
tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)

images = tf.placeholder(tf.float32, [None, 784], 'images')
labels = tf.placeholder(tf.int64, [None], 'labels')
one_hot_labels = tf.one_hot(labels, 10)

## defining the network
layers = tf.contrib.layers
ds = tfp.distributions

def encoder(images):
    net = layers.relu(2*images-1, 100)
    net = layers.relu(net, 100)
    params = layers.linear(net, 32)
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
steps_per_batch = int(mnist_data.train.num_examples / batch_size) # = 600

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

# Test vs Train data is utilised here
def evaluate():
    IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                            feed_dict={images: mnist_data.train.images, labels: mnist_data.train.labels})
    return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc

#%%

with open("./filename/csv-output/last/"+dataset+"/last-beta"+str(beta_num)+"-"+dataset+".csv", "a") as b:
    print("beta, IZY, IZX, acc, avg_acc, err, avg_err", file=b)
    for beta_num in range(13):
        beta = "1.E-"+str(beta_num)

        modelPath = './NMSN/DATA/'+dataset+'/NMSN-beta'+str(beta_num)+'-'+dataset+'-120000' # trained model (graph/meta, index, data/weights)
        graphfile = modelPath+'.meta' # the model used in training

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
        with open("./NMSN/csv-output/last/"+dataset+"/last-beta"+str(beta_num)+"-"+dataset+".csv", "a") as h:
            print("{},{},{},{},{},{}".format(cizy, cizx, cacc, cavg, cerr, cavgerr), file=h)
        print("{},{},{},{},{},{},{}".format(beta, cizy, cizx, cacc, cavg, cerr, cavgerr), file=b)
        h.close()
print('All values averaged')


#%%
import csv
with open("./NMSN/csv-output/last/"+dataset+"/last-"+dataset+"-ALL.csv", "a") as b:
    print("beta, IZY, IZX, acc, avg_acc, err, avg_err", file=b)
    for beta_num in range(13):
        beta = "1.E-"+str(beta_num)
        with open("./NMSN/csv-output/last/"+dataset+"/last-beta"+str(beta_num)+"-"+dataset+".csv") as h:
            reader = csv.reader(h, delimiter=',')
            for row in reader:
                print("{},{},{},{},{},{},{}".format(beta, row[0], row[1], row[2], row[3], row[4], row[5]), file=b)
print("Gathered all files into one file")
        




#%%
# trained model (graph/meta, index, data/weights)
beta_num = 0
dataset="test"
modelPath = './NMSN/DATA/'+dataset+'/NMSN-beta'+str(beta_num)+'-'+dataset+'-120000' # trained model (graph/meta, index, data/weights)
graphfile = modelPath+'.meta' # the model used in training
saver.restore(sess, modelPath)
evaluate()



#%%
