#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
print('All libraries imported')

# import dataset
from SMNN import small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')

## Initialising graph
tf.reset_default_graph()

# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)

images = tf.placeholder(tf.float32, [None, 64], 'images') # set to total number of pixels in input
labels = tf.placeholder(tf.int64, [None], 'labels')
one_hot_labels = tf.one_hot(labels, 10)

## defining the network
layers = tf.contrib.layers
ds = tfp.distributions

# Normal sized network
def encoder(images):
    net = layers.relu(2*images-1, 1024)
    net = layers.relu(net, 1024)
    params = layers.linear(net, 512)
    mu, rho = params[:, :256], params[:, 256:]
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

# BETA = float("1e-"+str(beta_num))
BETA = 1e-3

total_loss = class_loss + BETA * info_loss

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
IZY_bound = math.log(10, 2) - class_loss
IZX_bound = info_loss 

batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)
##################### don't change this

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

########################################## Test vs Train data is utilised here
def evaluate():
    IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                            feed_dict={images: mnist_data.train.images, labels: mnist_data.train.labels})
    return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc

#%%
# generating values of IZY, IZX, error, accuracy for each beta value, then collecting all of them into one file

# VARS
dataset = "train"
filename = "SMNN"

output_file = "./"+filename+"/csv-output/last/"+dataset+"/last-"+dataset+"-ALL.csv"

with open(output_file, "a") as b:
    print("beta, IZY, IZX, acc, avg_acc, err, avg_err", file=b)
    for beta_num in range(13):
        beta = "1.E-"+str(beta_num)
        print("-----------------------------------\nAcquiring values for beta =", beta)

        # the model used in training
        modelPath = './'+filename+'/DATA/'+dataset+'/'+filename+'-beta'+str(beta_num)+'-'+dataset+'-15000' # trained model (graph/meta, index, data/weights)
        graphfile = modelPath+'.meta' 

        (cizy, cizx, cacc, cavg, cerr, cavgerr) = (0,0,0,0,0,0)
        tot = 50 # number of samples of the last(max) epoch values taken per beta value
        for i in range(tot):
            saver_polyak.restore(sess, modelPath)
            (tizy, tizx, tacc, tavg, terr, tavgerr) = evaluate()
            cizy += tizy
            cizx += tizx
            cacc += tacc
            cavg += tavg
            cerr += terr
            cavgerr += tavgerr
        
        # now average the final values
        cizy /= tot
        cizx /= tot
        cacc /= tot
        cavg /= tot
        cerr /= tot
        cavgerr /= tot

        # for each beta, write each set of values to separate file
        with open("./"+filename+"/csv-output/last/"+dataset+"/last-beta"+str(beta_num)+"-"+dataset+".csv", "a") as h:
            print("{},{},{},{},{},{}".format(cizy, cizx, cacc, cavg, cerr, cavgerr), file=h)
        
        # collect all of the values in one file
        print("{},{},{},{},{},{},{}".format(beta, cizy, cizx, cacc, cavg, cerr, cavgerr), file=b)
        h.close()
b.close()
print('All values averaged')


#%%
import csv
with open("./"+filename+"/csv-output/last/"+dataset+"/last-"+dataset+"-ALL.csv", "a") as b:
    print("beta, IZY, IZX, acc, avg_acc, err, avg_err", file=b)
    for beta_num in range(13):
        beta = "1.E-"+str(beta_num)
        with open("./"+filename+"/csv-output/last/"+dataset+"/last-beta"+str(beta_num)+"-"+dataset+".csv") as h:
            reader = csv.reader(h, delimiter=',')
            for row in reader:
                print("{},{},{},{},{},{},{}".format(beta, row[0], row[1], row[2], row[3], row[4], row[5]), file=b)
print("Gathered all files into one file")


