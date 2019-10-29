'''
This is for the "huge" model that has large epoch = 10000 (compared to 1000 for the "l" normal tests)

'''

#%%
# START - SET VARS
dataset = "test" # test or train
h1 = 100
h2 = h1
bneck = 32 # = 2K
filename = 'h'+str(h1)+'b'+str(bneck)

## importing packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
print('All libraries imported')

# import dataset
import small_mnist_full_offline as small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')

#%%
for beta_num in range(0,13): # range(0, 13) results in beta = 1e-0 to 1e-12
    print("------ New beta: beta=1e-{}, dataset={} ------".format(beta_num, dataset))

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

    # Normal network:
    def encoder(images):
        net = layers.relu(2*images-1, h1)
        net = layers.relu(net, h2)
        params = layers.linear(net, bneck)
        mu, rho = params[:, :int(bneck/2)], params[:, int(bneck/2):]
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
    steps_per_batch = int(mnist_data.train.num_examples / batch_size) # should be TRAIN always
    ########## don't change

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

    def evaluate():
        if dataset == "test":
            IZY, IZX, acc, avg_acc, tot_loss = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy, total_loss],
                                feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
        elif dataset == "train":
            IZY, IZX, acc, avg_acc, tot_loss = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy, total_loss],
                                feed_dict={images: mnist_data.train.images, labels: mnist_data.train.labels})
        else:
            print("dataset value should either be 'test' or 'train'.")
        return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc, tot_loss


    ## Training occurs here
    csvname = "./data-csv/full-{}-beta{}-{}.csv".format(filename, beta_num, dataset)
    dataname = './data-model/{}-beta{}-{}'.format(filename, beta_num, dataset)

    print('Training beginning for beta=1e-{} and the {} dataset.'.format(beta_num, dataset))
    with open(csvname, "a") as f:
        for epoch in range(1000):
            for step in range(steps_per_batch):
                im, ls = mnist_data.train.next_batch(batch_size)
                sess.run(train_tensor, feed_dict={images: im, labels: ls})
            print(epoch)
            print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}".format(
                epoch, *evaluate()), file=f)
            sys.stdout.flush()
    f.close()
    print('Training complete for beta=1e-{} and the {} dataset.'.format(beta_num, dataset))
    savepth = saver.save(sess, dataname, global_step) ## data is stored here
    print('Checkpoint saved')

    sess.close()

#%%












