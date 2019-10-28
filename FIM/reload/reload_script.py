#%%
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
#%%
# dataset
import small_mnist_full_offline as small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')

# global variables
modelname = 'iterated'
h1 = 100
h2 = h1
bneck = 32
basepath = '.'
epoch_num = 100
batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)
# beta_num = 0
#%%
for beta_num in range(0,2):
    if beta_num == 0:
        print('Making new')
        brand_new = True

        images = tf.placeholder(tf.float32, [None, 64], 'images') # set to total number of pixels in input
        labels = tf.placeholder(tf.int64, [None], 'labels')
        one_hot_labels = tf.one_hot(labels, 10)

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
        avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(
                        tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
        IZY_bound = math.log(10, 2) - class_loss
        IZX_bound = info_loss 

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                                decay_steps=2*steps_per_batch,
                                                decay_rate=0.97, staircase=True)
    else:
        print("Reloading existing")
        brand_new = False

        modelpath = './data-model/{}-beta{}-1500'.format(modelname, beta_num-1)
        graphfile = modelpath + '.meta'
        graph = tf.get_default_graph()

        saver = tf.compat.v1.train.import_meta_graph(graphfile)

        images = graph.get_tensor_by_name('images:0')
        labels = graph.get_tensor_by_name('labels:0')
        # train_tensor = graph.get_tensor_by_name('train_op/control_dependency:0')
        global_step = graph.get_tensor_by_name('global_step:0')
        IZY_bound = graph.get_tensor_by_name('sub:0')
        IZX_bound = graph.get_tensor_by_name('truediv_1:0')
        accuracy = graph.get_tensor_by_name('Mean_1:0')
        avg_accuracy = graph.get_tensor_by_name('Mean_3:0')
        BETA = float("1e-"+str(beta_num))
        info_loss = graph.get_tensor_by_name('truediv_1:0')
        class_loss = graph.get_tensor_by_name('truediv:0')
        total_loss = graph.get_tensor_by_name('add:0')
        learning_rate = graph.get_tensor_by_name('ExponentialDecay:0')
        # opt = graph.get_operation_by_name('encoder/fully_connected/weights/Adam')

        # ------ using collections:
        train_tensor = tf.get_collection('train_op')[0]
        total_loss = tf.get_collection('cost_op')[0]
        # vars = tf.get_collection('vars')[0]
        # ma_update = tf.get_collection('ma_update')[0]
        # opt = tf.get_collection('opt')[0]

    sess = tf.Session()
    if brand_new:
        opt = tf.train.AdamOptimizer(learning_rate, 0.5, name='opt')

        # ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
        # vars = tf.model_variables()
        # ma_update = ma.apply(vars)  
        
        saver = tf.train.Saver()
        # saver_polyak = tf.train.Saver(ma.variables_to_restore())

        train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                        global_step)
                                                        # update_ops=[ma_update])

        init = tf.global_variables_initializer()
        sess.run(init)
        tf.add_to_collection('train_op', train_tensor)
        tf.add_to_collection('cost_op', total_loss)
        # tf.add_to_collection('vars', vars)
        # tf.add_to_collection('ma_update', ma_update)
        # tf.add_to_collection('opt', opt)
    else:
        saver.restore(sess, modelpath)
        init = tf.global_variables_initializer()
        sess.run(init)
        # vars = tf.get_collection("variables")[0]
        # ma_update = ma.apply(vars) 
        train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                        global_step)

    def evaluate():
        IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
        return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc
    
    csvname = basepath+"/data-csv/{}-beta{}.csv".format(modelname, beta_num)
    dataname = basepath+'/data-model/{}-beta{}'.format(modelname, beta_num) 

    print("Training beginning for beta = 1e-{}...".format(beta_num))
    with open(csvname, "a") as f:
        for epoch in range(epoch_num):
            for step in range(steps_per_batch):
                im, ls = mnist_data.train.next_batch(batch_size)
                sess.run(train_tensor, feed_dict={images:im, labels:ls})
            print(epoch)
            print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(epoch, *evaluate()), file=f)
            sys.stdout.flush()
    f.close()
    savepth = saver.save(sess, dataname, global_step)
    # savepth = saver.save(sess, dataname)
    sess.close()




# %%
