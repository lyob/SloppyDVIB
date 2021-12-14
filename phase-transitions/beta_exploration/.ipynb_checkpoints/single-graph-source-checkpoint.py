# script for using the same TF meta files for each beta and epoch run
# flow:  get initial meta file, then use this as a starting point for every beta training run. 

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
epoch_num = 1000
batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)

brand_new = True
beta_index = 24

# while beta_num < 13:
while beta_index >= 0:
    print("brand_new?", brand_new)
    beta_num = beta_index / 2
    print("beta_num:", beta_num)

    if brand_new == True:
        print('Making new')

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

        BETA = tf.Variable(1e-12, name='beta', trainable=False)
        total_loss = tf.add(class_loss, tf.multiply(BETA, info_loss), name="cost_op")

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
        avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(
                        tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
        IZY_bound = math.log(10, 2) - class_loss
        IZX_bound = info_loss 

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                                decay_steps=2*steps_per_batch,
                                                decay_rate=0.97, staircase=True)

        opt = tf.train.AdamOptimizer(learning_rate, 0.5, name="optimizer")
        train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                        global_step)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        tf.add_to_collection('train_op', train_tensor)

    else:
        print("Reloading existing")

        tf.reset_default_graph()

        modelpath = './data-model/{}-initial-0'.format(modelname)
        graphfile = modelpath + '.meta'
        graph = tf.get_default_graph()
        print("NUMBER OF KEYS IN THIS GRAPH:::", len(sess.graph._nodes_by_name.keys()))

        saver = tf.compat.v1.train.import_meta_graph(graphfile)


    with tf.Session() as sess:
        if brand_new:
            sess.run(init)

            def evaluate():
                IZY, IZX, acc, avg_acc, tot_loss = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy, total_loss],
                        feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
                return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc, tot_loss

        else:
            global_step = graph.get_tensor_by_name('global_step:0')
            images = graph.get_tensor_by_name('images:0')
            labels = graph.get_tensor_by_name('labels:0')

            BETA = graph.get_tensor_by_name('beta:0')
            
            train_tensor = tf.get_collection('train_op')[0]
            total_loss = graph.get_tensor_by_name('cost_op:0')

            opt = graph.get_tensor_by_name('optimizer:0')
            IZY_bound = graph.get_tensor_by_name('sub:0')
            IZX_bound = graph.get_tensor_by_name('truediv_1:0')
            accuracy = graph.get_tensor_by_name('Mean_1:0')
            avg_accuracy = graph.get_tensor_by_name('Mean_3:0')
            learning_rate = graph.get_tensor_by_name('ExponentialDecay:0')

            saver.restore(sess, modelpath)

            # update_beta = tf.assign(BETA, float("1e-"+str(beta_num)) )
            update_beta = tf.assign(BETA, 10**(-beta_num) )
            sess.run(update_beta)
            print("--------------- beta Tensor value is:", BETA.eval())

            def evaluate():
                IZY, IZX, acc, avg_acc, tot_loss, beta_out = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy, total_loss, BETA],
                        feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
                return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc, tot_loss, beta_out
        
        csvinit = basepath+"/data-csv/{}-initial.csv".format(modelname)
        csvname = basepath+"/data-csv/{}-beta{}.csv".format(modelname, beta_num)
        datainit = basepath+"/data-model/{}-initial".format(modelname)
        dataname = basepath+'/data-model/{}-beta{}'.format(modelname, beta_num) 
        
        if brand_new:
            with open(csvinit, "a") as e:
                    for epoch in range(1):
                        print(epoch)
                        # for step in range(1):
                        #     im, ls = mnist_data.train.next_batch(batch_size)
                        #     sess.run(train_tensor, feed_dict={images:im,
                        #                                       labels:ls})
                    print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{}".format(
                        epoch, *evaluate()), file=e)
                    sys.stdout.flush()
            e.close()
            savepth = saver.save(sess, datainit, global_step)
            brand_new = False
        else:
            print("Training beginning for beta = 1e-{}...".format(beta_num))
            with open(csvname, "a") as f:
                for epoch in range(epoch_num): 
                    if epoch % 100 == 0:
                        print(epoch)

                    if (epoch==0) or (epoch==50) or (epoch==200) or (epoch==500):
                        savepth = saver.save(sess,dataname,global_step)

                    for step in range(steps_per_batch):
                        im, ls = mnist_data.train.next_batch(batch_size)
                        sess.run(train_tensor, feed_dict={images:im, 
                                                          labels:ls})
                    print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{}".format(
                        epoch, *evaluate()), file=f)
                    sys.stdout.flush()
            f.close()

            savepth = saver.save(sess, dataname, global_step) # for epoch=1000
            beta_index -= 1

        sess.close()
