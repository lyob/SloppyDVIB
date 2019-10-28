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
beta_num = 0

while beta_num <= 13:
    print("brand_new?", brand_new)
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
        BETA = tf.get_variable(name="beta", initializer=1., dtype=tf.float32)
        total_loss = class_loss + BETA * info_loss
        
        # update_beta = tf.assign(BETA, float("1e-{}".format(beta_num)) )


        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
        avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(
                        tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
        IZY_bound = math.log(10, 2) - class_loss
        IZX_bound = info_loss 

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                                decay_steps=2*steps_per_batch,
                                                decay_rate=0.97, staircase=True)

        opt = tf.train.AdamOptimizer(learning_rate, 0.5)
        train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                        global_step)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        tf.add_to_collection('cost_op', total_loss)
        tf.add_to_collection('train_op', train_tensor)
        tf.add_to_collection('optimizer', opt)      
        # tf.add_to_collection('beta_updater', update_beta)     

    else:
        print("Reloading existing")

        modelpath = './data-model/{}-initial-0'.format(modelname)
        graphfile = modelpath + '.meta'
        graph = tf.get_default_graph()

        saver = tf.compat.v1.train.import_meta_graph(graphfile)


    with tf.Session() as sess:
        if brand_new:
            sess.run(init)

        else:
            global_step = graph.get_tensor_by_name('global_step:0')
            images = graph.get_tensor_by_name('images:0')
            labels = graph.get_tensor_by_name('labels:0')

            BETA = graph.get_tensor_by_name('beta:0')
            
            # update_beta = tf.get_collection('beta_updater')[0]
            
            train_tensor = tf.get_collection('train_op')[0]
            total_loss = tf.get_collection('cost_op')[0] # maybe use tf.assign here?
            opt = tf.get_collection('optimizer')[0]

            IZY_bound = graph.get_tensor_by_name('sub:0')
            IZX_bound = graph.get_tensor_by_name('truediv_1:0')
            accuracy = graph.get_tensor_by_name('Mean_1:0')
            avg_accuracy = graph.get_tensor_by_name('Mean_3:0')
            learning_rate = graph.get_tensor_by_name('ExponentialDecay:0')

            saver.restore(sess, modelpath)

            # update_beta = tf.assign(BETA, BETA/10.0)
            update_beta = tf.assign(BETA, float("1e-"+str(beta_num)) )
            sess.run(update_beta)
            print("--------------- beta Tensor value is:", BETA.eval())

        def evaluate():
            IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                    feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
            return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc
        
        csvinit = basepath+"/data-csv/{}-initial.csv".format(modelname)
        csvname = basepath+"/data-csv/{}-beta{}.csv".format(modelname, beta_num)
        datainit = basepath+"/data-model/{}-initial".format(modelname)
        dataname = basepath+'/data-model/{}-beta{}'.format(modelname, beta_num) 
        
        if brand_new:
            with open(csvinit, "a") as e:
                    for epoch in range(1):
                        print(epoch)
                        for step in range(1):
                            im, ls = mnist_data.train.next_batch(batch_size)
                            # sess.run(train_tensor, feed_dict={images:im,
                            #                                   labels:ls})
                    print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                        epoch, *evaluate()), file=e)
                    sys.stdout.flush()
            e.close()
            savepth = saver.save(sess, datainit, global_step)
            brand_new = False
        else:
            print("Training beginning for beta = 1e-{}...".format(beta_num))
            with open(csvname, "a") as f:
                for epoch in range(epoch_num): 
                    print(epoch)

                    if (epoch==0) or (epoch==50) or (epoch==200) or (epoch==500):
                        savepth = saver.save(sess,dataname,global_step)

                    for step in range(steps_per_batch):
                        im, ls = mnist_data.train.next_batch(batch_size)
                        sess.run(train_tensor, feed_dict={images:im, 
                                                          labels:ls})
                    print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                        epoch, *evaluate()), file=f)
                    sys.stdout.flush()
            f.close()

            savepth = saver.save(sess, dataname, global_step) # for epoch=1000
            beta_num += 1
        
        

        sess.close()
