'''
Load untrained model file and train it up to epoch = 200
'''

# global variables
modelname = 'unimodel'
basepath = '.'
dataset = 'test'

# ep_start should be 0, 200, 1000, 10000
ep_start = 1000
ep_end = 10000-ep_start
prev_eps = 200

## importing packages
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
print('All libraries imported')

# dataset
import small_mnist_full_offline as small_mnist
mnist_data = small_mnist.load_8x8_mnist()
print('All data imported')

for beta_num in range(0,13):
    print("----- New beta: beta = 1e-{} -----".format(beta_num))

    # get the model with epoch = 0
    modelpath = './data-model/{}-beta{}-{}-{}'.format(modelname, beta_num, dataset, (ep_start-prev_eps)*15)
    graphfile = modelpath + '.meta'

    graph0 = tf.Graph()
    with graph0.as_default():
        sess0 = tf.compat.v1.Session()


    with sess0 as sess:
        loader = tf.compat.v1.train.import_meta_graph(graphfile)
        loader.restore(sess, modelpath) # restores the graph

        images = graph0.get_tensor_by_name('images:0')
        labels = graph0.get_tensor_by_name('labels:0')
        IZY_bound = graph0.get_tensor_by_name('sub:0')
        IZX_bound = graph0.get_tensor_by_name('truediv_1:0')
        accuracy = graph0.get_tensor_by_name('Mean_1:0')
        avg_accuracy = graph0.get_tensor_by_name('Mean_3:0')

        batch_size = 100
        steps_per_batch = int(mnist_data.train.num_examples / batch_size)

        BETA = float("1e-"+str(beta_num))
        info_loss = graph0.get_tensor_by_name('truediv_1:0')
        class_loss = graph0.get_tensor_by_name('truediv:0')
        total_loss = graph0.get_tensor_by_name('add:0')


        global_step = graph0.get_tensor_by_name('global_step:0')
        learning_rate = graph0.get_tensor_by_name('ExponentialDecay:0')
        # opt = tf.compat.v1.train.AdamOptimizer(learning_rate, 0.5)
        opt = graph0.get_operation_by_name('encoder/fully_connected/weights/Adam')


        # EMA stuff
        # ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
        # ma = graph0.get_operation_by_name('encoder/fully_connected/weights/ExponentialMovingAverage')
        # vars = tf.model_variables()
        # ma_update = ma.apply(vars)
        # ma_update = graph0.get_operation_by_name('encoder/fully_connected/weights/ExponentialMovingAverage')

        # train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                # global_step,
                                                # update_ops=[ma_update])

        ## saves the model/data
        saver = tf.compat.v1.train.Saver()
        # saver_polyak = tf.compat.v1.train.Saver(ma.variables_to_restore()) 

        train_tensor = graph0.get_tensor_by_name('train_op/control_dependency:0')

        tf.global_variables_initializer().run()

        def evaluate():
            IZY, IZX, acc, avg_acc = sess.run([IZY_bound, IZX_bound, accuracy, avg_accuracy],
                        feed_dict={images: mnist_data.test.images, labels: mnist_data.test.labels})
            return IZY, IZX, acc, avg_acc, 1-acc, 1-avg_acc

        csvname = basepath+"/data-csv/{}-beta{}-{}.csv".format(modelname, beta_num, dataset)
        dataname = basepath+'/data-model/{}-beta{}-{}'.format(modelname, beta_num, dataset)

        print('Training beginning for beta=1e-{}.'.format(beta_num))
        with open(csvname, "a") as f:
            for epoch in range(ep_end): # set the number of epochs here
                for step in range(steps_per_batch):
                    im, ls = mnist_data.train.next_batch(batch_size)
                    sess.run(train_tensor, feed_dict={images: im, labels: ls})
                print(epoch)
                print("{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                    epoch, *evaluate()), file=f)
                sys.stdout.flush()
        f.close()
        print('Training complete for beta=1e-{}.'.format(beta_num))
        savepth = saver.save(sess, dataname, global_step) ## data is stored here
        print('Checkpoint saved')

        sess.close()
