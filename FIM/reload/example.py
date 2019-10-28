from __future__ import print_function
import tensorflow as tf
import os
import math
import numpy as np

output_dir = "/root/Data/temp"
model_checkpoint_file_base = os.path.join(output_dir, "model.ckpt")

input_length = 10
encoded_length = 3
learning_rate = 0.001
n_epochs = 10
n_batches = 10
if not os.path.exists(model_checkpoint_file_base + ".meta"):
    print("Making new")
    brand_new = True

    x_in = tf.placeholder(tf.float32, [None, input_length], name="x_in")
    W_enc = tf.Variable(tf.random_uniform([input_length, encoded_length],
                                          -1.0 / math.sqrt(input_length),
                                          1.0 / math.sqrt(input_length)), name="W_enc")
    b_enc = tf.Variable(tf.zeros(encoded_length), name="b_enc")
    encoded = tf.nn.tanh(tf.matmul(x_in, W_enc) + b_enc, name="encoded")
    W_dec = tf.transpose(W_enc, name="W_dec")
    b_dec = tf.Variable(tf.zeros(input_length), name="b_dec")
    decoded = tf.nn.tanh(tf.matmul(encoded, W_dec) + b_dec, name="decoded")
    cost = tf.sqrt(tf.reduce_mean(tf.square(decoded - x_in)), name="cost")

    saver = tf.train.Saver()
else:
    print("Reloading existing")
    brand_new = False
    saver = tf.train.import_meta_graph(model_checkpoint_file_base + ".meta")
    g = tf.get_default_graph()
    x_in = g.get_tensor_by_name("x_in:0")
    cost = g.get_tensor_by_name("cost:0")


sess = tf.Session()
if brand_new:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.add_to_collection("optimizer", optimizer)
else:
    saver.restore(sess, model_checkpoint_file_base)
    optimizer = tf.get_collection("optimizer")[0]

for epoch_i in range(n_epochs):
    for batch in range(n_batches):
        batch = np.random.rand(50, input_length)
        _, curr_cost = sess.run([optimizer, cost], feed_dict={x_in: batch})
        print("batch_cost:", curr_cost)
        save_path = tf.train.Saver().save(sess, model_checkpoint_file_base)