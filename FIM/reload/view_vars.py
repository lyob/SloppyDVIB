#%%

import tensorflow as tf

beta_num = 1
modelname = 'iterated'
modelpath = './data-model/{}-beta{}-1500'.format(modelname, beta_num-1)


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(modelpath+'.meta')
    saver.restore(sess, modelpath)

#%%
