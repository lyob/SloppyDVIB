'''
Plotting the values of the decoder in histogram.
'''

#%%
import numpy as np
import tensorflow as tf



tf.reset_default_graph()


# Turn on xla optimization
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.InteractiveSession(config=config)


#%%
modelPath = './DATA/mnistvib-12000' # trained model (graph/meta, index, data/weights)
graphfile = './DATA/mnistvib-12000.meta' # the model used in training
loader = tf.train.import_meta_graph(graphfile)
loader.restore(sess, modelPath) 


#%%
saver2 = tf.train.import_meta_graph(graphfile)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

#%%

# log_dir = "./log_dir/"
# tf.train.write_graph(input_graph_def, logdir=log_dir, name=out_file, as_text=True)
from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(modelPath)
var_to_shape_map = reader.get_variable_to_shape_map()
tensor_values = {}
for key in sorted(var_to_shape_map):
    # print(key)
    key_str = str(key)
    if ("decoder/fully_connected/" in key_str):
        tensor_val = reader.get_tensor(key).tolist()
        tensor_values[key_str] = tensor_val
    else:
        break
    ## tensor_values.update( key_str: tensor_val )
out = tensor_values['decoder/fully_connected/weights']



np.histogram(out)


#%%
from matplotlib import pyplot as plt 
plt.hist(out)
plt.title("histogram") 
plt.show()

#%%
