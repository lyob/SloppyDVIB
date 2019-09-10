#%%
# global variables
modelname = 'indep'
labelname = 'l72'
h1 = 100
h2 = h1
bneck = 32
basepath = '.'

## importing packages
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
import json
print('All libraries imported')

# dataset
import small_mnist_one_image_offline_l72 as small_mnist
mnist_data = small_mnist.load_8x8_mnist()

print('All data imported')


#%%
