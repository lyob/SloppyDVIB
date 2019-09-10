#%%
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math
import sys
print('All libraries imported')
#%%
# dataset
import small_mnist_full_offline as small_mnist
mnist_data = small_mnist.load_8x8_mnist()

print('dataset imported')

# list of labels
labels = mnist_data.test.labels
num_labels = len(labels)
# print(type(labels))
print('There are {} labels in total.'.format(num_labels))

n = 10
print('The first {} labels are'.format(n), labels[0:n])

def find_specific_label(array, label):
    pos = np.array([])
    i = np.where(array==7)
    pos = np.append(pos, i)
    return pos.astype(int)

find_specific_label(labels, 7)

# there is a train dataset and test dataset, of size 1527 and 270 respectively.
# We are looking for the label 7 among the test dataset only

#%%