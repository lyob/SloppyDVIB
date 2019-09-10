import tensorflow as tf
import numpy as np
import math
import pickle
from tensorflow.python.framework import dtypes
# from sklearn import datasets

# Defining the dataset class
class Dataset(object):

    def __init__(self, images, labels, dtype=dtypes.float32):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
            dtype)

        assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


## Importing 8x8 mnist data from scikit-learn
def read_data_sets():
    # data from the scikit-learn package = 8x8 images of digits
    # digits = datasets.load_digits()
    digits = pickle.load(open('./DATA/dataset.p', "rb"))

    # training and test sets 
    X_digits = digits.data / digits.data.max() # normalises 16-level sensitivity to 0to1 
    y_digits = digits.target

    # In the tensorflow-approved formats
    X_digits = np.array(X_digits, dtype='float32')
    y_digits = np.uint8(y_digits)

    # parameters
    n_samples = len(X_digits)
    print('Total number of samples (training + test):', n_samples)
    train_size = int(.85 * n_samples) # determines the size of the training set

    #################### Simple partitioning of test and training sets
    # images and labels for training and testing
    train_images = X_digits[:train_size]
    print("train images size", train_images.shape)
    train_labels = y_digits[:train_size]

    # Changing this to input only one image into the network
    test_images = X_digits[train_size:train_size+1]
    print("test images size", test_images.shape)
    test_labels = y_digits[train_size:train_size+1]


    #################### Chunking test and training sets 
    # buff = math.ceil(n_samples/10) # size of the buffer 

    # # using different parts of the training data:
    # first = n * buff
    # last = n * buff + train_size
    # if last > n_samples:
    #     print("the train set extends over the n_sample size")
    #     train_images = np.concatenate((X_digits[first:n_samples], X_digits[:last-n_samples]), axis=0)
    #     train_labels = np.concatenate((y_digits[first:n_samples], y_digits[:last-n_samples]), axis=0)

    #     test_images = X_digits[last-n_samples:first]
    #     test_labels = y_digits[last-n_samples:first]
    # else:
    #     print("the train set is within the n_sample size")
    #     train_images = X_digits[first:last]
    #     train_labels = y_digits[first:last]

    #     test_images = np.concatenate((X_digits[last:n_samples], X_digits[:first]), axis=0)
    #     test_labels = np.concatenate((y_digits[last:n_samples], y_digits[:first]), axis=0)


    print("size of train_images:", train_images.shape)
    print("size of train_labsls:", train_labels.shape)

    print("size of test_images:", test_images.shape)
    print("size of test_labels:", test_labels.shape)

    train = Dataset(train_images, train_labels)
    test = Dataset(test_images, test_labels)

    return namedDataset(train=train, test=test)

# required for naming the train and test datasets below
import collections
namedDataset = collections.namedtuple('Dataset', ['train', 'test'])

def load_8x8_mnist():
    return read_data_sets()