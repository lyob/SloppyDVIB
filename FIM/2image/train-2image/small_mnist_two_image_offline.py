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

# returns an array of positions in the original array that have a specific label value
def find_specific_label(label, array):
    # pos = np.array([])
    pos = np.where((array==label[0]) | (array==label[1]))
    # i = np.where(array==label[0])
    # pos = np.array(pos, i)
    # print(pos[0])
    return pos[0].astype(int)


## Importing 8x8 mnist data from scikit-learn
def read_data_sets():
    # data from the scikit-learn package
    digits = pickle.load(open('./data-mnist/dataset.p', "rb"))

    # training and test sets 
    X_digits = digits.data / digits.data.max() # normalises 16-level sensitivity to 0to1 
    y_digits = digits.target

    # In the tensorflow-approved formats
    X_digits = np.array(X_digits, dtype='float32')
    y_digits = np.uint8(y_digits)

    # parameters
    n_samples = len(X_digits)
    print(n_samples)
    train_size = int(.85 * n_samples) # determines the size of the training set

    #################### Simple partitioning of test and training sets
    # images and labels for training and testing
    train_images = X_digits[:train_size]
    print("train images size", train_images.shape)
    train_labels = y_digits[:train_size]

    test_images = X_digits[train_size:]
    print("test images size", test_images.shape)
    test_labels = y_digits[train_size:]

    # select only two labels (6 and 7) from the train dataset
    label_pos = find_specific_label([6,7], train_labels)

    train_images2 = np.array([])
    train_labels2 = np.zeros(len(label_pos))

    for i, val in enumerate(label_pos):
        if i == 0:
            train_images2 = train_images[val]
        else:
            train_images2 = np.vstack((train_images2, train_images[val]))
        train_labels2[i] = train_labels[val]

    print("size of train_images:", train_images.shape)
    print("size of train_labels:", train_labels.shape)

    print("size of train_images2:", train_images2.shape)
    print("size of train_labels2:", train_labels2.shape)

    print("size of test_images:", test_images.shape)
    print("size of test_labels:", test_labels.shape)

    train = Dataset(train_images2, train_labels2)
    test = Dataset(test_images, test_labels)

    return namedDataset(train=train, test=test)

# required for naming the train and test datasets below
import collections
namedDataset = collections.namedtuple('Dataset', ['train', 'test'])

def load_8x8_mnist():
    return read_data_sets()

load_8x8_mnist()