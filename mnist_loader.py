import cPickle
import gzip
import numpy as np


def load_data():
    # Returns the MNIST data as (training_data, validation_data, test_data)
    # The training_data is tuple with 2 entries
    #
    # First entry has the actual training images used, which is a 50,000 entry numpy ndarray
    #   where each entry is a numpy ndarray of 784 values (28px * 28px = 784px) - Input layer
    #
    # Second entry is a 50,000 entry numpy ndarray containing the actual digit (0, ..., 9) value for the
    #   first entry's 50,000 entries
    #
    # The validation_data and the test_data are similar to the above, but only 10,000 images
    #
    # This is a nice way to format the data here, but can be difficult to deal with training_data for the
    # backwards_propagation so the load_data_wrapper function modifies it slightly
    f = gzip.open('../neural_network_data/neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data
