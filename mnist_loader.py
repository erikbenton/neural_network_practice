import pickle
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
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    # Return (training_data, validation_data, test_data)
    # Based on the load_data, but a more convenient format
    #
    # training_data is a list of 50,000 2-tuples (x, y)
    # x - 784 dimensional numpy ndarray containing the input image
    # y - 10 dimensional numpy ndarray with corresponding classification, or correct digit, for x
    #
    # Therefore the format for the training_data and the validation_data/test_data
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data


def vectorized_result(j):
    # Returns a 10 dimensional unit vector with a 1 in the jth position and 0s elsewhere
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

