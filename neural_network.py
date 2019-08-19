import numpy as np
import math
import matplotlib as plt
import mnist_loader
import random


class Network:
    def __init__(self, sizes):
        # Number of neural layers in the network
        self.num_layers: int = len(sizes)
        # Number of input neurons
        self.sizes: int = sizes
        # Biases and weights for the network
        # Normal distribution
        self.biases: list = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights: list = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        return

    def feed_forward(self, a):
        # Creating a list of the zipped tuples
        zipped_biases_weights = list(zip(self.biases, self.weights))
        for b, w in zipped_biases_weights:
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
        return

    def update_mini_batch(self, mini_batch, learning_rate):
        # nabla - the upside-down greek Delta
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backwards_propagation(x, y)
            zipped_nablas_b = list(zip(nabla_b, delta_nabla_b))
            zipped_nablas_w = list(zip(nabla_w, delta_nabla_w))
            nabla_b = [nb + dnb for nb, dnb in zipped_nablas_b]
            nabla_w = [nw + dnw for nw, dnw in zipped_nablas_w]
        zipped_biases_nabla_b = list(zip(self.biases, nabla_b))
        zipped_weights_nabla_w = list(zip(self.weights, nabla_w))
        self.biases = [b - (learning_rate/len(mini_batch)) * nb for b, nb in zipped_biases_nabla_b]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zipped_weights_nabla_w]
        return

    def backwards_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed forward
        activation = x
        # Layer by layer list of the activations
        activations = [x]
        # Layer by layer list to store all the z vectors
        zs = []
        zipped_biases_weights = list(zip(self.biases, self.weights))
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return nabla_b, nabla_w

    def backwards_propagation_matrix(self, xs, y, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations = [xs]
        mbatch_activations = [activations]
        zs = []
        mbatch_zs = []
        for b, w in zip(self.biases, self.weights):
            # Create bias matrix
            bias_matrix = np.tile(self.biases, (n, 1))
            # Calculate all mini batch weighted inputs for one layer
            zs.append(np.add(np.matmul(w, activations), bias_matrix))
            # Empty activations
            activations = []
            # Calculate all the activations
            for z in zs:
                activations.append(sigmoid(z))
            # Save all the activations for that mini batch
            mbatch_activations.append(activations)
            mbatch_zs.append(zs)
        # Backward pass
        delta = np.multiply(cost_derivative_matrix(mbatch_activations[-1], y),
                            sigmoid_prime_matrix(np.array(mbatch_zs[-1])))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_matrix(z):
    denominator = np.add(np.ones(z.shape), np.exp(-z))
    return np.divide(np.ones(z.size), denominator)


def cost_derivative(output_activations, y):
    return np.subtract(output_activations - y)


def cost_derivative_matrix(output_activations, y):
    return np.subtract(output_activations, y)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_prime_matrix(z):
    return np.multiply(sigmoid_matrix(z), np.subtract(np.ones(z.shape), sigmoid_matrix(z)))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])
net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
