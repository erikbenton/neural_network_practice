import numpy as np
import math
import matplotlib as plt
import mnist_loader
import random
import json
import sys


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1 - y)*np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return np.subtract(a, y)


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        return np.subtract(a - y) * sigmoid_prime(z)


class Network:
    def __init__(self, sizes, cost=CrossEntropyCost):
        # Number of neural layers in the network
        self.num_layers: int = len(sizes)
        # Number of input neurons
        self.sizes: int = sizes
        self.biases: list = []
        self.weights: list = []
        self.default_weight_initializer()
        self.cost = cost
        return

    def default_weight_initializer(self):
        # Initializes the bias with a Gaussian random distribution with mean 0 and SD == 1
        # No biases are set for the input layer
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # Initializes the weights as a Gaussian distribution
        # with a mean of 0 and an SD == 1/sqrt(num_weights_to_same_neuron)
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in (self.sizes[:-1], self.sizes[1:])]
        return

    def large_weight_initializer(self):
        # Initializes the bias with a Gaussian random distribution with mean 0 and SD == 1
        # No biases are set for the input layer
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # Initializes the weights as a Gaussian distribution
        # with a mean of 0 and an SD == 1
        self.weights = [np.random.randn(y, x) for x, y in (self.sizes[:-1], self.sizes[1:])]
        return

    def feed_forward(self, a):
        # Creating a list of the zipped tuples
        zipped_biases_weights = list(zip(self.biases, self.weights))
        for b, w in zipped_biases_weights:
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        if evaluation_data:
            n_test = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_matrix(mini_batch, learning_rate, lmbda, len(training_data))
            print("Epoch {0} training complete".format(j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {0}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on the training data: {0} / {1}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on the evaluation data: {0} / {1}".format(accuracy, n_test))
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, n):
        # nabla - the upside-down greek Delta
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        xs = np.array([x[0] for x in mini_batch])
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backwards_propagation(x, y)
            zipped_nablas_b = list(zip(nabla_b, delta_nabla_b))
            zipped_nablas_w = list(zip(nabla_w, delta_nabla_w))
            nabla_b = [nb + dnb for nb, dnb in zipped_nablas_b]
            nabla_w = [nw + dnw for nw, dnw in zipped_nablas_w]
        zipped_biases_nabla_b = list(zip(self.biases, nabla_b))
        zipped_weights_nabla_w = list(zip(self.weights, nabla_w))
        self.weights = [(1-learning_rate*(lmbda/n))*w - (learning_rate/len(mini_batch))*nw
                        for w, nw in zipped_weights_nabla_w]
        self.biases = [b - (learning_rate/len(mini_batch))*nb
                       for b, nb in zipped_biases_nabla_b]
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
            fun = np.tile(w, (1, 1, 10))
            z = np.dot(w, activation) + b
            fun = np.array(w)
            fun = np.tile(fun, (1, 10))
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
        return nabla_b, nabla_w

    def update_mini_batch_matrix(self, mini_batch, learning_rate, lmbda, n):
        # nabla - the upside-down greek Delta
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        xs = np.array([x for x, y in mini_batch]).reshape(784, 10)
        ys = np.array([y for x, y in mini_batch]).reshape(10, 10)
        delta_nabla_b, delta_nabla_w = self.backwards_propagation_matrix(xs, ys)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backwards_propagation(x, y)
            zipped_nablas_b = list(zip(nabla_b, delta_nabla_b))
            zipped_nablas_w = list(zip(nabla_w, delta_nabla_w))
            nabla_b = [nb + dnb for nb, dnb in zipped_nablas_b]
            nabla_w = [nw + dnw for nw, dnw in zipped_nablas_w]
        zipped_biases_nabla_b = list(zip(self.biases, nabla_b))
        zipped_weights_nabla_w = list(zip(self.weights, nabla_w))
        self.weights = [(1-learning_rate*(lmbda/n))*w - (learning_rate/len(mini_batch))*nw
                        for w, nw in zipped_weights_nabla_w]
        self.biases = [b - (learning_rate/len(mini_batch))*nb
                       for b, nb in zipped_biases_nabla_b]
        return

    def backwards_propagation_matrix(self, xs, ys):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed forward
        activation = xs
        # Layer by layer list of the activations
        activations = [xs]
        # Layer by layer list to store all the z vectors
        zs = []
        for b, w in zip(self.biases, self.weights):
            # Make the bias matrix the appropriate size
            biases = np.tile(b, (1, np.shape(xs)[1]))
            biases.reshape(np.shape(xs)[1], len(b))
            z = np.matmul(w, activation) + biases
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        delta = self.cost.delta(zs[-1], activations[-1], ys)
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.matmul(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.matmul(delta, activations[-i - 1].transpose())
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        # Returns the number of inputs in data that the neural network interpreted correctly
        # Neural Network's output is assumed to be the index of the whichever neuron in the final layer
        # has the highest activation
        if convert:
            results = [(np.argmax(self.feed_forward(x)), np.argmax(y)) for x, y in data]
        else:
            results = [(np.argmax(self.feed_forward(x)), y) for x, y in data]
        return sum(int(x == y) for x, y in results)

    def total_cost(self, data, lmbda, convert=False):
        # Returns the total cost for the data set
        cost: float = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        # Save neural network to 'filename'
        data = {"sizes": self.sizes,
                "weights": [list(w) for w in self.weights],
                "biases": [list(b) for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        return

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def load(filename):
    # Load neural network from file
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    # Gives a 10 dim unit vector with 1 in the jth place and zeros elsewhere
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_matrix(z):
    denominator = np.add(np.ones(z.shape), np.exp(-z))
    return np.divide(np.ones(z.size), denominator)


def cost_derivative(output_activations, y):
    return np.subtract(output_activations, y)


def cost_derivative_matrix(output_activations, y):
    return np.subtract(output_activations, y)


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_prime_matrix(z):
    return np.multiply(sigmoid_matrix(z), np.subtract(np.ones(z.shape), sigmoid_matrix(z)))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])
net.sgd(training_data, 30, 10, 0.5,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True)
