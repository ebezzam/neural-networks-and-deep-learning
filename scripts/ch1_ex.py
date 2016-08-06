import sys
sys.path.insert(0, "../src")

import mnist_loader
import network

# load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# example
net = network.Network([784, 30, 10])
net.SGD(training_data, 5, 15, 3.0, test_data=test_data)