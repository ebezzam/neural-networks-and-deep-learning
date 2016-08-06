import sys
sys.path.insert(0, "../src")

import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 30, 30, 10])
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)