import sys
import json
sys.path.insert(0, "../src")

import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network2

def main():
    filename = 'test'
    # load data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # example
    epochs = 5
    net = network2.Network([784, 10], cost=network2.CrossEntropyCost)
    vc, va, tc, ta = net.SGD(training_data=training_data, epochs=epochs, mini_batch_size=100, eta=0.1, lmbda = 0.1, reg = 2,
        evaluation_data=validation_data, 
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

if __name__ == "__main__":
    main()