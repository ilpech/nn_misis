import sys
import os
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(first_point, second_point):
    if first_point.shape != second_point.shape:
        print("first_point.shape != second_point.shape")
        return
    sum = 0.0
    for dim in range(len(first_point)):
        sum += (first_point[dim] - second_point[dim])**2
    return math.sqrt(sum)


def nonlin_sigmoid(x, deriv=False):
    if (deriv == True):

        return nonlin_sigmoid(x)*(1 - nonlin_sigmoid(x))
    return 1 / (1 + np.exp(-x))


def draw_dataset(X,y):
    try:
        if X.shape[1] > 2:
            raise ValueError
    except ValueError:
        print("Drawing available only for 2 dim space")
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.show()

def read_data_csv(path):
    try:
        if not os.path.isfile(path):
            raise NameError
    except NameError:
        print('Check csv path', path)
    data = np.genfromtxt(path, delimiter=',')
    X = data[:,:-1]
    y = data[:,-1:]
    return X, y
