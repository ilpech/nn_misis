import sys
import os
import time
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt

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

class EngFunc:
    def __init__(self):
        self.upper_border = 511

    def value(self, x, y):
        sqrt1 = sqrt(abs(x/2 + (y+47)))
        sqrt2 = sqrt(abs(x - (y+47)))
        return -(y+47) * sin(sqrt1) - x * sin(sqrt2)

    def true_value(self):
        return self.value(512, 404.2319)

    def err(self, x, y):
        return abs(self.value(x,y) - self.true_value())

class SphereFunc:
    def __init__(self):
        self.upper_border = 8
    N = 10
    def value(self, x, y):
        res = 0
        res += x ** 2
        res += y ** 2
        return float(res)

    def true_value(self):
        return self.value(0, 0)

    def err(self, x, y):
        return abs(self.value(x,y) - self.value(0,0))

class ShapherN2Func:
    def __init__(self):
        self.upper_border = 128
    def value(self, x, y):
        up = sin(x**2 + y**2) ** 2 - 0.5
        bottom = (1 + 0.001 * (x**2 + y**2)) ** 2
        return 0.5 * up/bottom

    def true_value(self):
        return self.value(0,0)

    def err(self, x, y):
        return abs(self.value(x,y) - self.value(0,0))

class RosenbrockFunc:
    def __init__(self):
        self.upper_border = 16
    def value(self, x, y):
        return float((1-x)**2 + 100*(y-x**2)**2)
    def true_value(self):
        return self.value(1, 1)

    def err(self, x, y):
        return abs(self.value(x,y) - self.value(1,1))

class BillFunc:
    def __init__(self):
        self.upper_border = 5
    def value(self, x, y):
        r1 = (1.5 - x + x*y)**2
        r2 = (2.25 - x + x*(y**2))**2
        r3 = (2.625 - x + x*(y**3))**2
        return r1 + r2 + r3
    def true_value(self):
        return self.value(3, .5)
    
    def err(self, x, y):
        return abs(self.value(x,y) - self.true_value())

class CamelFunc:
    def __init__(self):
        self.upper_border = 8
    def value(self, x, y):
        r1 = 2 * (x**2)
        r2 = 1.05 * (x**4)
        r3 = (1/6) * (x**6)
        r4 = x*y
        r5 = y **2
        return r1 - r2 + r3 + r4 + r5

    def true_value(self):
        return self.value(0, 0)
    
    def err(self, x, y):
        return abs(self.value(x,y) - self.true_value())

class BootFunc:
    def __init__(self):
        self.upper_border = 10
    def value(self, x, y):
        r1 = (x + 2 * y - 7)**2
        r2 = (2*x + y - 5)**2
        return r1 + r2

    def true_value(self):
        return self.value(1, 3)
    
    def err(self, x, y):
        return abs(self.value(x,y) - self.true_value())    


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


from matplotlib import pyplot as plt
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import scipy.io as sio


def load_run_viz(dataset_name, sizes=[4,5,6], EPOCHS = 500):
    #load and prepare data
    data = sio.loadmat(dataset_name)["data"]
    clusters = sio.loadmat(dataset_name)["clusters"][0]
    CLUSTERS_NUMBER = clusters.max()
    
    def vectorize(cluster):
        z = np.zeros(CLUSTERS_NUMBER)
        z[cluster - 1] = 1.0
        return z
    
    sparse_clusters = np.array([vectorize(clusters[i]) for i in range(data.shape[0])])
    
    #split data to train and test
    [x_train, x_test, y_train, y_test] = train_test_split(data, sparse_clusters)
    
    #train Perceptrone
    my_network = Mlp([len(data[0])] + sizes + [CLUSTERS_NUMBER])
    my_network.train(x_train, y_train, EPOCHS)
    
    #test & log MSE
    y_pred = my_network.predict(x_test)
    print("Среднеквадратичная ошибка: ", mean_squared_error(y_pred, y_test))
    print("Точность: ", accuracy_score(np.argmax(y_pred, 1), np.argmax(y_test, 1)))
    
    #visualize grid from perceptrone's view and layout _right_ dataset on it
    X, Y = np.mgrid[-80:80:100j, -80:80:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    points = [[positions[0][x], positions[1][x]] for x in range(positions[0].size)]

    X = np.array(points)
    Z = [np.argmax(n) for n in my_network.predict(np.array(points))]
    
    

    plt.scatter(X[:,0], X[:,1], c=Z)
    plt.scatter(data[:,0], data[:,1], c=clusters)