import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import math

import time



center_box=(0, 100.0)
cluster_std=2.0
n_features = 2
n_samples = 200
n_classes = 4

def euclidean_distance(first_point, second_point):
    if first_point.shape != second_point.shape:
        print("first_point.shape != second_point.shape")
        return
    sum = 0.0
    for dim in range(len(first_point)):
        sum += (first_point[dim] - second_point[dim])**2

    return math.sqrt(sum)


def make_dataset(center_box=(0, 100.0), cluster_std=1.0,
                 n_features = 2, n_samples = 100000, n_classes = 100,
                 linear_separable=True):
    generator = np.random.RandomState(420)
    classes = generator.uniform(center_box[0], center_box[1], size=(n_classes, n_features))
    n_samples_per_class = [int(n_samples // n_classes)] * n_classes
    for i in range(n_samples % n_classes):
        n_samples_per_class[i] += 1
    cluster_std = np.full(len(classes), cluster_std)
    X = np.zeros((0,n_features))
    y = np.zeros((0))
    for i, (n, std) in enumerate(zip(n_samples_per_class, cluster_std)):
        if(i == 0):
            X = np.append(X, generator.normal(loc=classes[i], scale=std,
                                      size=(n, n_features)), axis=0)
            y = np.append(y, np.full(n,i), axis=0)
        else:
            # check for linear sep
            new_data = generator.normal(loc=classes[i], scale=std,
                                        size=(n, n_features))
            new_labels = np.full(n, i)
            new_data_center = classes[i]
            for target_class in range(i):
                target_data_center = classes[target_class]

                if(euclidean_distance(target_data_center, new_data_center) > 5.0):
                    continue

                target = X[(y == target_class)]
                target_labels = y[(y == target_class)]
                x_ = np.concatenate((target, new_data))
                y_ = np.concatenate((target_labels, new_labels))
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                x_ = sc.fit_transform(x_)
                from sklearn.linear_model import Perceptron
                perceptron = Perceptron(random_state=0)
                perceptron.fit(x_, y_)
                predicted = perceptron.predict(x_)
                was_unsep = False
                for p in range(len(predicted)):
                    if p < len(target_labels):
                        if predicted[p] == i:
                            if(linear_separable):
                                target_labels[p] = i
                                was_unsep = True
                    else:
                        if predicted[p] == target_class:
                            if(linear_separable):
                                new_labels[p - len(target_labels)] = target_class
                                was_unsep = True

                if was_unsep:
                    print((target_data_center ,new_data_center))
                    print(euclidean_distance(target_data_center ,new_data_center))

                y[(y == target_class)] = target_labels


            X = np.append(X, new_data, axis=0)
            y = np.append(y, new_labels, axis=0)

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y*5, s=25, edgecolor='k')
    plt.show()
    # print(X, y)

start = time.time()
make_dataset(linear_separable=True)
end = time.time()
# print(end - start)
